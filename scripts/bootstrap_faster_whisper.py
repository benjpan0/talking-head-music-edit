"""faster-whisper 跨机隔离引导安装与国内镜像按需预热脚本（Option A）。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import venv
from pathlib import Path
from typing import Any, Dict, List, Optional

# 强制终端使用 UTF-8 编码，防止 Windows 控制台输出乱码
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from portable_paths import (  # noqa: E402
    get_skill_cache_dir,
    get_whisper_models_cache_dir,
    get_whisper_venv_python,
    python_has_module,
    resolve_local_whisper_model,
    resolve_whisper_python,
)

DEFAULT_TUNA_PYPI = "https://pypi.tuna.tsinghua.edu.cn/simple"
DEFAULT_HF_MIRROR = "https://hf-mirror.com"


def _run_visible_pip(
    python_exe: Path,
    packages: List[str],
    pip_index: Optional[str],
    fallback_official: bool = True,
) -> str:
    """运行 pip install（实时输出到控制台），若指定国内镜像失败且允许回退则重试官方源。"""
    base_cmd = [str(python_exe), "-m", "pip", "install", "--upgrade"]

    if pip_index:
        cmd = [*base_cmd, "-i", pip_index, *packages]
        print(f"[pip] 执行：{' '.join(cmd)}")
        proc = subprocess.run(cmd)
        if proc.returncode == 0:
            return pip_index
        if not fallback_official:
            raise RuntimeError(f"pip 使用镜像 {pip_index} 安装失败（退出码 {proc.returncode}）")
        print(f"[提示] 镜像 {pip_index} 安装失败，自动回退默认 PyPI 官方源重试...")

    cmd_official = [*base_cmd, *packages]
    print(f"[pip] 执行：{' '.join(cmd_official)}")
    proc2 = subprocess.run(cmd_official)
    if proc2.returncode != 0:
        raise RuntimeError(f"pip 安装失败（退出码 {proc2.returncode}）：{packages}")
    return "default"


def ensure_faster_whisper(
    venv_dir: Optional[Path] = None,
    install_here: bool = False,
    pip_index: str = DEFAULT_TUNA_PYPI,
    fallback_official: bool = True,
    with_cuda_libs: bool = False,
    force_reinstall: bool = False,
    download_model_name: Optional[str] = None,
    model_dir: Optional[Path] = None,
    hf_endpoint: str = DEFAULT_HF_MIRROR,
) -> Dict[str, Any]:
    """
    确保本机存在可用的 faster-whisper 运行环境；默认安装到外部缓存隔离 venv（不污染技能目录与 Obsidian 仓库）。
    可选同时从 hf-mirror 预下载指定模型（如 small）。
    """
    created_venv = False
    reused_existing = False
    pip_index_used = ""

    if not force_reinstall and not install_here and venv_dir is None:
        existing_py = resolve_whisper_python()
        if existing_py:
            target_python = Path(existing_py)
            reused_existing = True
            print(f"[跳过安装] 已检测到可用 faster-whisper 环境：{target_python}")
        else:
            target_python = get_whisper_venv_python()
    elif install_here:
        target_python = Path(sys.executable)
    else:
        resolved_venv = venv_dir.resolve() if venv_dir else (get_skill_cache_dir() / "whisper-venv")
        bin_sub = Path("Scripts") / "python.exe" if sys.platform.startswith("win") else Path("bin") / "python"
        target_python = resolved_venv / bin_sub

    if not reused_existing:
        if not install_here and not target_python.is_file():
            target_venv_root = target_python.parent.parent
            print(f"[1/2] 在外部缓存目录创建独立 Python 虚拟环境（不污染技能包）：{target_venv_root}")
            target_venv_root.mkdir(parents=True, exist_ok=True)
            venv.EnvBuilder(with_pip=True).create(str(target_venv_root))
            created_venv = True

        if force_reinstall or not python_has_module(target_python, "faster_whisper"):
            pkgs = ["faster-whisper"]
            if with_cuda_libs:
                pkgs.extend(["nvidia-cublas-cu12", "nvidia-cudnn-cu12"])
            print(f"[2/2] 安装依赖到 {target_python} ...")
            pip_index_used = _run_visible_pip(
                target_python,
                packages=pkgs,
                pip_index=pip_index,
                fallback_official=fallback_official,
            )
        else:
            reused_existing = True
            print(f"[跳过安装] 目标环境已具备 faster-whisper：{target_python}")

    if not python_has_module(target_python, "faster_whisper"):
        raise RuntimeError(f"安装完成后仍无法导入 faster_whisper，请检查环境：{target_python}")

    downloaded_model_path: Optional[str] = None
    if download_model_name:
        local_hit = resolve_local_whisper_model(download_model_name)
        if Path(local_hit).is_dir():
            downloaded_model_path = local_hit
            print(f"[模型已就绪] 发现本地模型目录：{downloaded_model_path}")
        else:
            cache_models_root = (model_dir or get_whisper_models_cache_dir()).resolve()
            cache_models_root.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            if not env.get("HF_ENDPOINT") and hf_endpoint:
                env["HF_ENDPOINT"] = hf_endpoint
            py_snippet = (
                "import sys; "
                "from faster_whisper import download_model; "
                "p = download_model(sys.argv[1], cache_dir=sys.argv[2]); "
                "print('__MODEL_PATH__=' + str(p))"
            )
            print(f"[模型预热] 从 {env.get('HF_ENDPOINT', 'HuggingFace')} 拉取 '{download_model_name}' 至 {cache_models_root} ...")
            res = subprocess.run(
                [str(target_python), "-c", py_snippet, download_model_name, str(cache_models_root)],
                env=env,
            )
            if res.returncode != 0:
                raise RuntimeError(f"预下载 faster-whisper 模型 '{download_model_name}' 失败（退出码 {res.returncode}）")
            downloaded_model_path = str(cache_models_root)

    return {
        "ok": True,
        "whisper_python": str(target_python),
        "created_venv": created_venv,
        "reused_existing": reused_existing,
        "pip_index_used": pip_index_used,
        "downloaded_model": download_model_name,
        "downloaded_model_path": downloaded_model_path,
        "models_cache_dir": str(get_whisper_models_cache_dir()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="为 talking-head-music-edit 自动安装外部隔离 faster-whisper 环境（Option A）")
    parser.add_argument("--check", action="store_true", help="仅只读检查 faster-whisper 是否已就绪（无安装副作用）")
    parser.add_argument("--install-here", action="store_true", help="直接安装到当前 Python 环境（默认创建外部隔离 cache venv）")
    parser.add_argument("--venv-dir", help="显式指定隔离虚拟环境根目录")
    parser.add_argument("--pip-index", default=DEFAULT_TUNA_PYPI, help="PyPI 镜像地址（默认清华源；传 'official' 用官方源）")
    parser.add_argument("--no-fallback-official", action="store_true", help="镜像失败时不自动回退官方源")
    parser.add_argument("--with-cuda-libs", action="store_true", help="同时安装 pip 版 nvidia-cublas-cu12 与 nvidia-cudnn-cu12（N 卡加速用）")
    parser.add_argument("--download-model", help="安装完成后顺便预下载指定 Whisper 模型权重（例如 small）")
    parser.add_argument("--model-dir", help="显式指定模型缓存目录")
    parser.add_argument("--hf-endpoint", default=DEFAULT_HF_MIRROR, help="HuggingFace 镜像地址（默认 hf-mirror.com）")
    parser.add_argument("--force-reinstall", action="store_true", help="强制重新安装 faster-whisper")
    args = parser.parse_args()

    if args.check:
        hit = resolve_whisper_python()
        status = {
            "ready": bool(hit),
            "whisper_python": hit,
            "default_venv_dir": str(get_whisper_venv_python().parent.parent),
            "models_cache_dir": str(get_whisper_models_cache_dir()),
        }
        print(json.dumps(status, ensure_ascii=False, indent=2))
        sys.exit(0 if hit else 2)

    pip_index = None if str(args.pip_index).strip().lower() == "official" else args.pip_index
    result = ensure_faster_whisper(
        venv_dir=Path(args.venv_dir) if args.venv_dir else None,
        install_here=args.install_here,
        pip_index=pip_index or "",
        fallback_official=not args.no_fallback_official,
        with_cuda_libs=args.with_cuda_libs,
        force_reinstall=args.force_reinstall,
        download_model_name=args.download_model,
        model_dir=Path(args.model_dir) if args.model_dir else None,
        hf_endpoint=args.hf_endpoint,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
