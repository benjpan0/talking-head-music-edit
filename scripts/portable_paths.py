"""跨机环境路径自动发现模块：统一解析 ffmpeg/ffprobe、内置剪映引擎、剪映首页索引与草稿根目录。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

# 强制终端使用 UTF-8 编码，防止 Windows 控制台输出乱码
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _is_auto_value(val: Optional[str]) -> bool:
    if val is None:
        return True
    cleaned = str(val).strip()
    return not cleaned or cleaned.lower() == "auto"


def _candidate_ffmpeg_dirs() -> List[Path]:
    dirs: List[Path] = []
    env_bin = os.environ.get("FFMPEG_BIN", "").strip()
    if env_bin:
        p = Path(env_bin)
        dirs.append(p.parent if p.is_file() else p)

    profile = load_machine_profile()
    prof_ff = str(profile.get("ffmpeg_bin") or "").strip()
    if prof_ff:
        pp = Path(prof_ff)
        dirs.append(pp.parent if pp.is_file() else pp)

    if sys.platform.startswith("win"):
        local_app = Path(os.environ.get("LOCALAPPDATA", "") or (Path.home() / "AppData" / "Local"))
        dirs.extend(
            [
                Path(r"D:\Program\ffmpeg\bin"),
                Path(r"C:\ffmpeg\bin"),
                Path(r"C:\Program Files\ffmpeg\bin"),
                Path.home() / "scoop" / "shims",
                local_app / "Microsoft" / "WinGet" / "Links",
            ]
        )
    else:
        dirs.extend(
            [
                Path("/opt/homebrew/bin"),
                Path("/usr/local/bin"),
                Path("/usr/bin"),
            ]
        )
    return dirs


def find_tool(name: str) -> Optional[str]:
    """按 环境变量 FFMPEG_BIN -> PATH -> 常见安装目录 顺序探测可执行工具（如 ffmpeg/ffprobe）。"""
    ext = ".exe" if sys.platform.startswith("win") else ""

    env_bin = os.environ.get("FFMPEG_BIN", "").strip()
    if env_bin:
        ep = Path(env_bin)
        if ep.is_file() and ep.stem.lower() == name.lower():
            return str(ep)
        candidate = (ep.parent if ep.is_file() else ep) / f"{name}{ext}"
        if candidate.is_file():
            return str(candidate)

    which_hit = shutil.which(name)
    if which_hit:
        return which_hit

    for d in _candidate_ffmpeg_dirs():
        candidate = d / f"{name}{ext}"
        if candidate.is_file():
            return str(candidate)

    if name.lower() == "ffmpeg":
        try:
            import imageio_ffmpeg  # type: ignore

            exe = imageio_ffmpeg.get_ffmpeg_exe()
            if exe and Path(exe).is_file():
                return exe
        except Exception:
            pass
    return None


def require_tool(name: str) -> str:
    """获取工具绝对路径；找不到时给出跨平台清晰安装指引。"""
    hit = find_tool(name)
    if hit:
        return hit
    raise FileNotFoundError(
        f"未找到可执行文件 '{name}'。请先安装 FFmpeg 并加入系统 PATH，或设置环境变量 FFMPEG_BIN 指向包含 ffmpeg/ffprobe 的目录。\n"
        f"Windows 快速安装示例：winget install Gyan.FFmpeg\n"
        f"macOS 快速安装示例：brew install ffmpeg"
    )


def ensure_ffmpeg_in_path() -> Optional[str]:
    """若探测到 ffmpeg/ffprobe 所在目录，自动注入当前进程 PATH 供下游库（如 pyJianYingDraft）调用。"""
    for tool_name in ("ffprobe", "ffmpeg"):
        hit = find_tool(tool_name)
        if hit:
            bin_dir = str(Path(hit).resolve().parent)
            cur_path = os.environ.get("PATH", "")
            if bin_dir not in cur_path.split(os.pathsep):
                os.environ["PATH"] = bin_dir + os.pathsep + cur_path
            return hit
    return None


def get_bundled_jy_vendor_dir() -> Path:
    """返回技能内置剪映编辑与校验核心库目录（scripts/jy_vendor）。"""
    return Path(__file__).resolve().parent / "jy_vendor"


def resolve_jy_runtime() -> Path:
    """定位随技能分发的剪映运行库。"""
    directory = get_bundled_jy_vendor_dir()
    if not (directory / "jy_wrapper.py").is_file():
        raise FileNotFoundError(f"缺少本技能内的剪映运行库: {directory}")
    return directory


def resolve_draft_inspector() -> Path:
    inspector = resolve_jy_runtime() / "draft_inspector.py"
    if not inspector.is_file():
        raise FileNotFoundError(f"缺少本技能内的草稿检查程序: {inspector}")
    return inspector


def resolve_root_meta_path(cli_override: Optional[str] = None) -> Path:
    """跨平台自动定位剪映首页索引文件 root_meta_info.json。"""
    if not _is_auto_value(cli_override):
        return Path(str(cli_override).strip())

    env_meta = os.environ.get("JY_ROOT_META_PATH", "").strip()
    if env_meta:
        return Path(env_meta)

    profile = load_machine_profile()
    prof_meta = str((profile.get("jianying") or {}).get("root_meta_path") or "").strip()
    if prof_meta and not _is_auto_value(prof_meta):
        return Path(prof_meta)

    candidates: List[Path] = []
    if sys.platform.startswith("win"):
        local_app = Path(os.environ.get("LOCALAPPDATA", "") or (Path.home() / "AppData" / "Local"))
        candidates.append(
            local_app / "JianyingPro" / "User Data" / "Projects" / "com.lveditor.draft" / "root_meta_info.json"
        )
        candidates.append(
            Path.home() / "AppData" / "Local" / "JianyingPro" / "User Data" / "Projects" / "com.lveditor.draft" / "root_meta_info.json"
        )
    elif sys.platform == "darwin":
        candidates.extend(
            [
                Path.home() / "Movies" / "JianyingPro" / "User Data" / "Projects" / "com.lveditor.draft" / "root_meta_info.json",
                Path.home()
                / "Library"
                / "Containers"
                / "com.lemon.lvpro"
                / "Data"
                / "Movies"
                / "JianyingPro"
                / "User Data"
                / "Projects"
                / "com.lveditor.draft"
                / "root_meta_info.json",
            ]
        )

    for c in candidates:
        if c.exists():
            return c
    if candidates:
        return candidates[0]
    return Path.home() / "JianyingPro" / "root_meta_info.json"


def inspect_draft_roots_from_meta(root_meta_path: Path) -> Optional[Path]:
    """从剪映首页索引 root_meta_info.json 的既有条目中统计最常使用的真实草稿根目录。"""
    if not root_meta_path.is_file():
        return None
    try:
        data = json.loads(root_meta_path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None

    store = data.get("all_draft_store")
    if not isinstance(store, list):
        return None

    counter: Counter[str] = Counter()
    for entry in store:
        if not isinstance(entry, dict):
            continue
        root_str = str(entry.get("draft_root_path") or "").strip()
        if not root_str:
            fold_str = str(entry.get("draft_fold_path") or "").strip()
            if fold_str:
                root_str = str(Path(fold_str).parent)
        if not root_str:
            continue
        p = Path(root_str)
        if p.exists() and p.is_dir():
            counter[str(p)] += 1

    if not counter:
        return None
    most_common_path = counter.most_common(1)[0][0]
    return Path(most_common_path)


def resolve_drafts_root(
    cli_override: Optional[str] = None,
    root_meta_path: Optional[Path] = None,
    jy_scripts_dir: Optional[Path] = None,
) -> Path:
    """
    解析剪映真实草稿根目录：
    1) 非 auto 显式参数 -> 2) 环境变量 JY_PROJECTS_ROOT / JIANYING_DRAFTS_ROOT ->
    3) root_meta_info.json 中真实登记频率最高的现存草稿根目录 ->
    4) 内置引擎的 get_default_drafts_root() 或平台默认目录。
    """
    if not _is_auto_value(cli_override):
        return Path(str(cli_override).strip())

    for env_key in ("JY_PROJECTS_ROOT", "JIANYING_DRAFTS_ROOT"):
        val = os.environ.get(env_key, "").strip()
        if val:
            return Path(val)

    profile = load_machine_profile()
    prof_drafts = str((profile.get("jianying") or {}).get("drafts_root") or "").strip()
    if prof_drafts and not _is_auto_value(prof_drafts):
        return Path(prof_drafts)

    meta_path = root_meta_path or resolve_root_meta_path()
    discovered = inspect_draft_roots_from_meta(meta_path)
    if discovered is not None:
        return discovered

    if jy_scripts_dir is not None and str(jy_scripts_dir) not in sys.path:
        sys.path.insert(0, str(jy_scripts_dir))
    try:
        from utils.formatters import get_default_drafts_root  # type: ignore

        default_root = Path(get_default_drafts_root())
        return default_root
    except Exception:
        pass

    if meta_path.parent.exists():
        return meta_path.parent
    return Path.home() / "JianyingPro Drafts"


def get_skill_cache_dir() -> Path:
    """获取独立于技能目录/Obsidian 仓库之外的本机缓存目录（存放隔离 whisper-venv 与模型权重）。"""
    env_cache = os.environ.get("TALKING_HEAD_CACHE_DIR", "").strip()
    if env_cache:
        return Path(env_cache)

    if sys.platform.startswith("win"):
        base = Path(os.environ.get("LOCALAPPDATA", "") or (Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", "") or (Path.home() / ".cache"))
    return base / "talking-head-music-edit"


def get_whisper_venv_python() -> Path:
    """返回隔离版 whisper-venv 对应的 Python 解释器路径。"""
    venv_root = get_skill_cache_dir() / "whisper-venv"
    if sys.platform.startswith("win"):
        return venv_root / "Scripts" / "python.exe"
    return venv_root / "bin" / "python"


def get_whisper_models_cache_dir() -> Path:
    """返回外部缓存根目录下的模型存储文件夹。"""
    return get_skill_cache_dir() / "models"


def python_has_module(python_exe: Path, module_name: str) -> bool:
    """安全探测指定 Python 可执行文件是否已安装某个模块。"""
    if not python_exe.is_file():
        return False
    try:
        if python_exe.resolve() == Path(sys.executable).resolve():
            return importlib.util.find_spec(module_name) is not None
    except Exception:
        pass
    try:
        res = subprocess.run(
            [
                str(python_exe),
                "-c",
                f"import importlib.util, sys; sys.exit(0 if importlib.util.find_spec({module_name!r}) else 1)",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
        return res.returncode == 0
    except Exception:
        return False


def resolve_whisper_python(explicit: Optional[str] = None) -> Optional[str]:
    """
    自动查找可用的 faster-whisper Python 解释器：
    1) 显式传入 --whisper-python -> 2) 环境变量 TALKING_HEAD_WHISPER_PYTHON ->
    3) 当前运行中的 Python 解释器 -> 4) 外部缓存目录 whisper-venv 独立环境。
    """
    if not _is_auto_value(explicit):
        p = Path(str(explicit).strip())
        if python_has_module(p, "faster_whisper"):
            return str(p)

    env_py = os.environ.get("TALKING_HEAD_WHISPER_PYTHON", "").strip()
    if env_py:
        ep = Path(env_py)
        if python_has_module(ep, "faster_whisper"):
            return str(ep)

    if importlib.util.find_spec("faster_whisper") is not None:
        return sys.executable

    venv_py = get_whisper_venv_python()
    if python_has_module(venv_py, "faster_whisper"):
        return str(venv_py)

    return None


def get_machine_profile_path() -> Path:
    """返回本机初始化配置档案 machine_profile.json 路径（保存在技能包外用户缓存中）。"""
    env_p = os.environ.get("TALKING_HEAD_PROFILE_PATH", "").strip()
    if env_p:
        return Path(env_p)
    return get_skill_cache_dir() / "machine_profile.json"


def load_machine_profile(profile_path: Optional[Path] = None) -> Dict[str, Any]:
    """只读读取已保存的本机配置档案；若尚未初始化则返回空字典。"""
    p = profile_path or get_machine_profile_path()
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8-sig"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_machine_profile(data: Dict[str, Any], profile_path: Optional[Path] = None) -> Path:
    """原子保存本机初始化配置档案至用户缓存目录。"""
    p = profile_path or get_machine_profile_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)
    return p


def get_ffmpeg_version(ffmpeg_path: Optional[str]) -> Optional[str]:
    if not ffmpeg_path:
        return None
    try:
        res = subprocess.run(
            [ffmpeg_path, "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=5,
        )
        first_line = (res.stdout or "").splitlines()[0].strip() if res.stdout else ""
        return first_line or None
    except Exception:
        return None


def detect_jianying_installation(
    exe_override: Optional[str] = None,
    drafts_root_override: Optional[str] = None,
    root_meta_override: Optional[str] = None,
) -> Dict[str, Any]:
    """探测剪映专业版安装路径、版本号/型号、草稿格式版本、索引与草稿根目录。"""
    profile = load_machine_profile()
    prof_jy = profile.get("jianying") or {}

    exe_candidates: List[Path] = []
    if not _is_auto_value(exe_override):
        exe_candidates.append(Path(str(exe_override).strip()))
    prof_exe = str(prof_jy.get("exe_path") or "").strip()
    if prof_exe and not _is_auto_value(prof_exe):
        exe_candidates.append(Path(prof_exe))

    if sys.platform.startswith("win"):
        local_app = Path(os.environ.get("LOCALAPPDATA", "") or (Path.home() / "AppData" / "Local"))
        exe_candidates.extend(
            [
                local_app / "JianyingPro" / "Apps" / "JianyingPro.exe",
                Path(r"C:\Program Files\JianyingPro\Apps\JianyingPro.exe"),
                Path(r"D:\Program Files\jianying\Apps\JianyingPro.exe"),
                Path(r"D:\Program Files\JianyingPro\Apps\JianyingPro.exe"),
            ]
        )
    elif sys.platform == "darwin":
        exe_candidates.extend(
            [
                Path("/Applications/VideoFusion-macOS.app"),
                Path("/Applications/JianyingPro.app"),
            ]
        )

    found_exe: Optional[Path] = next((p for p in exe_candidates if p.exists()), None)
    app_version: Optional[str] = None

    if found_exe is not None:
        if sys.platform.startswith("win"):
            apps_dir = found_exe if found_exe.is_dir() else found_exe.parent
            cfg_ini = apps_dir / "Configure.ini"
            if cfg_ini.is_file():
                try:
                    ini_txt = cfg_ini.read_text(encoding="utf-8-sig", errors="ignore")
                    m = re.search(r"last_version\s*=\s*([0-9]+(?:\.[0-9]+)+)", ini_txt)
                    if m:
                        app_version = m.group(1)
                except Exception:
                    pass
            if not app_version and apps_dir.is_dir():
                ver_dirs = sorted(
                    (d.name for d in apps_dir.iterdir() if d.is_dir() and re.match(r"^\d+\.\d+\.\d+", d.name)),
                    reverse=True,
                )
                if ver_dirs:
                    app_version = ver_dirs[0]
        elif sys.platform == "darwin":
            plist = (found_exe / "Contents" / "Info.plist") if found_exe.is_dir() else None
            if plist and plist.is_file():
                try:
                    import plistlib

                    pl = plistlib.loads(plist.read_bytes())
                    app_version = str(pl.get("CFBundleShortVersionString") or pl.get("CFBundleVersion") or "") or None
                except Exception:
                    pass

    root_meta = resolve_root_meta_path(cli_override=root_meta_override)
    drafts_root = resolve_drafts_root(cli_override=drafts_root_override, root_meta_path=root_meta)

    draft_format_version: Optional[str] = None
    if root_meta.is_file():
        try:
            rm_data = json.loads(root_meta.read_text(encoding="utf-8-sig"))
            store = rm_data.get("all_draft_store") or []
            if store and isinstance(store[0], dict):
                v = str(store[0].get("draft_new_version") or "").strip()
                if v:
                    draft_format_version = v
        except Exception:
            pass

    return {
        "installed": found_exe is not None or root_meta.is_file() or drafts_root.is_dir(),
        "exe_path": str(found_exe) if found_exe else None,
        "app_version": app_version,
        "draft_format_version": draft_format_version,
        "root_meta_path": str(root_meta),
        "root_meta_exists": root_meta.is_file(),
        "drafts_root": str(drafts_root),
        "drafts_root_exists": drafts_root.is_dir(),
    }


def detect_qwen3_asr(dir_override: Optional[str] = None) -> Dict[str, Any]:
    """探测本机已部署的 Qwen3_ASR 目录及其模型完备情况。"""
    profile = load_machine_profile()
    prof_dir = str((profile.get("asr") or {}).get("qwen3_asr_dir") or "").strip()

    candidates: List[Path] = []
    if not _is_auto_value(dir_override):
        candidates.append(Path(str(dir_override).strip()))
    env_dir = os.environ.get("QWEN3_ASR_DIR", "").strip()
    if env_dir:
        candidates.append(Path(env_dir))
    if prof_dir and not _is_auto_value(prof_dir):
        candidates.append(Path(prof_dir))
    candidates.extend([Path(r"D:\Qwen3_ASR"), Path(r"C:\Qwen3_ASR"), Path(r"E:\Qwen3_ASR")])

    found_dir: Optional[Path] = None
    for c in candidates:
        if (c / "batch_transcribe.py").is_file():
            found_dir = c
            break

    if found_dir is None:
        return {
            "available": False,
            "qwen3_asr_dir": None,
            "python_executable": None,
            "batch_script": None,
            "available_models": [],
            "aligner_ready": False,
        }

    py_candidates = [
        found_dir / "WPy64-312101" / "python" / "python.exe",
        found_dir / "python" / "python.exe",
    ]
    qwen_py = next((p for p in py_candidates if p.is_file()), None)
    if qwen_py is None:
        wpy_dirs = sorted(found_dir.glob("WPy*"))
        for wd in wpy_dirs:
            cand_py = wd / "python" / "python.exe"
            if cand_py.is_file():
                qwen_py = cand_py
                break

    batch_script = found_dir / "batch_transcribe.py"
    model_paths: Dict[str, str] = {}
    align_path = ""
    try:
        src_txt = batch_script.read_text(encoding="utf-8-sig", errors="ignore")
        for key in ("1.7b", "0.6b"):
            m = re.search(rf'"{key}"\s*:\s*r?["\']([^"\']+)["\']', src_txt)
            if m:
                model_paths[key] = m.group(1)
        m_align = re.search(r'ALIGN_PATH\s*=\s*r?["\']([^"\']+)["\']', src_txt)
        if m_align:
            align_path = m_align.group(1)
    except Exception:
        pass

    available_models = [k for k, p in model_paths.items() if Path(p).is_dir()]
    aligner_ready = bool(align_path and Path(align_path).is_dir())

    return {
        "available": bool(qwen_py and qwen_py.is_file() and available_models and aligner_ready),
        "qwen3_asr_dir": str(found_dir),
        "python_executable": str(qwen_py) if qwen_py else None,
        "batch_script": str(batch_script),
        "model_paths": model_paths,
        "available_models": available_models,
        "align_path": align_path or None,
        "aligner_ready": aligner_ready,
    }


def resolve_local_whisper_model(model_arg: str) -> str:
    """若传入模型路径、profile 手动目录或本地预置目录命中 CTranslate2 权重（含 model.bin），返回本地目录路径；否则原样返回模型别名。"""
    cand_direct = Path(model_arg)
    if cand_direct.is_dir() and (cand_direct / "model.bin").is_file():
        return str(cand_direct.resolve())

    profile = load_machine_profile()
    manual_dir = str((profile.get("asr") or {}).get("whisper_manual_model_dir") or "").strip()
    if manual_dir:
        mp = Path(manual_dir)
        if mp.is_dir() and (mp / "model.bin").is_file():
            return str(mp.resolve())

    skill_root = Path(__file__).resolve().parents[1]
    cache_models = get_whisper_models_cache_dir()
    names = [model_arg, f"faster-whisper-{model_arg}"]
    search_bases = [skill_root / "models", cache_models]
    for base in search_bases:
        for name in names:
            d = base / name
            if d.is_dir() and (d / "model.bin").is_file():
                return str(d.resolve())
    return model_arg


def probe_environment(
    drafts_root_override: Optional[str] = None,
    root_meta_override: Optional[str] = None,
) -> Dict[str, Any]:
    """汇总当前机器的工具、Python、剪映版本与全部转写引擎状态（纯只读）。"""
    ffmpeg_path = find_tool("ffmpeg")
    ffprobe_path = find_tool("ffprobe")
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    jy_scripts: Optional[str] = None
    inspector_path: Optional[str] = None
    jy_error: Optional[str] = None
    jy_engine_mode: Optional[str] = None
    try:
        jy_dir = resolve_jy_runtime()
        jy_scripts = str(jy_dir)
        inspector_path = str(resolve_draft_inspector())
        jy_engine_mode = "bundled_jy_vendor"
    except Exception as exc:
        jy_error = str(exc)

    jianying_info = detect_jianying_installation(
        drafts_root_override=drafts_root_override,
        root_meta_override=root_meta_override,
    )
    qwen3_info = detect_qwen3_asr()
    whisper_py = resolve_whisper_python()
    profile_path = get_machine_profile_path()
    profile_data = load_machine_profile(profile_path)

    asr_status = {
        "dashscope_key_set": bool(os.environ.get("DASHSCOPE_API_KEY")),
        "tencent_keys_set": bool(
            os.environ.get("TENCENTCLOUD_SECRET_ID")
            and os.environ.get("TENCENTCLOUD_SECRET_KEY")
            and os.environ.get("TENCENTCLOUD_APPID")
        ),
        "groq_key_set": bool(os.environ.get("GROQ_API_KEY")),
        "openai_key_set": bool(os.environ.get("OPENAI_API_KEY")),
        "faster_whisper_installed_here": importlib.util.find_spec("faster_whisper") is not None,
        "faster_whisper_ready": bool(whisper_py),
        "faster_whisper_python": whisper_py,
        "faster_whisper_venv_default": str(get_whisper_venv_python()),
        "faster_whisper_models_cache_dir": str(get_whisper_models_cache_dir()),
        "qwen3_asr": qwen3_info,
        "requests_installed": importlib.util.find_spec("requests") is not None,
        "websockets_installed": importlib.util.find_spec("websockets") is not None,
        "aiohttp_installed": importlib.util.find_spec("aiohttp") is not None,
    }

    return {
        "platform": sys.platform,
        "python_version": py_ver,
        "python_310_plus": sys.version_info >= (3, 10),
        "python_executable": sys.executable,
        "ffmpeg": ffmpeg_path,
        "ffmpeg_version": get_ffmpeg_version(ffmpeg_path),
        "ffprobe": ffprobe_path,
        "jy_engine_mode": jy_engine_mode,
        "jianying_editor_scripts": jy_scripts,
        "draft_inspector": inspector_path,
        "jianying_editor_error": jy_error,
        "jianying_info": jianying_info,
        "root_meta_path": jianying_info["root_meta_path"],
        "root_meta_exists": jianying_info["root_meta_exists"],
        "drafts_root": jianying_info["drafts_root"],
        "drafts_root_exists": jianying_info["drafts_root_exists"],
        "asr_backends": asr_status,
        "machine_profile_path": str(profile_path),
        "machine_profile_exists": profile_path.is_file(),
        "machine_profile": profile_data,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="只读探测跨机工具链、剪映路径与可用转写后端")
    parser.add_argument("--drafts-root", help="显式指定剪映草稿根目录")
    parser.add_argument("--root-meta", help="显式指定 root_meta_info.json 路径")
    args = parser.parse_args()

    payload = probe_environment(
        drafts_root_override=args.drafts_root,
        root_meta_override=args.root_meta,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

