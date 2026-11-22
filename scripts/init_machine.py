"""电脑初始化与环境档案管理：检测 Python 3.10+、FFmpeg、剪映型号与路径、Qwen3_ASR / faster-whisper / 云端 API 与阿杜导演，并保存本机偏好以便后续自动复用。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

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
    detect_jianying_installation,
    detect_qwen3_asr,
    get_machine_profile_path,
    get_whisper_models_cache_dir,
    load_machine_profile,
    probe_environment,
    resolve_local_whisper_model,
    save_machine_profile,
)

WHISPER_MODEL_REPOS: Dict[str, str] = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}


def build_whisper_download_guide(model_name: str = "small", manual_dir_override: Optional[str] = None) -> Dict[str, Any]:
    """给出 faster-whisper 手动下载镜像地址、必需文件及精确本地保存路径（含程序自动下载命令）。"""
    repo_id = WHISPER_MODEL_REPOS.get(model_name, f"Systran/faster-whisper-{model_name}")
    default_save_dir = Path(manual_dir_override) if manual_dir_override else (get_whisper_models_cache_dir() / f"faster-whisper-{model_name}")
    local_hit = resolve_local_whisper_model(model_name)
    already_cached = Path(local_hit).is_dir()
    return {
        "model_name": model_name,
        "repo_id": repo_id,
        "already_downloaded_locally": already_cached,
        "resolved_local_path": local_hit if already_cached else None,
        "manual_download_url_mirror": f"https://hf-mirror.com/{repo_id}/tree/main",
        "manual_download_url_official": f"https://huggingface.co/{repo_id}/tree/main",
        "manual_save_directory": str(default_save_dir),
        "required_files": ["config.json", "model.bin", "tokenizer.json", "vocabulary.txt (or vocabulary.json)"],
        "auto_download_command": f'python "{SCRIPT_DIR / "bootstrap_faster_whisper.py"}" --download-model {model_name}',
    }


def inspect_machine(
    jianying_exe: Optional[str] = None,
    drafts_root: Optional[str] = None,
    root_meta: Optional[str] = None,
    qwen3_dir: Optional[str] = None,
    whisper_model: str = "small",
) -> Dict[str, Any]:
    """只读全面体检当前电脑环境，输出供 AI 或人类阅读的配置与缺口清单。"""
    env_data = probe_environment(
        drafts_root_override=drafts_root,
        root_meta_override=root_meta,
    )
    if jianying_exe:
        env_data["jianying_info"] = detect_jianying_installation(
            exe_override=jianying_exe,
            drafts_root_override=drafts_root,
            root_meta_override=root_meta,
        )
    if qwen3_dir:
        env_data["asr_backends"]["qwen3_asr"] = detect_qwen3_asr(dir_override=qwen3_dir)

    jy = env_data["jianying_info"]
    saved_profile = env_data.get("machine_profile") or {}
    saved_jy_root = (saved_profile.get("jianying") or {}).get("drafts_root")
    saved_music_dir = (saved_profile.get("music_library") or {}).get("root_dir")
    saved_style = (saved_profile.get("speech_edit") or {}).get("default_style")

    needs_prompts = {
        "python_missing_or_old": not env_data["python_310_plus"],
        "ffmpeg_missing": not bool(env_data["ffmpeg"]),
        "jianying_path_needed": not bool(saved_jy_root and Path(saved_jy_root).is_dir()),
        "asr_mode_needed": not bool((saved_profile.get("asr") or {}).get("default_backend")),
        "music_library_needed": not bool(saved_music_dir and Path(saved_music_dir).is_dir()),
        "preferences_needed": not bool(saved_style and saved_profile.get("editor_target")),
        "whisper_download_choice_needed": False,
    }
    whisper_guide = build_whisper_download_guide(model_name=whisper_model)
    if not env_data["asr_backends"]["faster_whisper_ready"] or not whisper_guide["already_downloaded_locally"]:
        needs_prompts["whisper_download_choice_needed"] = True

    missing_required_fields = [
        k for k in ("jianying_path_needed", "asr_mode_needed", "music_library_needed", "preferences_needed")
        if needs_prompts[k]
    ]

    return {
        "environment": env_data,
        "whisper_download_guide": whisper_guide,
        "needs_user_input": needs_prompts,
        "initialized_complete": len(missing_required_fields) == 0,
        "missing_required_fields": missing_required_fields,
    }


def save_profile_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """根据用户确认的参数保存或更新 machine_profile.json，供后续全流程免询问直接复用。"""
    profile = load_machine_profile()
    probe = inspect_machine(
        jianying_exe=args.jianying_exe,
        drafts_root=args.drafts_root,
        root_meta=args.root_meta,
        qwen3_dir=args.qwen3_dir,
        whisper_model=args.whisper_model or "small",
    )
    env = probe["environment"]
    jy_detected = env["jianying_info"]
    qwen_detected = env["asr_backends"]["qwen3_asr"]

    old_jy = dict(profile.get("jianying") or {})
    old_asr = dict(profile.get("asr") or {})
    old_adu = dict(profile.get("adudir") or {})

    jy_section = {
        "exe_path": args.jianying_exe or old_jy.get("exe_path") or jy_detected.get("exe_path"),
        "app_version": jy_detected.get("app_version") or old_jy.get("app_version"),
        "draft_format_version": jy_detected.get("draft_format_version") or old_jy.get("draft_format_version"),
        "drafts_root": args.drafts_root or old_jy.get("drafts_root") or jy_detected.get("drafts_root"),
        "root_meta_path": args.root_meta or old_jy.get("root_meta_path") or jy_detected.get("root_meta_path"),
    }

    asr_mode = args.asr_mode or old_asr.get("default_backend") or "auto"
    whisper_model = args.whisper_model or old_asr.get("faster_whisper_model") or "small"
    dl_mode = args.whisper_download_mode or old_asr.get("whisper_download_mode") or "unset"

    asr_section = {
        "default_backend": asr_mode,
        "qwen3_asr_dir": args.qwen3_dir or old_asr.get("qwen3_asr_dir") or qwen_detected.get("qwen3_asr_dir"),
        "qwen3_model_size": args.qwen3_model_size or old_asr.get("qwen3_model_size") or "1.7b",
        "faster_whisper_model": whisper_model,
        "whisper_download_mode": dl_mode,
        "whisper_manual_model_dir": (
            args.whisper_manual_model_dir
            or old_asr.get("whisper_manual_model_dir")
            or (
                str(get_whisper_models_cache_dir() / f"faster-whisper-{whisper_model}")
                if dl_mode == "manual"
                else None
            )
        ),
    }

    adu_section = {
        "default_mep_dir": args.adudir_mep_dir or old_adu.get("default_mep_dir"),
    }

    old_music = dict(profile.get("music_library") or {})
    music_section = {
        "root_dir": args.music_lib or old_music.get("root_dir"),
        "catalog_file": old_music.get("catalog_file") or "music_catalog.json",
    }

    old_speech = dict(profile.get("speech_edit") or {})
    speech_section = {
        "default_style": args.default_style or old_speech.get("default_style") or "balanced",
        "speech_volume": args.speech_volume if args.speech_volume is not None else old_speech.get("speech_volume", 1.0),
        "music_volume": args.music_volume if args.music_volume is not None else old_speech.get("music_volume", 0.18),
        "ffmpeg_burn_subtitles": (
            args.burn_subtitles
            if args.burn_subtitles is not None
            else old_speech.get("ffmpeg_burn_subtitles", False)
        ),
    }

    # 合并常用词汇表（只增不丢）
    old_glossary = dict(profile.get("glossary") or {})
    terms = list(old_glossary.get("terms") or [])
    for t in args.add_term or []:
        ts = str(t).strip()
        if ts and ts not in terms:
            terms.append(ts)
    replacements = dict(old_glossary.get("replacements") or {})
    for pair in args.add_replacement or []:
        if "=" in pair:
            wrong, right = pair.split("=", 1)
            wrong, right = wrong.strip(), right.strip()
            if wrong and right:
                replacements[wrong] = right
                if right not in terms:
                    terms.append(right)

    # 合并按视频类型的常用习惯
    presets = dict(profile.get("presets_by_type") or {})
    if args.save_preset:
        raw_preset = args.save_preset.strip()
        preset_path = Path(raw_preset)
        parsed_preset = (
            json.loads(preset_path.read_text(encoding="utf-8-sig"))
            if preset_path.is_file()
            else json.loads(raw_preset)
        )
        if isinstance(parsed_preset, dict):
            for k, v in parsed_preset.items():
                if isinstance(v, dict):
                    merged_item = dict(presets.get(k) or {})
                    merged_item.update(v)
                    presets[k] = merged_item

    updated = {
        "schema_version": 2,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python_executable": env["python_executable"],
        "python_version": env["python_version"],
        "ffmpeg_bin": args.ffmpeg_bin or profile.get("ffmpeg_bin") or env["ffmpeg"],
        "editor_target": args.editor_target or profile.get("editor_target") or "jianying",
        "jianying": jy_section,
        "asr": asr_section,
        "adudir": adu_section,
        "music_library": music_section,
        "speech_edit": speech_section,
        "glossary": {"terms": terms, "replacements": replacements},
        "speaker_profile": profile.get("speaker_profile") or {},
        "presets_by_type": presets,
    }

    download_result: Optional[Dict[str, Any]] = None
    if args.run_download and asr_mode == "faster-whisper":
        from bootstrap_faster_whisper import ensure_faster_whisper

        download_result = ensure_faster_whisper(download_model_name=whisper_model)
        updated["asr"]["whisper_download_mode"] = "auto_downloaded"

    profile_path = save_machine_profile(updated)
    return {
        "status": "saved",
        "profile_path": str(profile_path),
        "profile": updated,
        "whisper_guide": build_whisper_download_guide(
            whisper_model, manual_dir_override=updated["asr"].get("whisper_manual_model_dir")
        ),
        "download_result": download_result,
    }


def run_interactive_wizard() -> Dict[str, Any]:
    """终端交互式向导：一步步检查硬件/软件并询问配置偏好。"""
    report = inspect_machine()
    env = report["environment"]
    jy = env["jianying_info"]
    asr = env["asr_backends"]
    qwen = asr["qwen3_asr"]

    print("=== 电脑环境体检结果 ===")
    print(f"1. Python 版本：{env['python_version']} ({'符合 3.10+' if env['python_310_plus'] else '⚠️ 低于 3.10，建议升级'})")
    print(f"2. FFmpeg：{env['ffmpeg'] or '⚠️ 未检测到（需安装）'}")
    print(
        f"3. 剪映专业版：{'已检测到' if jy['installed'] else '未自动检测到'} | "
        f"版本/型号={jy['app_version'] or '未知'} (草稿格式={jy['draft_format_version'] or '未知'}) | "
        f"程序={jy['exe_path'] or '未找到'} | 草稿目录={jy['drafts_root']}"
    )
    print(f"4. 本机 Qwen3_ASR：{'已就绪 (' + str(qwen['available_models']) + ')' if qwen['available'] else '未检测到'}")
    print(f"5. 本地 faster-whisper：{'已安装' if asr['faster_whisper_ready'] else '未安装'}")

    jy_exe = input(f"\n请确认或输入剪映执行文件路径（直接回车保留 [{jy['exe_path'] or '跳过'}]）：").strip()
    drafts_root = input(f"请确认或输入剪映草稿根目录（直接回车保留 [{jy['drafts_root']}]）：").strip()

    print("\n请选择默认逐字时间戳识别模式：")
    print("  1) dashscope / tencent / groq / openai （云端 API）")
    print("  2) faster-whisper （本地离线轻量识别）")
    print("  3) qwen3-asr （本机既有 Qwen3_ASR 目录）")
    print("  4) auto （自动按已有配置与环境顺位挑选）")
    choice = input("请输入序号或后端名 [默认 4]：").strip().lower()
    mode_map = {"1": "dashscope", "2": "faster-whisper", "3": "qwen3-asr", "4": "auto", "": "auto"}
    asr_mode = mode_map.get(choice, choice)

    whisper_model = "small"
    dl_mode = "skip"
    manual_dir: Optional[str] = None
    run_dl = False
    if asr_mode == "faster-whisper":
        wm = input("请选择 faster-whisper 模型档位（small / base / medium / large-v3-turbo）[默认 small]：").strip()
        if wm:
            whisper_model = wm
        guide = build_whisper_download_guide(whisper_model)
        print(f"\n[下载方式确认] 模型 {whisper_model} 需准备权重大约数百 MB：")
        print(f"  方式 A：由程序在当前终端自动安装 faster-whisper 并通过国内镜像下载至：{guide['manual_save_directory']}")
        print(f"  方式 B：你自行手动下载 {guide['manual_download_url_mirror']} 的全部文件，保存进：{guide['manual_save_directory']}")
        dl_pick = input("请选择：1=程序自动下载安装 / 2=我稍后手动下载 [默认 1]：").strip()
        if dl_pick == "2":
            dl_mode = "manual"
            custom_d = input(f"请确认手动保存目录（直接回车使用 [{guide['manual_save_directory']}]）：").strip()
            manual_dir = custom_d or guide["manual_save_directory"]
        else:
            dl_mode = "auto"
            run_dl = True

    print("\n请选择默认剪辑工程输出目标：1) 剪映草稿 (jianying)  2) 阿杜导演 (adudir)  3) 两者 (both)")
    ed_pick = input("请输入序号 [默认 1]：").strip()
    ed_map = {"1": "jianying", "": "jianying", "2": "adudir", "3": "both"}
    editor_target = ed_map.get(ed_pick, "jianying")

    ns = argparse.Namespace(
        jianying_exe=jy_exe or None,
        drafts_root=drafts_root or None,
        root_meta=None,
        ffmpeg_bin=None,
        qwen3_dir=None,
        qwen3_model_size="1.7b",
        asr_mode=asr_mode,
        whisper_model=whisper_model,
        whisper_download_mode=dl_mode,
        whisper_manual_model_dir=manual_dir,
        editor_target=editor_target,
        adudir_mep_dir=None,
        music_lib=None,
        default_style=None,
        speech_volume=None,
        music_volume=None,
        burn_subtitles=None,
        add_term=None,
        add_replacement=None,
        save_preset=None,
        run_download=run_dl,
    )
    return save_profile_from_args(ns)


def main() -> None:
    parser = argparse.ArgumentParser(description="电脑环境初始化与配置保存工具（Python / FFmpeg / 剪映型号 / ASR / 阿杜导演）")
    parser.add_argument("--inspect", action="store_true", help="只读输出当前电脑配置检查 JSON 与下载路径指引")
    parser.add_argument("--save", action="store_true", help="将命令行指定的初始化配置保存至 machine_profile.json")
    parser.add_argument("--interactive", action="store_true", help="在终端运行交互式配置向导并保存结果")

    parser.add_argument("--jianying-exe", help="剪映可执行文件或安装目录路径")
    parser.add_argument("--drafts-root", help="剪映草稿根目录路径")
    parser.add_argument("--root-meta", help="剪映 root_meta_info.json 路径")
    parser.add_argument("--ffmpeg-bin", help="FFmpeg 所在目录或可执行文件路径")
    parser.add_argument("--asr-mode", choices=["auto", "faster-whisper", "qwen3-asr", "dashscope", "tencent", "groq", "openai"], help="默认逐字识别模式")
    parser.add_argument("--qwen3-dir", help="本机 Qwen3_ASR 目录路径")
    parser.add_argument("--qwen3-model-size", default="1.7b", choices=["1.7b", "0.6b"], help="Qwen3_ASR 默认模型大小")
    parser.add_argument("--whisper-model", default="small", help="faster-whisper 默认模型（small/base/medium/large-v3-turbo）")
    parser.add_argument("--whisper-download-mode", choices=["auto", "manual", "skip"], help="faster-whisper 下载方式偏好")
    parser.add_argument("--whisper-manual-model-dir", help="用户手动下载 faster-whisper 模型存放的本地目录")
    parser.add_argument("--run-download", action="store_true", help="配合 --save 执行 faster-whisper 自动安装与拉取模型")
    parser.add_argument("--editor-target", choices=["jianying", "adudir", "ffmpeg", "plan", "both"], help="默认交付方式")
    parser.add_argument("--adudir-mep-dir", help="默认阿杜导演 MEP 协作目录")
    parser.add_argument("--music-lib", help="音乐库根目录路径")
    parser.add_argument("--default-style", choices=["balanced", "compact", "natural"], help="默认口播剪辑档位")
    parser.add_argument("--speech-volume", type=float, help="默认口播人声音量倍数（如 1.0 或 1.2）")
    parser.add_argument("--music-volume", type=float, help="默认背景音乐音量倍数（如 0.18）")
    parser.add_argument("--burn-subtitles", action=argparse.BooleanOptionalAction, default=None, help="FFmpeg 出片是否默认烧录硬字幕")
    parser.add_argument("--add-term", action="append", help="向常用词汇表追加专有名词/人名（可多次传入）")
    parser.add_argument("--add-replacement", action="append", help="向常用词汇表追加纠错映射，格式为 错词=对词（可多次传入）")
    parser.add_argument("--save-preset", help="保存按视频类型的常用习惯（JSON 字符串或 JSON 文件路径）")
    args = parser.parse_args()

    if args.interactive:
        res = run_interactive_wizard()
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return

    if args.save:
        res = save_profile_from_args(args)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return

    res = inspect_machine(
        jianying_exe=args.jianying_exe,
        drafts_root=args.drafts_root,
        root_meta=args.root_meta,
        qwen3_dir=args.qwen3_dir,
        whisper_model=args.whisper_model,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
