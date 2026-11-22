"""剪映草稿底层工具：进程检测、运行时加载、可选精剪字幕轨写入、双写同步与首页索引登记。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from portable_paths import (  # noqa: E402
    ensure_ffmpeg_in_path,
    resolve_drafts_root,
    resolve_jy_runtime,
    resolve_root_meta_path,
)
from scan_talking_head import parse_srt  # noqa: E402

ensure_ffmpeg_in_path()
US = 1_000_000


def load_jy_runtime():
    """加载内置 pyJianYingDraft 与 JyProject，返回 (jy_runtime_dir, JyProject, draft)。"""
    jy_runtime_dir = resolve_jy_runtime()
    if str(jy_runtime_dir) not in sys.path:
        sys.path.insert(0, str(jy_runtime_dir))
    from jy_wrapper import JyProject  # noqa: E402
    import pyJianYingDraft as draft  # noqa: E402

    return jy_runtime_dir, JyProject, draft


def check_jianying_running() -> bool:
    """检查当前系统是否有剪映主进程正在运行，避免首页索引并发写冲突。"""
    if not sys.platform.startswith("win"):
        return False
    res = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq JianyingPro.exe"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    return "JianyingPro.exe" in res.stdout


def add_srt_subtitles_to_project(
    jy_project: Any,
    draft_module: Any,
    srt_path: Path,
    width: int,
    height: int,
) -> int:
    """读取已精剪的 .srt 字幕文件，按横竖屏默认样式写入剪映 Subtitles 轨，返回写入条数。"""
    if not srt_path.is_file():
        raise FileNotFoundError(f"字幕文件不存在: {srt_path}")
    cues = parse_srt(srt_path)
    if not cues:
        return 0

    is_portrait = height > width
    sub_style = draft_module.TextStyle(
        size=7.0 if is_portrait else 6.5,
        color=(1.0, 1.0, 1.0),
        align=1,
        auto_wrapping=False,
    )
    sub_border = draft_module.TextBorder(
        color=(0.0, 0.0, 0.0),
        alpha=1.0,
        width=40.0,
    )
    sub_clip = draft_module.ClipSettings(transform_y=-0.60 if is_portrait else -0.78)

    jy_project._ensure_track(draft_module.TrackType.text, "Subtitles")
    written = 0
    for c in cues:
        text = str(c.get("text") or "").strip()
        if not text:
            continue
        start_us = int(round(float(c["start"]) * US))
        end_us = int(round(float(c["end"]) * US))
        dur_us = end_us - start_us
        if dur_us <= 0:
            continue
        seg = jy_project.add_text_simple(
            text,
            start_time=start_us,
            duration=dur_us,
            track_name="Subtitles",
            style=sub_style,
            border=sub_border,
            clip_settings=sub_clip,
        )
        if seg is None:
            raise RuntimeError(f"添加字幕片段失败: #{c.get('index')} {text}")
        written += 1
    return written


def sync_draft_files_and_register(
    draft_dir: Path,
    draft_name: str,
    drafts_root: Path,
    root_meta_path: Path,
    backup_dir: Path,
    total_duration_us: int,
    skip_register: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """双写 draft_info.json & draft_content.json，更新 draft_meta_info.json，并安全登记到首页索引。"""
    draft_info_path = draft_dir / "draft_info.json"
    draft_content_path = draft_dir / "draft_content.json"
    draft_meta_path = draft_dir / "draft_meta_info.json"

    draft_data = json.loads(draft_info_path.read_text(encoding="utf-8"))
    draft_id = draft_data.get("id") or str(uuid.uuid4()).upper()
    draft_data["id"] = draft_id

    pretty_json = json.dumps(draft_data, ensure_ascii=False, indent=2)
    draft_info_path.write_text(pretty_json, encoding="utf-8")
    draft_content_path.write_text(pretty_json, encoding="utf-8")

    now_us = int(time.time() * US)
    now_sec = int(time.time())
    meta = json.loads(draft_meta_path.read_text(encoding="utf-8")) if draft_meta_path.exists() else {}

    draft_fold_win = str(draft_dir).replace("/", "\\")
    draft_root_win = str(drafts_root).replace("/", "\\")
    draft_cover_win = str(draft_dir / "draft_cover.jpg").replace("/", "\\")

    meta.update(
        {
            "draft_id": draft_id,
            "draft_name": draft_name,
            "draft_fold_path": draft_fold_win,
            "draft_root_path": draft_root_win,
            "draft_cover": draft_cover_win,
            "tm_duration": total_duration_us,
            "tm_draft_create": meta.get("tm_draft_create") or now_us,
            "tm_draft_modified": now_us,
            "tm_draft_cloud_modified": now_sec * 1000,
            "draft_need_rename_folder": False,
        }
    )
    draft_meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    reg_info: Dict[str, Any] = {"registered": False, "skipped": skip_register}
    if skip_register:
        return draft_id, reg_info

    if not root_meta_path.exists():
        reg_info["warning"] = f"root_meta_info.json 不存在，已跳过首页登记: {root_meta_path}"
        return draft_id, reg_info

    backup_name = f"root_meta_info_backup_{time.strftime('%Y%m%d_%H%M%S')}.json"
    backup_path = backup_dir / backup_name
    shutil.copy2(root_meta_path, backup_path)

    before_bytes = root_meta_path.read_bytes()
    before_sha = hashlib.sha256(before_bytes).hexdigest()
    root_data = json.loads(before_bytes.decode("utf-8"))

    store = root_data.get("all_draft_store", [])
    old_count = len(store)

    filtered_store = [
        item
        for item in store
        if item.get("draft_name") != draft_name
        and os.path.normcase(str(item.get("draft_fold_path", ""))) != os.path.normcase(draft_fold_win)
    ]

    new_entry = {
        "cloud_draft_cover": False,
        "cloud_draft_sync": False,
        "draft_cloud_last_action_download": False,
        "draft_cloud_purchase_info": "",
        "draft_cloud_template_id": "",
        "draft_cloud_tutorial_info": "",
        "draft_cloud_videocut_purchase_info": "",
        "draft_cover": draft_cover_win,
        "draft_fold_path": draft_fold_win,
        "draft_id": draft_id,
        "draft_is_ai_shorts": False,
        "draft_is_cloud_temp_draft": False,
        "draft_is_invisible": False,
        "draft_json_file": str(draft_content_path).replace("/", "\\"),
        "draft_name": draft_name,
        "draft_new_version": meta.get("draft_new_version", ""),
        "draft_root_path": draft_root_win,
        "draft_timeline_materials_size": meta.get("draft_timeline_materials_size_", 0),
        "draft_type": "",
        "tm_draft_cloud_completed": "",
        "tm_draft_cloud_modified": now_sec * 1000,
        "tm_draft_create": now_us,
        "tm_draft_modified": now_us,
        "tm_draft_removed": 0,
        "tm_duration": total_duration_us,
    }
    filtered_store.insert(0, new_entry)
    root_data["all_draft_store"] = filtered_store

    new_bytes = json.dumps(root_data, ensure_ascii=False).encode("utf-8")
    root_meta_path.write_bytes(new_bytes)
    after_sha = hashlib.sha256(new_bytes).hexdigest()

    reg_info.update(
        {
            "registered": True,
            "backup_path": str(backup_path),
            "old_count": old_count,
            "new_count": len(filtered_store),
            "before_sha256": before_sha,
            "after_sha256": after_sha,
        }
    )
    return draft_id, reg_info


__all__ = [
    "add_srt_subtitles_to_project",
    "check_jianying_running",
    "load_jy_runtime",
    "resolve_drafts_root",
    "resolve_root_meta_path",
    "sync_draft_files_and_register",
]
