"""读取 timeline_edit.json（主时间线执行过渡文件），校验并输出方案指引、剪映草稿、ADu 工程或 FFmpeg 成片。"""

from __future__ import annotations

import argparse
import getpass
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

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

US = 1_000_000
CLIP_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
ASSET_TYPES = ("video", "audio", "image")
DURATION_TOLERANCE_SEC = 0.05


def sec_to_us(sec: float) -> int:
    return int(round(float(sec) * US))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _num(
    value: Any,
    field: str,
    minimum: Optional[float] = None,
    exclusive_min: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    require(type(value) in (int, float) and math.isfinite(float(value)), f"{field} 必须是有效数字")
    val = float(value)
    if minimum is not None:
        require(val >= minimum, f"{field} 不能小于 {minimum}（当前为 {val}）")
    if exclusive_min is not None:
        require(val > exclusive_min, f"{field} 必须大于 {exclusive_min}（当前为 {val}）")
    if maximum is not None:
        require(val <= maximum, f"{field} 不能大于 {maximum}（当前为 {val}）")
    return val


def _int(value: Any, field: str, minimum: int = 0) -> int:
    require(type(value) is int and value >= minimum, f"{field} 必须是 >= {minimum} 的整数")
    return value


def _text(value: Any, field: str) -> str:
    require(isinstance(value, str) and bool(value.strip()), f"{field} 不能为空")
    return value.strip()


def load_timeline_edit(plan_path: Path, check_files: bool = True) -> Dict[str, Any]:
    """读取并校验 timeline_edit.json，返回标准化后的结构。"""
    plan_path = Path(plan_path).resolve()
    require(plan_path.is_file(), f"时间线执行文件不存在: {plan_path}")
    raw = read_json(plan_path)
    require(isinstance(raw, dict), "时间线执行文件根节点必须是 JSON 对象")

    proj_raw = raw.get("project")
    require(isinstance(proj_raw, dict), "缺少 project 对象")
    project = {
        "name": _text(proj_raw.get("name"), "project.name"),
        "width": _int(proj_raw.get("width"), "project.width", minimum=1),
        "height": _int(proj_raw.get("height"), "project.height", minimum=1),
        "fps": _num(proj_raw.get("fps"), "project.fps", minimum=1),
    }

    assets_raw = raw.get("assets")
    require(isinstance(assets_raw, dict) and len(assets_raw) >= 1, "assets 至少需要包含 1 个素材")
    assets: Dict[str, Dict[str, Any]] = {}
    for asset_id, info in assets_raw.items():
        require(isinstance(asset_id, str) and bool(asset_id.strip()), "assets 中存在空素材编号")
        require(isinstance(info, dict), f"assets.{asset_id} 必须是对象")
        a_type = info.get("type")
        require(a_type in ASSET_TYPES, f"assets.{asset_id}.type 必须是 {ASSET_TYPES} 之一")
        raw_path = _text(info.get("path"), f"assets.{asset_id}.path")
        p = Path(raw_path)
        resolved = p.resolve() if p.is_absolute() else (plan_path.parent / p).resolve()
        if check_files:
            require(resolved.is_file(), f"素材文件不存在 ({asset_id}): {resolved}")
        assets[asset_id] = {
            "id": asset_id,
            "type": a_type,
            "raw_path": raw_path,
            "path": resolved,
        }

    tracks_raw = raw.get("tracks")
    require(isinstance(tracks_raw, dict), "缺少 tracks 对象")
    video_raw = tracks_raw.get("video")
    audio_raw = tracks_raw.get("audio")
    require(isinstance(video_raw, list), "tracks.video 必须是数组")
    require(isinstance(audio_raw, list), "tracks.audio 必须是数组")
    require(bool(video_raw or audio_raw), "tracks.video 和 tracks.audio 不能同时为空")

    seen_ids = set()

    def parse_clips(rows: List[Any], lane: str) -> List[Dict[str, Any]]:
        allowed_types = ("video", "image") if lane == "video" else ("audio",)
        clips: List[Dict[str, Any]] = []
        by_track: Dict[int, List[Dict[str, Any]]] = {}

        for idx, row in enumerate(rows):
            prefix = f"tracks.{lane}[{idx}]"
            require(isinstance(row, dict), f"{prefix} 必须是对象")
            clip_id = _text(row.get("id"), f"{prefix}.id")
            require(bool(CLIP_ID_RE.fullmatch(clip_id)), f"{prefix}.id 格式不合法: {clip_id}")
            require(clip_id not in seen_ids, f"片段 id 重复: {clip_id}")
            seen_ids.add(clip_id)

            track_idx = _int(row.get("track"), f"{clip_id}.track", minimum=0)
            asset_id = _text(row.get("asset"), f"{clip_id}.asset")
            require(asset_id in assets, f"{clip_id} 引用了未定义的素材: {asset_id}")
            asset_type = assets[asset_id]["type"]
            require(
                asset_type in allowed_types,
                f"{clip_id} 位于 {lane} 轨，但素材 {asset_id} 类型为 {asset_type}",
            )

            t_start = _num(row.get("timeline_start"), f"{clip_id}.timeline_start", minimum=0)
            t_end = _num(row.get("timeline_end"), f"{clip_id}.timeline_end", exclusive_min=0)
            require(t_end > t_start, f"{clip_id} 的 timeline_end({t_end}) 必须大于 timeline_start({t_start})")

            s_start = _num(row.get("source_start"), f"{clip_id}.source_start", minimum=0)
            s_end = _num(row.get("source_end"), f"{clip_id}.source_end", exclusive_min=0)
            require(s_end > s_start, f"{clip_id} 的 source_end({s_end}) 必须大于 source_start({s_start})")

            speed = _num(row.get("speed", 1.0), f"{clip_id}.speed", exclusive_min=0)
            volume = _num(row.get("volume", 1.0), f"{clip_id}.volume", minimum=0, maximum=4)
            fade_in = _num(row.get("fade_in", 0.0), f"{clip_id}.fade_in", minimum=0)
            fade_out = _num(row.get("fade_out", 0.0), f"{clip_id}.fade_out", minimum=0)

            t_dur = t_end - t_start
            s_dur = s_end - s_start
            expected_t_dur = s_dur / speed
            if asset_type != "image":
                require(
                    abs(expected_t_dur - t_dur) <= DURATION_TOLERANCE_SEC,
                    f"{clip_id} 时长与倍速不一致：源时长 {s_dur:.3f}s / 倍速 {speed} = {expected_t_dur:.3f}s，"
                    f"但时间线时长为 {t_dur:.3f}s",
                )
            require(
                fade_in <= t_dur + 1e-6 and fade_out <= t_dur + 1e-6,
                f"{clip_id} 的淡入({fade_in}s)或淡出({fade_out}s)超过片段时长({t_dur:.3f}s)",
            )

            clip = {
                "id": clip_id,
                "lane": lane,
                "track": track_idx,
                "asset": asset_id,
                "asset_type": asset_type,
                "timeline_start": t_start,
                "timeline_end": t_end,
                "timeline_duration": t_dur,
                "source_start": s_start,
                "source_end": s_end,
                "source_duration": s_dur,
                "timeline_start_us": sec_to_us(t_start),
                "timeline_end_us": sec_to_us(t_end),
                "source_start_us": sec_to_us(s_start),
                "source_end_us": sec_to_us(s_end),
                "speed": speed,
                "volume": volume,
                "fade_in": fade_in,
                "fade_out": fade_out,
                "fade_in_us": sec_to_us(fade_in),
                "fade_out_us": sec_to_us(fade_out),
            }
            clips.append(clip)
            by_track.setdefault(track_idx, []).append(clip)

        for track_idx, group in sorted(by_track.items()):
            group.sort(key=lambda c: (c["timeline_start_us"], c["timeline_end_us"]))
            for prev, curr in zip(group, group[1:]):
                require(
                    curr["timeline_start_us"] >= prev["timeline_end_us"],
                    f"{lane} 轨 track={track_idx} 存在时间重叠：{prev['id']} 结束于 {prev['timeline_end']}s，"
                    f"但 {curr['id']} 开始于 {curr['timeline_start']}s。"
                    f"如需交叉淡入淡出，请把两段放到不同 track（如 0 和 1）。",
                )

        return clips

    video_clips = parse_clips(video_raw, "video")
    audio_clips = parse_clips(audio_raw, "audio")
    all_clips = video_clips + audio_clips
    total_duration_us = max(c["timeline_end_us"] for c in all_clips)

    return {
        "path": plan_path,
        "project": project,
        "assets": assets,
        "tracks": {
            "video": video_clips,
            "audio": audio_clips,
        },
        "total_duration_us": total_duration_us,
        "total_duration_sec": round(total_duration_us / US, 4),
        "raw": raw,
    }


def _group_by_track(clips: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for c in clips:
        grouped.setdefault(c["track"], []).append(c)
    for k in grouped:
        grouped[k].sort(key=lambda c: (c["timeline_start_us"], c["timeline_end_us"]))
    return dict(sorted(grouped.items()))


def generate_handoff_guide(
    plan: Dict[str, Any],
    output_dir: Path,
    target: str,
    snapshot_path: Path,
    extra_notes: Optional[List[str]] = None,
) -> Path:
    """生成人话版操作指引与 JSON 解读（含可直接复制给智能体的说明）。"""
    proj = plan["project"]
    video_clips = sorted(plan["tracks"]["video"], key=lambda c: (c["timeline_start"], c["track"]))
    audio_clips = sorted(plan["tracks"]["audio"], key=lambda c: (c["timeline_start"], c["track"]))

    # 自动识别音乐交叉淡入淡出与停顿留白
    audio_events: List[str] = []
    for i in range(len(audio_clips) - 1):
        a = audio_clips[i]
        b = audio_clips[i + 1]
        if a["timeline_end"] > b["timeline_start"] + 1e-4:
            overlap = round(min(a["timeline_end"], b["timeline_end"]) - b["timeline_start"], 3)
            audio_events.append(
                f"- **交叉淡入淡出**：`{a['id']}`（音轨 {a['track']}，淡出 {a['fade_out']}s）与 "
                f"`{b['id']}`（音轨 {b['track']}，淡入 {b['fade_in']}s）在成片 "
                f"`{b['timeline_start']}s ~ {min(a['timeline_end'], b['timeline_end'])}s` 重叠 `{overlap}s`。"
            )
        elif b["timeline_start"] > a["timeline_end"] + 0.05:
            gap = round(b["timeline_start"] - a["timeline_end"], 3)
            audio_events.append(
                f"- **停顿留白**：`{a['id']}` 于 `{a['timeline_end']}s` 结束，留白 `{gap}s` 后由 "
                f"`{b['id']}` 于 `{b['timeline_start']}s` 续播。"
            )

    lines: List[str] = [
        f"# 操作指引与剪辑说明 — {proj['name']}",
        "",
        "> **复制给智能体执行**：如需交给剪辑智能体（如 ADu / 剪映助手）执行，请直接将下方文件路径与说明复制给智能体。",
        "",
        "## 一、文件清单说明",
        "",
        f"- **主时间线执行文件（JSON）**：`{snapshot_path}`（记录全部视频轨、音频轨起止与淡入淡出参数）",
        f"- **本次执行模式**：`{target}`",
        f"- **画布与成片总长**：`{proj['width']}x{proj['height']}` @ `{proj['fps']}fps`，总长 `{plan['total_duration_sec']}s`",
    ]
    for asset_id, info in plan["assets"].items():
        lines.append(f"- **素材 `{asset_id}`（{info['type']}）**：`{info['path']}`")
    if extra_notes:
        for note in extra_notes:
            lines.append(f"- {note}")

    lines.extend(["", "## 二、视频轨剪辑清单（JSON 人话对照）", ""])
    if video_clips:
        lines.extend([
            "| 片段 | 轨道 | 素材 | 成片起止(s) | 源素材起止(s) | 倍速 | 音量 | 淡入/淡出(s) |",
            "| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for c in video_clips:
            lines.append(
                f"| `{c['id']}` | V{c['track']} | `{c['asset']}` | "
                f"`{c['timeline_start']} ~ {c['timeline_end']}` | "
                f"`{c['source_start']} ~ {c['source_end']}` | "
                f"`{c['speed']}x` | `{c['volume']}` | `{c['fade_in']} / {c['fade_out']}` |"
            )
    else:
        lines.append("（无视频片段）")

    lines.extend(["", "## 三、音频轨剪辑清单（含双音轨交叉淡入淡出）", ""])
    if audio_clips:
        lines.extend([
            "| 片段 | 音轨 | 素材 | 成片起止(s) | 源素材起止(s) | 音量 | 淡入(s) | 淡出(s) |",
            "| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for c in audio_clips:
            lines.append(
                f"| `{c['id']}` | A{c['track']} | `{c['asset']}` | "
                f"`{c['timeline_start']} ~ {c['timeline_end']}` | "
                f"`{c['source_start']} ~ {c['source_end']}` | "
                f"`{c['volume']}` | `{c['fade_in']}` | `{c['fade_out']}` |"
            )
    else:
        lines.append("（无音频片段）")

    if audio_events:
        lines.extend(["", "### 音乐衔接与留白重点", ""])
        lines.extend(audio_events)

    guide_path = output_dir / "操作指引与剪辑说明.md"
    guide_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return guide_path


def build_jianying(
    plan: Dict[str, Any],
    output_dir: Path,
    drafts_root_override: Optional[str] = None,
    root_meta_override: Optional[str] = None,
    overwrite: bool = False,
    skip_register: bool = False,
    jianying_guard: bool = True,
    srt_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """根据 timeline_edit.json 生成剪映草稿（支持多视频轨、双音轨交叉淡入淡出与可选精剪字幕轨）。"""
    from jianying_io import (
        add_srt_subtitles_to_project,
        check_jianying_running,
        load_jy_runtime,
        resolve_drafts_root,
        resolve_root_meta_path,
        sync_draft_files_and_register,
    )

    jy_runtime_dir, JyProject, draft = load_jy_runtime()

    proj_cfg = plan["project"]
    draft_name = proj_cfg["name"]
    width = int(proj_cfg["width"])
    height = int(proj_cfg["height"])
    fps = int(round(float(proj_cfg["fps"])))

    root_meta_path = resolve_root_meta_path(cli_override=root_meta_override)
    drafts_root = resolve_drafts_root(
        cli_override=drafts_root_override,
        root_meta_path=root_meta_path,
        jy_scripts_dir=jy_runtime_dir,
    )
    target_draft_dir = drafts_root / draft_name
    if target_draft_dir.exists() and not overwrite:
        raise FileExistsError(
            f"剪映草稿已存在且未允许覆盖: {target_draft_dir}。"
            f"请修改 project.name 或传入 --overwrite。"
        )
    if jianying_guard and check_jianying_running():
        raise RuntimeError(
            "检测到 JianyingPro.exe 正在运行！请先退出剪映专业版后再执行，或加 --no-jianying-guard。"
        )

    drafts_root.mkdir(parents=True, exist_ok=True)
    jy = JyProject(
        draft_name,
        width=width,
        height=height,
        drafts_root=str(drafts_root),
        overwrite=overwrite,
    )
    jy._explicit_res = True
    jy.script.width = width
    jy.script.height = height
    jy.script.fps = fps

    used_asset_ids = {c["asset"] for c in (plan["tracks"]["video"] + plan["tracks"]["audio"])}
    materials: Dict[str, Any] = {}
    for asset_id in sorted(used_asset_ids):
        info = plan["assets"][asset_id]
        if info["type"] in ("video", "image"):
            mat = draft.VideoMaterial(str(info["path"]))
        else:
            mat = draft.AudioMaterial(str(info["path"]))
        jy.script.add_material(mat)
        materials[asset_id] = mat

    video_by_track = _group_by_track(plan["tracks"]["video"])
    for track_idx, clips in video_by_track.items():
        track_name = "VideoTrack" if track_idx == 0 else f"VideoTrack_{track_idx}"
        jy._ensure_track(draft.TrackType.video, track_name)
        for c in clips:
            mat = materials[c["asset"]]
            src_start_us = c["source_start_us"]
            src_end_us = min(c["source_end_us"], int(mat.duration))
            src_dur_us = src_end_us - src_start_us
            require(src_dur_us > 0, f"{c['id']} 源区间超出素材实际时长 ({mat.duration / US:.3f}s)")
            target_dur_us = c["timeline_end_us"] - c["timeline_start_us"]

            seg_kwargs: Dict[str, Any] = {
                "target_timerange": draft.Timerange(c["timeline_start_us"], target_dur_us),
                "volume": c["volume"],
            }
            if c["asset_type"] != "image":
                seg_kwargs["source_timerange"] = draft.Timerange(src_start_us, src_dur_us)
                if abs(c["speed"] - 1.0) > 1e-6:
                    seg_kwargs["speed"] = c["speed"]

            seg = draft.VideoSegment(mat, **seg_kwargs)
            if c["asset_type"] == "video" and (c["fade_in_us"] > 0 or c["fade_out_us"] > 0):
                seg.add_fade(c["fade_in_us"], c["fade_out_us"])
            jy.script.add_segment(seg, track_name)

    audio_by_track = _group_by_track(plan["tracks"]["audio"])
    for track_idx, clips in audio_by_track.items():
        track_name = f"AudioTrack_{track_idx}"
        jy._ensure_track(draft.TrackType.audio, track_name)
        for c in clips:
            mat = materials[c["asset"]]
            src_start_us = c["source_start_us"]
            src_end_us = min(c["source_end_us"], int(mat.duration))
            src_dur_us = src_end_us - src_start_us
            require(src_dur_us > 0, f"{c['id']} 源区间超出音频实际时长 ({mat.duration / US:.3f}s)")
            target_dur_us = c["timeline_end_us"] - c["timeline_start_us"]

            seg_kwargs = {
                "target_timerange": draft.Timerange(c["timeline_start_us"], target_dur_us),
                "source_timerange": draft.Timerange(src_start_us, src_dur_us),
                "volume": c["volume"],
            }
            if abs(c["speed"] - 1.0) > 1e-6:
                seg_kwargs["speed"] = c["speed"]

            seg = draft.AudioSegment(mat, **seg_kwargs)
            if c["fade_in_us"] > 0 or c["fade_out_us"] > 0:
                seg.add_fade(c["fade_in_us"], c["fade_out_us"])
            jy.script.add_segment(seg, track_name)

    subtitle_count = 0
    if srt_path is not None:
        subtitle_count = add_srt_subtitles_to_project(jy, draft, srt_path, width, height)

    jy._explicit_res = True
    jy.script.width = width
    jy.script.height = height
    jy.script.fps = fps

    track_errors = []
    if hasattr(jy, "_validate_tracks"):
        track_errors = jy._validate_tracks() or []
    jy.save()

    draft_id, reg_info = sync_draft_files_and_register(
        draft_dir=target_draft_dir,
        draft_name=draft_name,
        drafts_root=drafts_root,
        root_meta_path=root_meta_path,
        backup_dir=output_dir,
        total_duration_us=plan["total_duration_us"],
        skip_register=skip_register,
    )

    if track_errors:
        raise RuntimeError(f"剪映草稿轨道校验发现异常: {track_errors}")

    return {
        "draft_name": draft_name,
        "draft_id": draft_id,
        "draft_dir": str(target_draft_dir),
        "canvas": [width, height],
        "fps": fps,
        "duration_sec": plan["total_duration_sec"],
        "video_tracks": len(video_by_track),
        "video_clips": len(plan["tracks"]["video"]),
        "audio_tracks": len(audio_by_track),
        "audio_clips": len(plan["tracks"]["audio"]),
        "subtitle_count": subtitle_count,
        "root_meta_registration": reg_info,
    }


def build_adudir(
    plan: Dict[str, Any],
    output_dir: Path,
    mep_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """根据 timeline_edit.json 生成 ADu Director MEP 1 离线请求包，或在提供 --mep-dir 时在线提交。"""
    from adudir_mep import (
        ALLOWED_MEP_FPS,
        MEP_MAX_OPS_PER_REQUEST,
        check_mep_workspace,
        normalize_mep_fps,
        submit_mep_request,
    )

    proj_cfg = plan["project"]
    width = int(proj_cfg["width"])
    height = int(proj_cfg["height"])
    require(64 <= width <= 4096 and 64 <= height <= 4096, f"ADu 画布尺寸超出 64~4096 范围: {width}x{height}")
    raw_fps = float(proj_cfg["fps"])
    mep_fps = raw_fps if raw_fps in ALLOWED_MEP_FPS else normalize_mep_fps(raw_fps)

    safe_slug = re.sub(r"[^A-Za-z0-9_-]+", "_", proj_cfg["name"]).strip("_") or "timeline"
    workspace_folder = f"{safe_slug[:32]}-{uuid4().hex[:8]}"
    inbox_root = f"inbox/{workspace_folder}"
    asset_root = f"assets/imported/{workspace_folder}"

    used_asset_ids = sorted({c["asset"] for c in (plan["tracks"]["video"] + plan["tracks"]["audio"])})
    media_inbox_to_local: Dict[str, Path] = {}
    asset_to_src: Dict[str, str] = {}
    for asset_id in used_asset_ids:
        info = plan["assets"][asset_id]
        src_path: Path = info["path"]
        file_name = f"{asset_id}{src_path.suffix}"
        rel_inbox = f"{inbox_root}/{file_name}"
        media_inbox_to_local[rel_inbox] = src_path
        asset_to_src[asset_id] = f"{asset_root}/{file_name}"

    ops: List[Dict[str, Any]] = [
        {"op": "stage", "w": width, "h": height, "fps": mep_fps},
    ]

    def append_track_group(clips: List[Dict[str, Any]]) -> None:
        for idx, c in enumerate(clips):
            clip_payload: Dict[str, Any] = {
                "id": c["id"],
                "src": asset_to_src[c["asset"]],
                "at": round(c["timeline_start"], 4),
                "time": [round(c["source_start"], 4), round(c["source_end"], 4)],
                "volume": round(c["volume"], 4),
            }
            if abs(c["speed"] - 1.0) > 1e-6:
                clip_payload["speed"] = round(c["speed"], 4)
            ops.append({
                "op": "add",
                "track": "top" if idx == 0 else 0,
                "clip": clip_payload,
            })
            if c["fade_in"] > 0 or c["fade_out"] > 0:
                ops.append({
                    "op": "set",
                    "clip": c["id"],
                    "field": "fade",
                    "value": [round(c["fade_in"], 4), round(c["fade_out"], 4)],
                })

    audio_by_track = _group_by_track(plan["tracks"]["audio"])
    for track_idx in sorted(audio_by_track.keys(), reverse=True):
        append_track_group(audio_by_track[track_idx])

    video_by_track = _group_by_track(plan["tracks"]["video"])
    for track_idx in sorted(video_by_track.keys()):
        append_track_group(video_by_track[track_idx])

    import_ops = [{"op": "import", "path": rel} for rel in media_inbox_to_local]
    import_batches = [
        import_ops[i : i + MEP_MAX_OPS_PER_REQUEST]
        for i in range(0, len(import_ops), MEP_MAX_OPS_PER_REQUEST)
    ]
    timeline_batches = [
        ops[i : i + MEP_MAX_OPS_PER_REQUEST]
        for i in range(0, len(ops), MEP_MAX_OPS_PER_REQUEST)
    ]

    bundle_path = output_dir / "timeline_mep_bundle.json"
    write_json(bundle_path, {
        "protocol": "mep/1",
        "mode": "offline_request",
        "inbox_files": {rel: str(p) for rel, p in media_inbox_to_local.items()},
        "import_ops_batches": import_batches,
        "timeline_ops_batches": timeline_batches,
    })

    if mep_dir is None:
        return {
            "mode": "offline_request",
            "bundle": str(bundle_path),
            "video_tracks": len(video_by_track),
            "audio_tracks": len(audio_by_track),
            "ops_count": len(ops),
        }

    _, state = check_mep_workspace(mep_dir)
    latest_rev = str(state["revision"])
    key = getpass.getpass("阿杜导演本单口令: ")
    require(bool(key), "在线提交需要输入本单口令")

    inbox_dir = mep_dir / inbox_root
    inbox_dir.mkdir(parents=True, exist_ok=False)
    for rel, local_path in media_inbox_to_local.items():
        dst = mep_dir / rel
        try:
            dst.hardlink_to(local_path)
        except OSError:
            shutil.copy2(local_path, dst)

    receipts_path = output_dir / "mep_receipts.json"
    receipts: List[Dict[str, Any]] = []
    imported_map: Dict[str, str] = {}

    for batch in import_batches:
        res = submit_mep_request(mep_dir, key, {"base": latest_rev, "ops": batch})
        latest_rev = str(res["revision"])
        imported_map.update((res.get("result") or {}).get("imported") or {})
        receipts.append(res)
        write_json(receipts_path, receipts)

    src_remap = {
        rel.replace("inbox/", "assets/imported/", 1): imported_map.get(rel)
        for rel in media_inbox_to_local
    }
    for op in ops:
        if op["op"] == "add":
            old_src = op["clip"]["src"]
            require(isinstance(src_remap.get(old_src), str), f"导入回执未确认素材: {old_src}")
            op["clip"]["src"] = src_remap[old_src]

    for batch in timeline_batches:
        res = submit_mep_request(mep_dir, key, {"base": latest_rev, "ops": batch})
        latest_rev = str(res["revision"])
        receipts.append(res)
        write_json(receipts_path, receipts)

    return {
        "mode": "live_submitted",
        "bundle": str(bundle_path),
        "receipts": str(receipts_path),
        "final_revision": latest_rev,
        "ops_count": len(ops),
    }


def _atempo_chain(speed: float) -> List[str]:
    """将任意正倍速拆成 FFmpeg atempo 合法链（每级 0.5~2.0）。"""
    filters: List[str] = []
    rem = float(speed)
    while rem > 2.0 + 1e-6:
        filters.append("atempo=2.0")
        rem /= 2.0
    while rem < 0.5 - 1e-6:
        filters.append("atempo=0.5")
        rem /= 0.5
    if abs(rem - 1.0) > 1e-5:
        filters.append(f"atempo={rem:.6f}")
    return filters


def _escape_ffmpeg_sub_path(path: Path) -> str:
    s = str(path.resolve()).replace("\\", "/")
    s = s.replace(":", "\\:")
    s = s.replace("'", "\\'")
    return s


def build_ffmpeg(
    plan: Dict[str, Any],
    output_dir: Path,
    burn_srt: Optional[Path] = None,
) -> Dict[str, Any]:
    """直接使用 FFmpeg 按 timeline_edit.json 渲染成片（支持多视频轨、变速、双音轨交叉淡入淡出与可选烧录字幕）。"""
    from portable_paths import ensure_ffmpeg_in_path

    ensure_ffmpeg_in_path()
    ffmpeg = shutil.which("ffmpeg")
    require(bool(ffmpeg), "未找到 ffmpeg，请先安装 FFmpeg 或配置 PATH")

    proj = plan["project"]
    width = int(proj["width"])
    height = int(proj["height"])
    fps = float(proj["fps"])
    total_dur = float(plan["total_duration_sec"])
    safe_name = re.sub(r'[\\/:*?"<>|]+', "_", proj["name"]).strip() or "output"
    out_mp4 = output_dir / f"{safe_name}.mp4"

    used_asset_ids = sorted({c["asset"] for c in (plan["tracks"]["video"] + plan["tracks"]["audio"])})
    asset_input_idx: Dict[str, int] = {}
    cmd: List[str] = [str(ffmpeg), "-y", "-v", "error"]
    for idx, asset_id in enumerate(used_asset_ids):
        info = plan["assets"][asset_id]
        if info["type"] == "image":
            cmd.extend(["-loop", "1", "-i", str(info["path"])])
        else:
            cmd.extend(["-i", str(info["path"])])
        asset_input_idx[asset_id] = idx

    filters: List[str] = []
    # 底画布
    filters.append(f"color=c=black:s={width}x{height}:r={fps}:d={total_dur:.4f}[base]")
    current_v = "base"

    # 视频轨从低到高叠加
    video_by_track = _group_by_track(plan["tracks"]["video"])
    v_counter = 0
    a_labels: List[str] = []

    for track_idx in sorted(video_by_track.keys()):
        for c in video_by_track[track_idx]:
            in_i = asset_input_idx[c["asset"]]
            v_counter += 1
            v_tag = f"vc{v_counter}"
            dur = c["timeline_duration"]
            if c["asset_type"] == "image":
                v_chain = (
                    f"[{in_i}:v]trim=duration={dur:.4f},setpts=PTS-STARTPTS,"
                    f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                    f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,fps={fps},"
                    f"setpts=PTS+{c['timeline_start']:.4f}/TB[{v_tag}]"
                )
            else:
                pts_expr = f"(PTS-STARTPTS)/{c['speed']:.6f}" if abs(c["speed"] - 1.0) > 1e-6 else "PTS-STARTPTS"
                v_chain = (
                    f"[{in_i}:v]trim=start={c['source_start']:.4f}:end={c['source_end']:.4f},"
                    f"setpts={pts_expr},"
                    f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                    f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,fps={fps},"
                    f"setpts=PTS+{c['timeline_start']:.4f}/TB[{v_tag}]"
                )
            filters.append(v_chain)
            next_v = f"ov{v_counter}"
            filters.append(
                f"[{current_v}][{v_tag}]overlay=eof_action=pass:shortest=0:"
                f"enable='between(t,{c['timeline_start']:.4f},{c['timeline_end']:.4f})'[{next_v}]"
            )
            current_v = next_v

            # 视频自带音轨（音量 > 0 时纳入混音）
            if c["asset_type"] == "video" and c["volume"] > 0:
                a_tag = f"va{v_counter}"
                a_parts = [
                    f"atrim=start={c['source_start']:.4f}:end={c['source_end']:.4f}",
                    "asetpts=PTS-STARTPTS",
                ]
                a_parts.extend(_atempo_chain(c["speed"]))
                if abs(c["volume"] - 1.0) > 1e-5:
                    a_parts.append(f"volume={c['volume']:.4f}")
                if c["fade_in"] > 0:
                    a_parts.append(f"afade=t=in:st=0:d={c['fade_in']:.4f}")
                if c["fade_out"] > 0:
                    st_out = max(0.0, dur - c["fade_out"])
                    a_parts.append(f"afade=t=out:st={st_out:.4f}:d={c['fade_out']:.4f}")
                delay_ms = int(round(c["timeline_start"] * 1000))
                if delay_ms > 0:
                    a_parts.append(f"adelay={delay_ms}|{delay_ms}")
                filters.append(f"[{in_i}:a]{','.join(a_parts)}[{a_tag}]")
                a_labels.append(f"[{a_tag}]")

    # 独立音频轨（含双音轨交叉淡入淡出）
    a_counter = 0
    for c in sorted(plan["tracks"]["audio"], key=lambda x: (x["track"], x["timeline_start"])):
        if c["volume"] <= 0:
            continue
        in_i = asset_input_idx[c["asset"]]
        a_counter += 1
        a_tag = f"ac{a_counter}"
        dur = c["timeline_duration"]
        a_parts = [
            f"atrim=start={c['source_start']:.4f}:end={c['source_end']:.4f}",
            "asetpts=PTS-STARTPTS",
        ]
        a_parts.extend(_atempo_chain(c["speed"]))
        if abs(c["volume"] - 1.0) > 1e-5:
            a_parts.append(f"volume={c['volume']:.4f}")
        if c["fade_in"] > 0:
            a_parts.append(f"afade=t=in:st=0:d={c['fade_in']:.4f}")
        if c["fade_out"] > 0:
            st_out = max(0.0, dur - c["fade_out"])
            a_parts.append(f"afade=t=out:st={st_out:.4f}:d={c['fade_out']:.4f}")
        delay_ms = int(round(c["timeline_start"] * 1000))
        if delay_ms > 0:
            a_parts.append(f"adelay={delay_ms}|{delay_ms}")
        filters.append(f"[{in_i}:a]{','.join(a_parts)}[{a_tag}]")
        a_labels.append(f"[{a_tag}]")

    if burn_srt is not None:
        require(burn_srt.is_file(), f"字幕文件不存在: {burn_srt}")
        escaped_srt = _escape_ffmpeg_sub_path(burn_srt)
        filters.append(f"[{current_v}]subtitles='{escaped_srt}'[vout]")
        current_v = "vout"

    if a_labels:
        if len(a_labels) == 1:
            filters.append(f"{a_labels[0]}apad=whole_dur={total_dur:.4f},atrim=duration={total_dur:.4f}[aout]")
        else:
            filters.append(
                f"{''.join(a_labels)}amix=inputs={len(a_labels)}:duration=longest:normalize=0,"
                f"apad=whole_dur={total_dur:.4f},atrim=duration={total_dur:.4f}[aout]"
            )
        has_audio = True
    else:
        has_audio = False

    cmd.extend(["-filter_complex", ";".join(filters), "-map", f"[{current_v}]"])
    if has_audio:
        cmd.extend(["-map", "[aout]", "-c:a", "aac", "-b:a", "192k"])
    cmd.extend([
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(out_mp4),
    ])

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="ignore")
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg 出片失败: {proc.stderr.strip()}")

    return {
        "output_mp4": str(out_mp4),
        "canvas": [width, height],
        "fps": fps,
        "duration_sec": total_dur,
        "burn_srt": str(burn_srt.resolve()) if burn_srt else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="读取 timeline_edit.json，校验并输出方案指引、剪映草稿、ADu 工程或 FFmpeg 成片"
    )
    parser.add_argument("--plan", type=Path, required=True, help="timeline_edit.json 文件路径")
    parser.add_argument("--check", action="store_true", help="只做格式与时间线校验，不生成工程")
    parser.add_argument(
        "--skip-files",
        action="store_true",
        help="配合 --check 或 --target plan 使用：不检查本地素材文件是否存在",
    )
    parser.add_argument(
        "--target",
        choices=("plan", "jianying", "adudir", "ffmpeg", "both"),
        help="交付方式：plan（仅方案与人话指引）、jianying（剪映草稿）、adudir（ADu 工程）、ffmpeg（直接出片）或 both（剪映+ADu）",
    )
    parser.add_argument("--output", type=Path, help="输出目录（建议新目录）")
    parser.add_argument("--drafts-root", help="覆盖剪映草稿根目录（测试时可指定隔离目录）")
    parser.add_argument("--root-meta", help="覆盖剪映首页索引 root_meta_info.json 路径")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖同名剪映草稿")
    parser.add_argument("--skip-register", action="store_true", help="跳过剪映首页索引登记（测试用）")
    parser.add_argument("--no-jianying-guard", action="store_true", help="跳过剪映进程运行检查")
    parser.add_argument("--mep-dir", type=Path, help="ADu 协作目录路径（不填则只导出离线 MEP 包）")
    parser.add_argument("--srt", type=Path, help="可选精剪字幕 .srt 路径（写入剪映原生字幕轨，并记入操作指引）")
    parser.add_argument("--burn-srt", type=Path, help="配合 --target ffmpeg 使用：将指定 .srt 烧录进画面")
    args = parser.parse_args()

    if not args.check:
        if args.target is None or args.output is None:
            parser.error("执行时必须同时指定 --target 和 --output")
        if args.skip_files and args.target != "plan":
            parser.error("--skip-files 仅用于 --check 或 --target plan 模式")

    plan = load_timeline_edit(args.plan, check_files=not args.skip_files)

    v_tracks = sorted({c["track"] for c in plan["tracks"]["video"]})
    a_tracks = sorted({c["track"] for c in plan["tracks"]["audio"]})

    # 防过度切碎检查：同一视频轨、同一素材相邻两段若在原片中几乎连续（差值 < 0.15s）则给出明确警告
    frag_warnings: List[str] = []
    for t_idx, group in _group_by_track(plan["tracks"]["video"]).items():
        for prev, curr in zip(group, group[1:]):
            if (
                prev["asset"] == curr["asset"]
                and prev["asset_type"] == "video"
                and abs(prev["speed"] - curr["speed"]) < 1e-6
                and abs(prev["volume"] - curr["volume"]) < 1e-6
            ):
                src_gap = curr["source_start"] - prev["source_end"]
                if 0.0 <= src_gap < 0.15:
                    frag_warnings.append(
                        f"警告：视频轨 V{t_idx} 的 {prev['id']} 与 {curr['id']} 在原片中几乎连续"
                        f"（仅差 {src_gap:.3f}s），顺畅表达请合并为一个完整片段，避免逐句切碎。"
                    )

    if args.check:
        print(
            f"校验通过：{plan['project']['name']} "
            f"({plan['project']['width']}x{plan['project']['height']} @ {plan['project']['fps']}fps，"
            f"总长 {plan['total_duration_sec']}s)\n"
            f"  视频：{len(plan['tracks']['video'])} 段，轨道 {v_tracks}\n"
            f"  音频：{len(plan['tracks']['audio'])} 段，轨道 {a_tracks}"
        )
        for w in frag_warnings:
            print(f"  [!] {w}")
        return

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / "timeline_edit.json"
    write_json(snapshot_path, plan["raw"])

    report: Dict[str, Any] = {
        "source_plan": str(plan["path"]),
        "project": plan["project"],
        "total_duration_sec": plan["total_duration_sec"],
        "outputs": {},
    }
    extra_notes: List[str] = []

    if args.srt is not None:
        extra_notes.append(f"**精剪字幕（SRT）**：`{args.srt.resolve()}`")

    if args.target in ("jianying", "both"):
        jy_dir = output_dir / "jianying"
        jy_dir.mkdir(parents=True, exist_ok=True)
        jy_res = build_jianying(
            plan=plan,
            output_dir=jy_dir,
            drafts_root_override=args.drafts_root,
            root_meta_override=args.root_meta,
            overwrite=args.overwrite,
            skip_register=args.skip_register,
            jianying_guard=not args.no_jianying_guard,
            srt_path=args.srt,
        )
        report["outputs"]["jianying"] = jy_res
        extra_notes.append(f"**剪映草稿目录**：`{jy_res['draft_dir']}`")

    if args.target in ("adudir", "both"):
        adu_dir = output_dir / "adudir"
        adu_dir.mkdir(parents=True, exist_ok=True)
        adu_res = build_adudir(plan=plan, output_dir=adu_dir, mep_dir=args.mep_dir)
        report["outputs"]["adudir"] = adu_res
        extra_notes.append(f"**ADu MEP 包**：`{adu_res['bundle']}`")

    if args.target == "ffmpeg":
        ff_dir = output_dir / "ffmpeg"
        ff_dir.mkdir(parents=True, exist_ok=True)
        ff_res = build_ffmpeg(plan=plan, output_dir=ff_dir, burn_srt=args.burn_srt)
        report["outputs"]["ffmpeg"] = ff_res
        extra_notes.append(f"**FFmpeg 成片**：`{ff_res['output_mp4']}`")

    guide_path = generate_handoff_guide(
        plan=plan,
        output_dir=output_dir,
        target=args.target,
        snapshot_path=snapshot_path,
        extra_notes=extra_notes,
    )
    report["outputs"]["handoff_guide"] = str(guide_path)

    report_path = output_dir / "build_timeline_report.json"
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
