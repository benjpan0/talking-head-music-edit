"""音乐库文件夹规范校验、元数据自动补全与清理整理工具（audio/ + analysis/ + music_catalog.json）。"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# 强制终端使用 UTF-8 编码，防止 Windows 控制台输出乱码
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg"}
ANALYSIS_SUFFIX = "_剪辑数据.json"


def _fmt_mmss(sec: float) -> str:
    total = max(0, int(round(sec)))
    return f"{total // 60}:{total % 60:02d}"


def _read_analysis_meta(analysis_path: Path) -> Dict[str, Any]:
    """从 *_剪辑数据.json 提取时长、BPM、情绪、场景、试听段与简要备注。"""
    empty = {
        "duration_sec": None,
        "bpm": None,
        "mood": [],
        "scenes": [],
        "preview_segments": [],
        "notes": "",
    }
    try:
        data = json.loads(analysis_path.read_text(encoding="utf-8-sig"))
    except Exception:
        return empty

    dur_ms = data.get("duration_ms")
    dur_sec = round(float(dur_ms) / 1000.0, 2) if isinstance(dur_ms, (int, float)) else None

    metrics = data.get("metrics") or {}
    beat_grid = data.get("beat_grid") or {}
    raw_bpm = metrics.get("bpm") or beat_grid.get("bpm") or data.get("bpm")
    bpm_val = round(float(raw_bpm), 1) if isinstance(raw_bpm, (int, float)) else None

    music = data.get("music") or {}
    mood_list: List[str] = []
    raw_mood = str(music.get("mood") or "").strip()
    if raw_mood:
        for part in re.split(r"[，,、；;]+", raw_mood):
            ps = part.strip()
            if ps and ps not in mood_list:
                mood_list.append(ps)
    genres = [str(g.get("name") or "").strip() for g in (music.get("genres") or []) if isinstance(g, dict)]
    genres = [g for g in genres if g]

    scenes: List[str] = []
    for sc in music.get("film_scenarios") or []:
        if isinstance(sc, dict):
            name = str(sc.get("name") or "").strip()
            if name and name not in scenes:
                scenes.append(name)

    preview_segments: List[Dict[str, str]] = []
    sections = data.get("sections") or []
    if sections and isinstance(sections[0], dict):
        s0_start = float(sections[0].get("start_ms", 0)) / 1000.0
        s0_end = float(sections[0].get("end_ms", min(20000, dur_ms or 20000))) / 1000.0
        preview_segments.append({"label": "开头引入", "range": f"{_fmt_mmss(s0_start)}-{_fmt_mmss(min(s0_end, s0_start + 20.0))}"})
    elif dur_sec:
        preview_segments.append({"label": "开头引入", "range": f"0:00-{_fmt_mmss(min(20.0, dur_sec))}"})

    climax = data.get("climax") or {}
    if isinstance(climax, dict) and climax.get("start_ms") is not None and climax.get("end_ms") is not None:
        c_s = float(climax["start_ms"]) / 1000.0
        c_e = float(climax["end_ms"]) / 1000.0
        preview_segments.append({"label": "高潮段", "range": f"{_fmt_mmss(c_s)}-{_fmt_mmss(c_e)}"})

    if len(sections) >= 2:
        non_silent = [s for s in sections if isinstance(s, dict) and float(s.get("energy", 1)) > 0.02]
        if non_silent:
            last_s = non_silent[-1]
            ls_s = float(last_s.get("start_ms", 0)) / 1000.0
            ls_e = float(last_s.get("end_ms", dur_ms or 0)) / 1000.0
            if ls_e > ls_s:
                preview_segments.append({"label": "结尾段", "range": f"{_fmt_mmss(max(ls_s, ls_e - 18.0))}-{_fmt_mmss(ls_e)}"})

    instruments = [str(i.get("name") or "").strip() for i in (music.get("instruments") or []) if isinstance(i, dict)]
    instruments = [i for i in instruments if i][:4]
    note_parts = []
    if metrics.get("key") or metrics.get("meter"):
        note_parts.append(" ".join(x for x in [str(metrics.get("key") or ""), str(metrics.get("meter") or "")] if x))
    if genres:
        note_parts.append("风格：" + "/".join(genres))
    if instruments:
        note_parts.append("乐器：" + "、".join(instruments))

    return {
        "duration_sec": dur_sec,
        "bpm": bpm_val,
        "mood": mood_list,
        "scenes": scenes,
        "preview_segments": preview_segments,
        "notes": "；".join(note_parts),
    }


def _match_analysis_stem(filename: str) -> Optional[str]:
    if filename.endswith(ANALYSIS_SUFFIX):
        return filename[: -len(ANALYSIS_SUFFIX)]
    return None


def _find_extra_items(lib_dir: Path) -> List[Path]:
    """查找音乐库中不属于标准结构（audio/*.音频、analysis/*_剪辑数据.json、根目录 music_catalog.json）的无关文件或目录。"""
    extras: List[Path] = []
    for child in sorted(lib_dir.iterdir()):
        if child.is_dir():
            if child.name not in ("audio", "analysis"):
                extras.append(child)
            elif child.name == "audio":
                for sub in sorted(child.iterdir()):
                    if sub.is_dir() or sub.suffix.lower() not in AUDIO_EXTS:
                        extras.append(sub)
            elif child.name == "analysis":
                for sub in sorted(child.iterdir()):
                    if sub.is_dir() or not sub.name.endswith(ANALYSIS_SUFFIX):
                        extras.append(sub)
        else:
            if child.name != "music_catalog.json":
                extras.append(child)
    return extras


def inspect_music_library(lib_dir: Path) -> Dict[str, Any]:
    """检查音乐库是否符合标准目录结构：audio/ + analysis/ + music_catalog.json 且无多余无关文件。"""
    lib_dir = Path(lib_dir).resolve()
    if not lib_dir.is_dir():
        return {
            "exists": False,
            "compliant": False,
            "library_dir": str(lib_dir),
            "issues": [f"音乐库目录不存在: {lib_dir}"],
        }

    audio_dir = lib_dir / "audio"
    analysis_dir = lib_dir / "analysis"
    catalog_path = lib_dir / "music_catalog.json"

    issues: List[str] = []
    if not audio_dir.is_dir():
        issues.append("缺少 audio/ 文件夹（用于存放音乐源文件）")
    if not analysis_dir.is_dir():
        issues.append("缺少 analysis/ 文件夹（用于存放 *_剪辑数据.json）")
    if not catalog_path.is_file():
        issues.append("缺少 music_catalog.json（音乐快速配对索引文件）")

    loose_audio: List[str] = []
    loose_analysis: List[str] = []
    for p in lib_dir.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(lib_dir)
        top = rel.parts[0] if rel.parts else ""
        if p.name == "music_catalog.json" and len(rel.parts) == 1:
            continue
        if p.suffix.lower() in AUDIO_EXTS and top != "audio":
            loose_audio.append(str(rel).replace("\\", "/"))
        elif p.name.endswith(ANALYSIS_SUFFIX) and top != "analysis":
            loose_analysis.append(str(rel).replace("\\", "/"))

    if loose_audio:
        issues.append(f"有 {len(loose_audio)} 个音乐文件未放入 audio/ 目录")
    if loose_analysis:
        issues.append(f"有 {len(loose_analysis)} 个分析 JSON 未放入 analysis/ 目录")

    extra_paths = _find_extra_items(lib_dir)
    extra_items = [str(p.relative_to(lib_dir)).replace("\\", "/") + ("/" if p.is_dir() else "") for p in extra_paths]
    if extra_items:
        issues.append(f"存在 {len(extra_items)} 项与标准音乐库无关的文件/目录: {extra_items}")

    audio_files = sorted(
        [p for p in audio_dir.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
    ) if audio_dir.is_dir() else []
    analysis_files = sorted(
        [p for p in analysis_dir.iterdir() if p.is_file() and p.name.endswith(ANALYSIS_SUFFIX)]
    ) if analysis_dir.is_dir() else []

    analysis_by_stem = {
        _match_analysis_stem(p.name): p
        for p in analysis_files
        if _match_analysis_stem(p.name)
    }
    audio_without_analysis = [p.name for p in audio_files if p.stem not in analysis_by_stem]

    catalog_tracks: List[Dict[str, Any]] = []
    indexed_audio_paths = set()
    incomplete_meta = 0
    if catalog_path.is_file():
        try:
            cat = json.loads(catalog_path.read_text(encoding="utf-8-sig"))
            catalog_tracks = list(cat.get("tracks") or [])
            for t in catalog_tracks:
                ap = str(t.get("audio_path") or "").replace("\\", "/")
                if ap:
                    indexed_audio_paths.add(ap)
                if not t.get("bpm") or not t.get("mood") or not t.get("preview_segments"):
                    incomplete_meta += 1
        except Exception as exc:
            issues.append(f"music_catalog.json 解析失败: {exc}")

    unindexed_audio = [
        f"audio/{p.name}"
        for p in audio_files
        if f"audio/{p.name}" not in indexed_audio_paths
    ]
    if unindexed_audio:
        issues.append(f"有 {len(unindexed_audio)} 首音乐尚未登记到 music_catalog.json")

    return {
        "exists": True,
        "compliant": len(issues) == 0,
        "library_dir": str(lib_dir),
        "audio_count": len(audio_files),
        "analysis_count": len(analysis_files),
        "catalog_count": len(catalog_tracks),
        "incomplete_metadata_count": incomplete_meta,
        "loose_audio": loose_audio,
        "loose_analysis": loose_analysis,
        "extra_items": extra_items,
        "audio_without_analysis": audio_without_analysis,
        "unindexed_audio": unindexed_audio,
        "issues": issues,
    }


def organize_music_library(lib_dir: Path, clean_extra: bool = False) -> Dict[str, Any]:
    """按标准结构整理音乐库：归位散落文件、从 *_剪辑数据.json 补全 music_catalog.json，并可选清理无关文件。"""
    lib_dir = Path(lib_dir).resolve()
    lib_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = lib_dir / "audio"
    analysis_dir = lib_dir / "analysis"
    audio_dir.mkdir(exist_ok=True)
    analysis_dir.mkdir(exist_ok=True)

    moved_audio: List[Dict[str, str]] = []
    moved_analysis: List[Dict[str, str]] = []
    skipped_conflicts: List[str] = []

    for p in list(lib_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(lib_dir)
        top = rel.parts[0] if rel.parts else ""
        if p.name == "music_catalog.json" and len(rel.parts) == 1:
            continue
        if p.suffix.lower() in AUDIO_EXTS and top != "audio":
            dst = audio_dir / p.name
            if dst.exists():
                skipped_conflicts.append(str(rel))
            else:
                shutil.move(str(p), str(dst))
                moved_audio.append({"from": str(rel).replace("\\", "/"), "to": f"audio/{p.name}"})
        elif p.name.endswith(ANALYSIS_SUFFIX) and top != "analysis":
            dst = analysis_dir / p.name
            if dst.exists():
                skipped_conflicts.append(str(rel))
            else:
                shutil.move(str(p), str(dst))
                moved_analysis.append({"from": str(rel).replace("\\", "/"), "to": f"analysis/{p.name}"})

    # 读取已有 catalog，保留用户已手工填写的非空标签，自动补全空缺的 bpm/mood/scenes/preview_segments/notes
    catalog_path = lib_dir / "music_catalog.json"
    existing_catalog: Dict[str, Any] = {"library_name": lib_dir.name, "tracks": []}
    if catalog_path.is_file():
        try:
            loaded = json.loads(catalog_path.read_text(encoding="utf-8-sig"))
            if isinstance(loaded, dict):
                existing_catalog = loaded
        except Exception:
            pass

    existing_tracks: List[Dict[str, Any]] = list(existing_catalog.get("tracks") or [])
    by_audio_path: Dict[str, Dict[str, Any]] = {}
    used_ids = set()
    for t in existing_tracks:
        ap = str(t.get("audio_path") or "").replace("\\", "/")
        if ap:
            by_audio_path[ap] = t
        if t.get("id"):
            used_ids.add(str(t["id"]))

    analysis_by_stem = {
        _match_analysis_stem(p.name): p
        for p in analysis_dir.iterdir()
        if p.is_file() and _match_analysis_stem(p.name)
    }

    added_tracks: List[str] = []
    enriched_tracks: List[str] = []
    next_idx = 1
    for p in sorted(audio_dir.iterdir()):
        if not (p.is_file() and p.suffix.lower() in AUDIO_EXTS):
            continue
        rel_audio = f"audio/{p.name}"
        ana = analysis_by_stem.get(p.stem)
        rel_ana = f"analysis/{ana.name}" if ana else ""
        meta = _read_analysis_meta(ana) if ana else _read_analysis_meta(Path("__missing__"))

        if rel_audio in by_audio_path:
            entry = by_audio_path[rel_audio]
            changed = False
            if rel_ana and not entry.get("analysis_json"):
                entry["analysis_json"] = rel_ana
                changed = True
            for field in ("duration_sec", "bpm", "mood", "scenes", "preview_segments", "notes"):
                if meta.get(field) and not entry.get(field):
                    entry[field] = meta[field]
                    changed = True
            if changed:
                enriched_tracks.append(rel_audio)
            continue

        while f"bgm_{next_idx:03d}" in used_ids:
            next_idx += 1
        new_id = f"bgm_{next_idx:03d}"
        used_ids.add(new_id)
        next_idx += 1

        new_entry = {
            "id": new_id,
            "name": p.stem,
            "audio_path": rel_audio,
            "analysis_json": rel_ana,
            "duration_sec": meta["duration_sec"],
            "bpm": meta["bpm"],
            "mood": meta["mood"],
            "scenes": meta["scenes"],
            "has_vocals": False,
            "energy": "中",
            "enabled": True,
            "is_default": False,
            "priority": 3,
            "default_volume": 0.18,
            "preview_segments": meta["preview_segments"],
            "notes": meta["notes"],
        }
        existing_tracks.append(new_entry)
        added_tracks.append(rel_audio)

    existing_catalog["library_name"] = existing_catalog.get("library_name") or lib_dir.name
    existing_catalog["tracks"] = existing_tracks
    catalog_path.write_text(
        json.dumps(existing_catalog, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    removed_extra: List[str] = []
    if clean_extra:
        for item in _find_extra_items(lib_dir):
            rel_str = str(item.relative_to(lib_dir)).replace("\\", "/") + ("/" if item.is_dir() else "")
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
            removed_extra.append(rel_str)
            print(f"[已删除无关项] {rel_str}")

    after_status = inspect_music_library(lib_dir)
    return {
        "moved_audio": moved_audio,
        "moved_analysis": moved_analysis,
        "skipped_conflicts": skipped_conflicts,
        "added_to_catalog": added_tracks,
        "enriched_metadata": enriched_tracks,
        "removed_extra": removed_extra,
        "catalog_path": str(catalog_path),
        "status_after": after_status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="检查或按标准整理音乐库文件夹（audio/ + analysis/ + music_catalog.json）")
    parser.add_argument("--dir", type=Path, required=True, help="音乐库根目录路径")
    parser.add_argument("--check", action="store_true", help="只读检查音乐库是否符合目录规范")
    parser.add_argument("--organize", action="store_true", help="归位散落文件并从 *_剪辑数据.json 补全 music_catalog.json")
    parser.add_argument("--clean-extra", action="store_true", help="配合 --organize 使用：删除标准结构之外的无关文件与目录")
    args = parser.parse_args()

    if args.organize:
        res = organize_music_library(args.dir, clean_extra=bool(args.clean_extra))
    else:
        res = inspect_music_library(args.dir)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
