"""只读扫描口播视频、ASR字幕、逐字时间JSON与双阈值物理静音，生成scan_report.json。"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

from portable_paths import ensure_ffmpeg_in_path, require_tool  # noqa: E402

ensure_ffmpeg_in_path()

FILLER_CHARS_RE = re.compile(r"^[呃嗯啊哦唉呵唔嘛呢吧，。、；：？！…,.!?\s]+$")
STRIP_PUNCT_RE = re.compile(r"[，。、；：？！…,.!?;:\s“”\"'《》〈〉（）()—·\-]+")


def visual_hanzi_width(text: str) -> float:
    """估算中英混排屏幕显示宽度（1个汉字=1.0，1个ASCII字符≈0.45）。"""
    width = 0.0
    for ch in text:
        if ord(ch) < 128:
            width += 0.45
        else:
            width += 1.0
    return round(width, 2)


def parse_srt_timestamp(ts: str) -> float:
    """将 SRT 时间戳 HH:MM:SS,mmm 转为秒。"""
    ts = ts.strip().replace(".", ",")
    hms, ms = ts.split(",")
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_srt(srt_path: Path) -> List[Dict[str, Any]]:
    """解析标准 UTF-8 / UTF-8-SIG SRT 文件。"""
    raw = srt_path.read_text(encoding="utf-8-sig").strip()
    if not raw:
        return []
    blocks = re.split(r"\r?\n\r?\n+", raw)
    cues: List[Dict[str, Any]] = []
    time_re = re.compile(r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})")
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        ts_idx = 0
        cue_id = len(cues) + 1
        if lines[0].isdigit():
            cue_id = int(lines[0])
            ts_idx = 1
        if ts_idx >= len(lines):
            continue
        m = time_re.search(lines[ts_idx])
        if not m:
            continue
        start_sec = parse_srt_timestamp(m.group(1))
        end_sec = parse_srt_timestamp(m.group(2))
        text = " ".join(lines[ts_idx + 1 :]).strip()
        cues.append(
            {
                "id": cue_id,
                "start": round(start_sec, 4),
                "end": round(end_sec, 4),
                "duration": round(max(0.0, end_sec - start_sec), 4),
                "text": text,
                "clean_text": STRIP_PUNCT_RE.sub("", text),
                "visual_width": visual_hanzi_width(text.rstrip("，。；：、,.!?;:")),
            }
        )
    return cues


def resolve_tool(name: str) -> str:
    return require_tool(name)


def probe_video(video_path: Path) -> Dict[str, Any]:
    """调用 ffprobe 探测主视频流（跳过 mjpeg 封面流，处理 rotation 旋转）。"""
    cmd = [
        resolve_tool("ffprobe"),
        "-v",
        "error",
        "-show_entries",
        "format=duration,size:stream=index,codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,pix_fmt:stream_tags=rotate:stream_side_data=rotation:stream_disposition=attached_pic",
        "-of",
        "json",
        str(video_path),
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
    if res.returncode != 0:
        raise RuntimeError(f"ffprobe 探测失败: {res.stderr.strip()}")
    data = json.loads(res.stdout)
    streams = data.get("streams", [])
    video_stream = next(
        (
            s
            for s in streams
            if s.get("codec_type") == "video"
            and s.get("codec_name") != "mjpeg"
            and s.get("disposition", {}).get("attached_pic", 0) != 1
        ),
        None,
    )
    if not video_stream:
        raise RuntimeError(f"未找到有效主视频流: {video_path}")

    width = int(video_stream["width"])
    height = int(video_stream["height"])
    rotate = 0
    if "rotate" in video_stream.get("tags", {}):
        rotate = int(float(video_stream["tags"]["rotate"]))
    for sd in video_stream.get("side_data_list", []):
        if "rotation" in sd:
            rotate = int(float(sd["rotation"]))
    if abs(rotate) in (90, 270):
        width, height = height, width

    rate_str = video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "30/1"
    if "/" in rate_str:
        num, den = rate_str.split("/", 1)
        raw_fps = float(num) / float(den) if float(den) > 0 else 30.0
    else:
        raw_fps = float(rate_str)
    project_fps = max(1, int(round(raw_fps)))
    duration = float(data["format"]["duration"])
    file_size = int(data["format"].get("size") or video_path.stat().st_size)

    return {
        "codec_name": video_stream.get("codec_name", ""),
        "pix_fmt": video_stream.get("pix_fmt", ""),
        "width": width,
        "height": height,
        "orientation": "portrait" if width < height else "landscape",
        "rotate_deg": rotate,
        "raw_fps": round(raw_fps, 4),
        "project_fps": project_fps,
        "duration_sec": round(duration, 6),
        "file_size_bytes": file_size,
    }


def detect_silences(video_path: Path, noise_db: str, min_dur: float) -> List[Dict[str, float]]:
    """运行 ffmpeg silencedetect 提取物理静音区间列表。"""
    cmd = [
        resolve_tool("ffmpeg"),
        "-hide_banner",
        "-i",
        str(video_path),
        "-af",
        f"silencedetect=noise={noise_db}:d={min_dur}",
        "-vn",
        "-f",
        "null",
        "-",
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    # silencedetect 输出在 stderr 中，返回码通常为 0
    output = (res.stdout or "") + "\n" + (res.stderr or "")
    intervals: List[Dict[str, float]] = []
    cur_start: Optional[float] = None
    start_re = re.compile(r"silence_start:\s*([0-9eE\.\+\-]+)")
    end_re = re.compile(r"silence_end:\s*([0-9eE\.\+\-]+)\s*\|\s*silence_duration:\s*([0-9eE\.\+\-]+)")
    for line in output.splitlines():
        ms = start_re.search(line)
        if ms:
            cur_start = float(ms.group(1))
            continue
        me = end_re.search(line)
        if me:
            end_val = float(me.group(1))
            dur_val = float(me.group(2))
            start_val = cur_start if cur_start is not None else max(0.0, end_val - dur_val)
            intervals.append(
                {
                    "start": round(start_val, 4),
                    "end": round(end_val, 4),
                    "duration": round(dur_val, 4),
                }
            )
            cur_start = None
    return intervals


def find_silence_near(point_sec: float, silences: List[Dict[str, float]], tol: float = 0.25) -> Optional[Dict[str, float]]:
    """查找包含或紧邻某时间点的物理静音区间。"""
    best = None
    best_dist = 1e9
    for s in silences:
        if s["start"] - tol <= point_sec <= s["end"] + tol:
            dist = 0.0 if s["start"] <= point_sec <= s["end"] else min(abs(point_sec - s["start"]), abs(point_sec - s["end"]))
            if dist < best_dist:
                best_dist = dist
                best = s
    return best


def compute_native_pause_baseline(
    cues: List[Dict[str, Any]],
    chars: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """从原文非卡顿流畅停顿中采样说话人原生节奏基线。"""
    normal_gaps: List[float] = []
    for i in range(1, len(cues)):
        prev = cues[i - 1]
        cur = cues[i]
        gap = round(cur["start"] - prev["end"], 4)
        if FILLER_CHARS_RE.match(prev["text"]) or FILLER_CHARS_RE.match(cur["text"]):
            continue
        if 0.0 <= gap <= 1.5:
            normal_gaps.append(gap)

    micro_char_gaps: List[float] = []
    for i in range(1, len(chars)):
        gap = round(float(chars[i]["start_time"]) - float(chars[i - 1]["end_time"]), 4)
        if 0.04 <= gap <= 0.6:
            micro_char_gaps.append(gap)

    def summarize(vals: List[float]) -> Dict[str, Any]:
        if not vals:
            return {"count": 0}
        sv = sorted(vals)
        n = len(sv)
        return {
            "count": n,
            "min": round(sv[0], 3),
            "p25": round(sv[max(0, int(0.25 * (n - 1)))], 3),
            "median": round(statistics.median(sv), 3),
            "p75": round(sv[min(n - 1, int(0.75 * (n - 1)))], 3),
            "max": round(sv[-1], 3),
        }

    return {
        "inter_cue_gaps_0_to_1_5s": summarize(normal_gaps),
        "intra_cue_micro_gaps_0_04_to_0_6s": summarize(micro_char_gaps),
        "long_gaps_over_1_4s": [
            {
                "after_srt_id": cues[i - 1]["id"],
                "before_srt_id": cues[i]["id"],
                "gap_sec": round(cues[i]["start"] - cues[i - 1]["end"], 3),
                "span": f"{cues[i - 1]['end']:.2f}s -> {cues[i]['start']:.2f}s",
                "prev_text": cues[i - 1]["text"],
                "next_text": cues[i]["text"],
            }
            for i in range(1, len(cues))
            if cues[i]["start"] - cues[i - 1]["end"] >= 1.4
        ],
    }


def compute_speech_rate(cues: List[Dict[str, Any]], chars: List[Dict[str, Any]]) -> Dict[str, Any]:
    """统计有效语速（字/分钟）并给出节奏评级与倍速建议。"""
    speaking_sec = sum(max(0.0, float(c["end"]) - float(c["start"])) for c in cues)
    effective_chars = sum(
        len(STRIP_PUNCT_RE.sub("", str(c.get("text", ""))))
        for c in cues
        if not FILLER_CHARS_RE.match(str(c.get("text", "")))
    )
    if effective_chars == 0 and chars:
        effective_chars = len(chars)

    if speaking_sec <= 0.1 or effective_chars == 0:
        return {
            "effective_chars": effective_chars,
            "speaking_duration_sec": round(speaking_sec, 2),
            "effective_wpm": 0,
            "pace_level": "无法评估",
            "recommended_speed": 1.0,
            "suggestion": "未检测到足够有效语音，保持 1.0x 原速",
        }

    wpm = int(round(effective_chars / speaking_sec * 60.0))
    if wpm < 180:
        level = "偏慢"
        rec_speed = 1.1
        suggestion = "语速偏慢，短视频建议 1.1x 加速（偏好沉稳风格可保持 1.0x）"
    elif wpm <= 230:
        level = "适中（主流舒适区）"
        rec_speed = 1.0
        suggestion = "语速适中，建议保持 1.0x 原速"
    elif wpm <= 270:
        level = "偏快（强节奏）"
        rec_speed = 1.0
        suggestion = "节奏紧凑，保持 1.0x 并在重点处留气口"
    else:
        level = "过快"
        rec_speed = 1.0
        suggestion = "语速过快，保持 1.0x 并适当保留句间停顿"

    return {
        "effective_chars": effective_chars,
        "speaking_duration_sec": round(speaking_sec, 2),
        "effective_wpm": wpm,
        "pace_level": level,
        "recommended_speed": rec_speed,
        "suggestion": suggestion,
    }


def resolve_inputs(args: argparse.Namespace) -> Tuple[Path, Path, Path, Path]:
    """根据参数解析 video / srt / chars / report 输出路径。"""
    if args.dir:
        base_dir = Path(args.dir)
        if not base_dir.is_dir():
            raise FileNotFoundError(f"输入目录不存在: {base_dir}")
        mp4s = [p for p in base_dir.iterdir() if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".mkv"}]
        if not mp4s:
            raise FileNotFoundError(f"目录顶层未找到视频文件（*.mp4/*.mov/*.mkv），若在子目录请显式传 --video: {base_dir}")
        if len(mp4s) > 1 and not args.video:
            raise RuntimeError(f"目录中存在多个视频文件，请通过 --video 指定明确文件: {[p.name for p in mp4s]}")
        video_path = Path(args.video) if args.video else mp4s[0]
        stem = video_path.stem
        srt_path = Path(args.srt) if args.srt else base_dir / f"{stem}.srt"
        char_path = Path(args.chars) if args.chars else base_dir / f"{stem}_逐字时间.json"
        out_path = Path(args.out) if args.out else base_dir / "scan_report.json"
        return video_path, srt_path, char_path, out_path

    if not args.video:
        raise ValueError("必须提供 --dir 或 --video 参数")
    video_path = Path(args.video)
    stem_dir = video_path.parent
    stem = video_path.stem
    srt_path = Path(args.srt) if args.srt else stem_dir / f"{stem}.srt"
    char_path = Path(args.chars) if args.chars else stem_dir / f"{stem}_逐字时间.json"
    out_path = Path(args.out) if args.out else stem_dir / "scan_report.json"
    return video_path, srt_path, char_path, out_path


def run_scan(video_path: Path, srt_path: Path, char_path: Path, out_path: Path) -> Dict[str, Any]:
    for req in (video_path, srt_path, char_path):
        if not req.exists():
            raise FileNotFoundError(
                f"必需输入文件不存在: {req}。若目录尚无 .srt 或 *_逐字时间.json，请先执行 Step 0：python scripts/transcribe_talking_head.py --dir \"{video_path.parent}\""
            )

    video_info = probe_video(video_path)
    hard_limit = 13.0 if video_info["orientation"] == "portrait" else 16.0

    cues = parse_srt(srt_path)
    chars: List[Dict[str, Any]] = json.loads(char_path.read_text(encoding="utf-8-sig"))

    silences_35 = detect_silences(video_path, "-35dB", 0.12)
    silences_28 = detect_silences(video_path, "-28dB", 0.05)

    zero_dur_chars = [
        {
            "index": idx,
            "text": ch.get("text", ""),
            "start_time": ch.get("start_time"),
            "end_time": ch.get("end_time"),
        }
        for idx, ch in enumerate(chars)
        if float(ch.get("end_time", 0.0)) - float(ch.get("start_time", 0.0)) < 0.005
    ]

    annotated_cues = []
    filler_cues = []
    flash_cues = []
    overlength_cues = []

    for i, c in enumerate(cues):
        gap_before = round(c["start"] - cues[i - 1]["end"], 3) if i > 0 else round(c["start"], 3)
        near_sil_before = find_silence_near(c["start"], silences_35, tol=0.20)
        near_sil_after = find_silence_near(c["end"], silences_35, tol=0.20)
        item = {
            **c,
            "gap_before_sec": gap_before,
            "silence_35db_before": near_sil_before,
            "silence_35db_after": near_sil_after,
        }
        annotated_cues.append(item)

        if FILLER_CHARS_RE.match(c["text"]):
            filler_cues.append({"id": c["id"], "start": c["start"], "end": c["end"], "text": c["text"]})
        if c["duration"] < 0.45:
            flash_cues.append({"id": c["id"], "start": c["start"], "end": c["end"], "duration": c["duration"], "text": c["text"]})
        if c["visual_width"] > hard_limit:
            overlength_cues.append({"id": c["id"], "visual_width": c["visual_width"], "text": c["text"]})

    report: Dict[str, Any] = {
        "inputs": {
            "video_path": str(video_path),
            "srt_path": str(srt_path),
            "char_json_path": str(char_path),
        },
        "video_info": video_info,
        "subtitle_limit_for_orientation": hard_limit,
        "counts": {
            "srt_cues": len(cues),
            "char_tokens": len(chars),
            "zero_duration_char_tokens": len(zero_dur_chars),
            "silence_intervals_35db": len(silences_35),
            "silence_intervals_28db": len(silences_28),
            "filler_only_cues": len(filler_cues),
            "short_flash_cues_under_0_45s": len(flash_cues),
            "overlength_cues": len(overlength_cues),
        },
        "speech_rate": compute_speech_rate(cues, chars),
        "native_pause_baseline": compute_native_pause_baseline(cues, chars),
        "warnings": {
            "zero_duration_chars_sample": zero_dur_chars[:30],
            "filler_only_cues": filler_cues,
            "short_flash_cues": flash_cues,
            "overlength_cues": overlength_cues,
        },
        "silences_35db": silences_35,
        "silences_28db": silences_28,
        "cues": annotated_cues,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="扫描口播原片、SRT、逐字时间与物理静音区间")
    parser.add_argument("--dir", help="包含同前缀 video/srt/_逐字时间.json 的目录")
    parser.add_argument("--video", help="显式指定源视频路径")
    parser.add_argument("--srt", help="显式指定原 SRT 路径")
    parser.add_argument("--chars", help="显式指定 *_逐字时间.json 路径")
    parser.add_argument("--out", help="输出 scan_report.json 路径")
    args = parser.parse_args()

    video_path, srt_path, char_path, out_path = resolve_inputs(args)
    report = run_scan(video_path, srt_path, char_path, out_path)

    compact = {
        "scan_report": str(out_path),
        "video_info": report["video_info"],
        "subtitle_limit_for_orientation": report["subtitle_limit_for_orientation"],
        "counts": report["counts"],
        "speech_rate": report["speech_rate"],
        "native_pause_baseline": report["native_pause_baseline"],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

