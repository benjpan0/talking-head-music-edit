"""口播音频/视频独立转写脚本：支持云端 API（DashScope / 腾讯云极速版 / Groq / OpenAI兼容）与本地离线 faster-whisper，输出与下游完全兼容的 .srt / .txt / *_逐字时间.json。"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import hmac
import importlib.util
import json
import os
import re
import site
import subprocess
import sys
import tempfile
import time
import urllib.parse
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
    detect_qwen3_asr,
    ensure_ffmpeg_in_path,
    get_whisper_models_cache_dir,
    load_machine_profile,
    require_tool,
    resolve_local_whisper_model,
    resolve_whisper_python,
)

MEDIA_EXTS = {".mp4", ".mov", ".mkv", ".mp3", ".wav", ".m4a", ".flac", ".aac"}
CORE_ALNUM_RE = re.compile(r"[A-Za-z0-9]+(?:[.\-+][A-Za-z0-9]+)*")
DEFAULT_ZH_PROMPT = "以下是一段普通话口播内容，请使用简体中文并包含准确标点符号进行转录。"
CLAUSE_SPLIT_PUNCT = set("。？！?!；;，,")


def is_core_token_char(ch: str) -> bool:
    """与 subtitle_processor 保持 100% 一致的核心文字判定（汉字或字母数字）。"""
    return ("\u4e00" <= ch <= "\u9fff") or ch.isalnum()


def format_srt_ms(sec: float) -> str:
    """将秒数转为进位安全的 SRT 时间戳 HH:MM:SS,mmm。"""
    total_ms = max(0, int(round(sec * 1000.0)))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def extract_core_units(text: str) -> List[str]:
    """
    将原始词语文本剥除标点与空白后拆分为标准核心单元：
    - 连续 ASCII 字母/数字保留为整体单元（如 AI、2026、4S）；
    - 中文及其他字母按单字拆开，保证与 subtitle_processor._match_text_to_char_spans 零误差对齐。
    """
    units: List[str] = []
    idx = 0
    n = len(text)
    while idx < n:
        m = CORE_ALNUM_RE.match(text, idx)
        if m:
            units.append(m.group(0))
            idx = m.end()
            continue
        ch = text[idx]
        if is_core_token_char(ch):
            units.append(ch)
        idx += 1
    return units


def expand_word_to_core_tokens(
    raw_text: str,
    start_sec: float,
    end_sec: float,
) -> List[Dict[str, Any]]:
    """把单个词级时间片（可能含空格、标点或多字词组）展开为规范 *_逐字时间.json 条目。"""
    units = extract_core_units(raw_text)
    if not units:
        return []
    st = max(0.0, float(start_sec))
    et = max(st, float(end_sec))
    if len(units) == 1:
        return [{"text": units[0], "start_time": round(st, 4), "end_time": round(et, 4)}]

    dur = max(0.0, et - st)
    step = dur / len(units)
    out: List[Dict[str, Any]] = []
    for i, u in enumerate(units):
        ust = st + i * step
        uet = st + (i + 1) * step if i < len(units) - 1 else et
        out.append({"text": u, "start_time": round(ust, 4), "end_time": round(max(ust, uet), 4)})
    return out


def _align_subclauses_to_char_tokens(
    clauses: List[str],
    seg_start: float,
    seg_end: float,
    seg_chars: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """把被长句切分出的多个短句子块尽可能贴合真实字级时间戳分配起止时间。"""
    if len(clauses) <= 1 or not seg_chars:
        return [{"text": "".join(clauses).strip(), "start": round(seg_start, 4), "end": round(max(seg_start + 0.05, seg_end), 4)}]

    result: List[Dict[str, Any]] = []
    cursor = 0
    total_chars = len(seg_chars)
    total_core = sum(len(extract_core_units(c)) for c in clauses) or 1

    for idx, cl in enumerate(clauses):
        units = extract_core_units(cl)
        take = len(units)
        if cursor < total_chars and take > 0:
            sub_slice = seg_chars[cursor : min(total_chars, cursor + take)]
            cursor += len(sub_slice)
            st = float(sub_slice[0]["start_time"])
            et = float(sub_slice[-1]["end_time"])
            if idx == 0:
                st = min(st, seg_start) if seg_start <= st else st
            if idx == len(clauses) - 1:
                et = max(et, seg_end)
            result.append({"text": cl.strip(), "start": round(st, 4), "end": round(max(st + 0.05, et), 4)})
        else:
            prev_end = result[-1]["end"] if result else seg_start
            prop_dur = max(0.25, (seg_end - prev_end) * (len(units) / max(1, total_core)))
            et = min(seg_end, prev_end + prop_dur) if idx < len(clauses) - 1 else max(seg_end, prev_end + 0.1)
            result.append({"text": cl.strip(), "start": round(prev_end, 4), "end": round(max(prev_end + 0.05, et), 4)})
    return result


def split_long_segment_for_srt(
    text: str,
    seg_start: float,
    seg_end: float,
    char_tokens: Sequence[Dict[str, Any]],
    max_cue_chars: int = 26,
) -> List[Dict[str, Any]]:
    """确保导出的原片 .srt 每条具备自然句读长度，方便 Step 2 宏观阅读与排查。"""
    cleaned = text.strip()
    if not cleaned:
        return []
    if len(cleaned) <= max_cue_chars:
        return [{"text": cleaned, "start": round(seg_start, 4), "end": round(max(seg_start + 0.05, seg_end), 4)}]

    pieces: List[str] = []
    buf: List[str] = []
    for ch in cleaned:
        buf.append(ch)
        if ch in CLAUSE_SPLIT_PUNCT and len(buf) >= 6:
            pieces.append("".join(buf))
            buf = []
    if buf:
        if pieces and len(buf) <= 4:
            pieces[-1] += "".join(buf)
        else:
            pieces.append("".join(buf))

    if len(pieces) <= 1:
        return [{"text": cleaned, "start": round(seg_start, 4), "end": round(max(seg_start + 0.05, seg_end), 4)}]

    seg_chars = [
        c
        for c in char_tokens
        if float(c["end_time"]) >= seg_start - 0.15 and float(c["start_time"]) <= seg_end + 0.15
    ]
    return _align_subclauses_to_char_tokens(pieces, seg_start, seg_end, seg_chars)


def normalize_asr_result(
    segments: Sequence[Dict[str, Any]],
    raw_words: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], bool]:
    """
    将任意 ASR 后端的 (segments, raw_words) 归一化为：
    - char_tokens: 标准 *_逐字时间.json（单调非递减，剥除标点空格）
    - srt_cues: 标准自然短句 SRT 条目
    - estimated_words: 是否缺失词级时间戳而使用句级插值（严格透明标识，不静默伪造）
    """
    char_tokens: List[Dict[str, Any]] = []
    estimated_words = False

    if raw_words:
        for w in raw_words:
            txt = str(w.get("text", w.get("word", "")))
            st = float(w.get("start", w.get("start_time", 0.0)))
            et = float(w.get("end", w.get("end_time", st)))
            char_tokens.extend(expand_word_to_core_tokens(txt, st, et))
    else:
        estimated_words = True
        for seg in segments:
            txt = str(seg.get("text", ""))
            st = float(seg.get("start", 0.0))
            et = float(seg.get("end", st + 0.1))
            char_tokens.extend(expand_word_to_core_tokens(txt, st, et))

    # 强制时间轴单调不回退
    last_t = 0.0
    for tok in char_tokens:
        if tok["start_time"] < last_t:
            tok["start_time"] = round(last_t, 4)
        if tok["end_time"] < tok["start_time"]:
            tok["end_time"] = tok["start_time"]
        last_t = tok["start_time"]

    srt_cues: List[Dict[str, Any]] = []
    if segments:
        for seg in segments:
            seg_text = str(seg.get("text", "")).strip()
            if not seg_text:
                continue
            seg_st = float(seg.get("start", 0.0))
            seg_et = float(seg.get("end", seg_st + 0.1))
            srt_cues.extend(split_long_segment_for_srt(seg_text, seg_st, seg_et, char_tokens))
    elif char_tokens:
        # 仅返回 words 无 segments 时，按微停顿 > 0.30s 或长度自动汇成 SRT
        buf: List[Dict[str, Any]] = []
        for idx, tok in enumerate(char_tokens):
            if buf:
                gap = tok["start_time"] - buf[-1]["end_time"]
                if gap > 0.30 or len(buf) >= 18:
                    srt_cues.append(
                        {
                            "text": "".join(x["text"] for x in buf),
                            "start": buf[0]["start_time"],
                            "end": max(buf[0]["start_time"] + 0.05, buf[-1]["end_time"]),
                        }
                    )
                    buf = []
            buf.append(tok)
        if buf:
            srt_cues.append(
                {
                    "text": "".join(x["text"] for x in buf),
                    "start": buf[0]["start_time"],
                    "end": max(buf[0]["start_time"] + 0.05, buf[-1]["end_time"]),
                }
            )

    for idx, cue in enumerate(srt_cues, start=1):
        cue["id"] = idx
        if idx > 1 and cue["start"] < srt_cues[idx - 2]["end"]:
            cue["start"] = srt_cues[idx - 2]["end"]
        if cue["end"] <= cue["start"]:
            cue["end"] = round(cue["start"] + 0.05, 4)

    return char_tokens, srt_cues, estimated_words


def write_transcription_outputs(
    out_dir: Path,
    stem: str,
    srt_cues: Sequence[Dict[str, Any]],
    char_tokens: Sequence[Dict[str, Any]],
    meta: Dict[str, Any],
    write_txt: bool = False,
    write_meta: bool = False,
) -> Dict[str, str]:
    """默认严格写出兼容度最高的 2 个核心文件（.srt 与 *_逐字时间.json），可选导出 .txt 与 meta JSON。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    srt_path = out_dir / f"{stem}.srt"
    char_path = out_dir / f"{stem}_逐字时间.json"

    srt_blocks = [
        f"{c['id']}\n{format_srt_ms(float(c['start']))} --> {format_srt_ms(float(c['end']))}\n{c['text']}"
        for c in srt_cues
    ]
    srt_path.write_text("\n\n".join(srt_blocks) + ("\n" if srt_blocks else ""), encoding="utf-8")
    char_path.write_text(json.dumps(list(char_tokens), ensure_ascii=False, indent=2), encoding="utf-8")

    outputs: Dict[str, str] = {
        "srt": str(srt_path),
        "char_json": str(char_path),
    }
    if write_txt:
        txt_path = out_dir / f"{stem}.txt"
        txt_path.write_text("\n".join(str(c["text"]) for c in srt_cues) + ("\n" if srt_cues else ""), encoding="utf-8")
        outputs["txt"] = str(txt_path)
    if write_meta:
        meta_path = out_dir / f"{stem}_transcribe_meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        outputs["meta_json"] = str(meta_path)

    return outputs


def probe_media_duration(media_path: Path) -> float:
    ffprobe = require_tool("ffprobe")
    res = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "json", str(media_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    if res.returncode != 0:
        raise RuntimeError(f"ffprobe 探测媒体时长失败: {res.stderr.strip()}")
    data = json.loads(res.stdout)
    return float((data.get("format") or {}).get("duration") or 0.0)


def detect_silence_midpoints(media_path: Path, noise_db: str = "-30dB", min_dur: float = 0.15) -> List[float]:
    ffmpeg = require_tool("ffmpeg")
    res = subprocess.run(
        [ffmpeg, "-i", str(media_path), "-af", f"silencedetect=noise={noise_db}:d={min_dur}", "-f", "null", "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    start_re = re.compile(r"silence_start:\s*([0-9eE\.\+\-]+)")
    end_re = re.compile(r"silence_end:\s*([0-9eE\.\+\-]+)")
    mids: List[float] = []
    cur_start: Optional[float] = None
    for line in res.stderr.splitlines():
        ms = start_re.search(line)
        if ms:
            cur_start = float(ms.group(1))
            continue
        me = end_re.search(line)
        if me and cur_start is not None:
            end_t = float(me.group(1))
            if end_t >= cur_start:
                mids.append(round(0.5 * (cur_start + end_t), 3))
            cur_start = None
    return mids


def plan_silence_chunks(
    duration_sec: float,
    max_chunk_sec: float,
    silence_midpoints: Sequence[float],
) -> List[Tuple[float, float]]:
    """按静音中点切分长音视频（避免一刀砍在字中间），返回 [(chunk_start, chunk_end), ...]。"""
    if duration_sec <= max_chunk_sec or max_chunk_sec <= 0:
        return [(0.0, max(0.1, round(duration_sec, 3)))]

    chunks: List[Tuple[float, float]] = []
    cur = 0.0
    mids = sorted(silence_midpoints)
    search_window = min(45.0, max_chunk_sec * 0.25)

    while cur + max_chunk_sec < duration_sec - 1.0:
        target = cur + max_chunk_sec
        cands = [m for m in mids if (target - search_window) <= m <= target]
        split_at = cands[-1] if cands else target
        if split_at <= cur + 5.0:
            split_at = target
        chunks.append((round(cur, 3), round(split_at, 3)))
        cur = split_at

    chunks.append((round(cur, 3), round(duration_sec, 3)))
    return chunks


def extract_audio_chunk(
    media_path: Path,
    out_audio_path: Path,
    codec_fmt: str,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
) -> Path:
    """用 ffmpeg 提取轻量单声道 16kHz 音频片段（pcm_s16le wav/pcm 或 48k mp3）。"""
    ffmpeg = require_tool("ffmpeg")
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
    if start_sec is not None:
        cmd.extend(["-ss", f"{start_sec:.3f}"])
    if end_sec is not None:
        cmd.extend(["-to", f"{end_sec:.3f}"])
    cmd.extend(["-i", str(media_path), "-vn", "-ac", "1", "-ar", "16000"])
    if codec_fmt == "wav":
        cmd.extend(["-acodec", "pcm_s16le", "-f", "wav"])
    elif codec_fmt == "pcm":
        cmd.extend(["-acodec", "pcm_s16le", "-f", "s16le"])
    elif codec_fmt == "mp3":
        cmd.extend(["-acodec", "libmp3lame", "-b:a", "48k", "-f", "mp3"])
    else:
        raise ValueError(f"不支持的音频提取格式: {codec_fmt}")
    cmd.append(str(out_audio_path))
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="ignore")
    if res.returncode != 0 or not out_audio_path.is_file():
        raise RuntimeError(f"ffmpeg 提取音频失败 ({codec_fmt}): {res.stderr.strip()}")
    return out_audio_path


def _inject_windows_nvidia_dll_paths() -> List[str]:
    """在 Windows 下自动发现 pip 安装的 nvidia-cublas-cu12 / nvidia-cudnn-cu12 bin 目录并注入加载路径。"""
    if not sys.platform.startswith("win"):
        return []
    added: List[str] = []
    roots: List[Path] = []
    for p in list(site.getsitepackages()) + [site.getusersitepackages()]:
        if p:
            roots.append(Path(p) / "nvidia")
    for nv_root in roots:
        if not nv_root.is_dir():
            continue
        for sub in ("cublas", "cudnn", "cuda_runtime"):
            bin_dir = nv_root / sub / "bin"
            if bin_dir.is_dir():
                bstr = str(bin_dir)
                cur_path = os.environ.get("PATH", "")
                if bstr not in cur_path.split(os.pathsep):
                    os.environ["PATH"] = bstr + os.pathsep + cur_path
                try:
                    os.add_dll_directory(bstr)
                except Exception:
                    pass
                added.append(bstr)
    return added


def transcribe_faster_whisper(
    audio_path: Path,
    model_name: str,
    language: str,
    device: str,
    compute_type: str,
    hf_endpoint: str,
    initial_prompt: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """离线本地 faster-whisper 推理，支持本地预置模型、国内 HF 镜像与 CUDA->CPU int8 自动兜底。"""
    if not os.environ.get("HF_ENDPOINT") and hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint
    dll_dirs = _inject_windows_nvidia_dll_paths()

    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "当前 Python 未安装 faster-whisper。请运行引导脚本自动安装到外部隔离环境：\n"
            f'  python "{SCRIPT_DIR / "bootstrap_faster_whisper.py"}" --download-model {model_name}'
        ) from exc

    effective_model = resolve_local_whisper_model(model_name)
    is_local_dir = Path(effective_model).is_dir()
    custom_hf_cache = os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE")
    download_root: Optional[str] = None
    if not is_local_dir and not custom_hf_cache:
        cache_dir = get_whisper_models_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        download_root = str(cache_dir)

    def _run_model(dev: str, ctype: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        print(
            f"[faster-whisper] 加载模型 '{effective_model}' "
            f"(device={dev}, compute_type={ctype}, HF_ENDPOINT={os.environ.get('HF_ENDPOINT', 'default')})"
        )
        model = WhisperModel(effective_model, device=dev, compute_type=ctype, download_root=download_root)
        seg_iter, info = model.transcribe(
            str(audio_path),
            language=language or None,
            vad_filter=True,
            word_timestamps=True,
            initial_prompt=initial_prompt or None,
        )
        segments_out: List[Dict[str, Any]] = []
        words_out: List[Dict[str, Any]] = []
        for seg in seg_iter:
            segments_out.append({"start": float(seg.start), "end": float(seg.end), "text": str(seg.text).strip()})
            if getattr(seg, "words", None):
                for w in seg.words:
                    words_out.append(
                        {"word": str(w.word), "start": float(w.start), "end": float(w.end)}
                    )
        return segments_out, words_out, {
            "effective_device": dev,
            "effective_compute_type": ctype,
            "resolved_model": effective_model,
            "models_cache_dir": download_root,
            "detected_language": getattr(info, "language", language),
            "language_probability": round(float(getattr(info, "language_probability", 1.0)), 4),
            "injected_nvidia_dll_dirs": dll_dirs,
        }

    if device == "auto":
        try:
            ctype = "float16" if compute_type == "auto" else compute_type
            return _run_model("cuda", ctype)
        except Exception as exc:
            print(f"[提示] CUDA 模式初始化或推理未通过（{exc}），自动切换至 CPU (int8) 模式...")
            return _run_model("cpu", "int8")

    ctype = ("float16" if device == "cuda" else "int8") if compute_type == "auto" else compute_type
    return _run_model(device, ctype)


def _run_faster_whisper_with_dispatch(
    audio_path: Path,
    model_name: str,
    language: str,
    device: str,
    compute_type: str,
    hf_endpoint: str,
    initial_prompt: str,
    tmp_dir: Path,
    whisper_python_override: Optional[str] = None,
    auto_setup: bool = False,
    pip_index: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """调度当前 Python 或外部隔离 whisper-venv 执行 faster-whisper；必要时按需触发引导安装。"""
    whisper_py = resolve_whisper_python(explicit=whisper_python_override)
    if not whisper_py and auto_setup:
        from bootstrap_faster_whisper import DEFAULT_TUNA_PYPI, ensure_faster_whisper

        setup_res = ensure_faster_whisper(
            pip_index=pip_index or DEFAULT_TUNA_PYPI,
            hf_endpoint=hf_endpoint,
        )
        whisper_py = str(setup_res["whisper_python"])

    if not whisper_py:
        bootstrap_path = SCRIPT_DIR / "bootstrap_faster_whisper.py"
        raise RuntimeError(
            "未在当前 Python 或隔离缓存环境检测到 faster-whisper。\n"
            "请任选以下方式完成一次轻量安装（不污染技能目录）：\n"
            f'  方式 1（推荐，独立隔离环境 + 国内镜像）：python "{bootstrap_path}" --download-model {model_name}\n'
            "  方式 2（本次直接自动安装并转写）：在当前命令追加 --auto-setup"
        )

    if Path(whisper_py).resolve() == Path(sys.executable).resolve():
        segs, wds, info = transcribe_faster_whisper(
            audio_path=audio_path,
            model_name=model_name,
            language=language,
            device=device,
            compute_type=compute_type,
            hf_endpoint=hf_endpoint,
            initial_prompt=initial_prompt,
        )
        info["worker_python"] = whisper_py
        return segs, wds, info

    raw_out_json = tmp_dir / "fw_worker_raw.json"
    env = os.environ.copy()
    if not env.get("HF_ENDPOINT") and hf_endpoint:
        env["HF_ENDPOINT"] = hf_endpoint

    cmd = [
        whisper_py,
        str(Path(__file__).resolve()),
        "--_fw-worker",
        "--_audio",
        str(audio_path),
        "--_raw-out",
        str(raw_out_json),
        "--model",
        model_name,
        "--language",
        language,
        "--device",
        device,
        "--compute-type",
        compute_type,
        "--hf-endpoint",
        hf_endpoint,
        "--prompt",
        initial_prompt,
    ]
    print(f"[faster-whisper] 调度隔离环境执行推理：{whisper_py}")
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0 or not raw_out_json.is_file():
        raise RuntimeError(f"隔离环境 faster-whisper 推理失败（退出码 {proc.returncode}）：{whisper_py}")

    payload = json.loads(raw_out_json.read_text(encoding="utf-8-sig"))
    info = dict(payload.get("info") or {})
    info["worker_python"] = whisper_py
    return list(payload.get("segments") or []), list(payload.get("words") or []), info


def transcribe_openai_compatible_chunk(
    audio_chunk: Path,
    endpoint: str,
    api_key: str,
    model_name: str,
    language: str,
    prompt: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """调用 Groq / OpenAI / OpenAI-compatible /v1/audio/transcriptions 接口。"""
    import requests

    headers = {"Authorization": f"Bearer {api_key}"}
    data: List[Tuple[str, str]] = [
        ("model", model_name),
        ("response_format", "verbose_json"),
        ("timestamp_granularities[]", "word"),
        ("timestamp_granularities[]", "segment"),
    ]
    if language:
        data.append(("language", language))
    if prompt:
        data.append(("prompt", prompt))

    with audio_chunk.open("rb") as f:
        files = {"file": (audio_chunk.name, f, "audio/mpeg")}
        resp = requests.post(endpoint, headers=headers, files=files, data=data, timeout=300)

    if resp.status_code >= 400:
        raise RuntimeError(f"转写 API 请求失败 ({resp.status_code}): {resp.text[:500]}")

    payload = resp.json()
    segments: List[Dict[str, Any]] = []
    words: List[Dict[str, Any]] = []

    for seg in payload.get("segments") or []:
        segments.append(
            {
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": str(seg.get("text", "")).strip(),
            }
        )
        for w in seg.get("words") or []:
            words.append(
                {
                    "word": str(w.get("word", w.get("text", ""))),
                    "start": float(w.get("start", 0.0)),
                    "end": float(w.get("end", 0.0)),
                }
            )

    if not words and payload.get("words"):
        for w in payload["words"]:
            words.append(
                {
                    "word": str(w.get("word", w.get("text", ""))),
                    "start": float(w.get("start", 0.0)),
                    "end": float(w.get("end", 0.0)),
                }
            )

    if not segments and payload.get("text"):
        max_end = max((w["end"] for w in words), default=0.0)
        segments.append({"start": 0.0, "end": max_end, "text": str(payload["text"]).strip()})

    return segments, words


def transcribe_tencent_flash_chunk(
    audio_chunk: Path,
    appid: str,
    secret_id: str,
    secret_key: str,
    engine_type: str,
    voice_format: str = "mp3",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """调用腾讯云录音文件识别极速版（免 OSS，直接 POST 音频并开启 word_info=1 词级时间戳）。"""
    import requests

    host = "asr.cloud.tencent.com"
    path = f"/asr/flash/v1/{appid}"
    params = {
        "engine_type": engine_type,
        "first_channel_only": 1,
        "secretid": secret_id,
        "timestamp": int(time.time()),
        "voice_format": voice_format,
        "word_info": 1,
    }
    sorted_items = sorted(params.items(), key=lambda x: x[0])
    query_str = urllib.parse.urlencode(sorted_items)
    sign_source = f"POST{host}{path}?{query_str}"
    digest = hmac.new(secret_key.encode("utf-8"), sign_source.encode("utf-8"), hashlib.sha1).digest()
    auth = base64.b64encode(digest).decode("utf-8")

    headers = {
        "Host": host,
        "Authorization": auth,
        "Content-Type": "application/octet-stream",
    }
    url = f"https://{host}{path}?{query_str}"
    audio_bytes = audio_chunk.read_bytes()
    resp = requests.post(url, headers=headers, data=audio_bytes, timeout=300)
    if resp.status_code >= 400:
        raise RuntimeError(f"腾讯云极速版 ASR HTTP 失败 ({resp.status_code}): {resp.text[:500]}")

    payload = resp.json()
    if int(payload.get("code", -1)) != 0:
        raise RuntimeError(f"腾讯云极速版 ASR 返回错误 code={payload.get('code')}: {payload.get('message')}")

    segments: List[Dict[str, Any]] = []
    words: List[Dict[str, Any]] = []
    flash_list = payload.get("flash_result") or []
    channel = flash_list[0] if flash_list else {}
    for sent in channel.get("sentence_list") or []:
        st = float(sent.get("start_time", 0)) / 1000.0
        et = float(sent.get("end_time", 0)) / 1000.0
        txt = str(sent.get("text", "")).strip()
        if txt:
            segments.append({"start": st, "end": max(st, et), "text": txt})
        for w in sent.get("word_list") or []:
            wst = float(w.get("start_time", 0)) / 1000.0
            wet = float(w.get("end_time", 0)) / 1000.0
            words.append({"word": str(w.get("word", "")), "start": wst, "end": max(wst, wet)})

    return segments, words


async def _dashscope_ws_transcribe_async(
    pcm_path: Path,
    api_key: str,
    model_name: str,
    language_hints: List[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """通过 WebSocket 双工协议调用阿里云百炼 Paraformer-realtime-v2（支持字/词级时间戳，免 OSS）。"""
    ws_url = "wss://dashscope.aliyuncs.com/api-ws/v1/inference"
    task_id = uuid.uuid4().hex
    pcm_bytes = pcm_path.read_bytes()

    run_task_msg = {
        "header": {
            "action": "run-task",
            "task_id": task_id,
            "streaming": "duplex",
        },
        "payload": {
            "task_group": "audio",
            "task": "asr",
            "function": "recognition",
            "model": model_name,
            "parameters": {
                "format": "pcm",
                "sample_rate": 16000,
                "disfluency_removal_enabled": False,
                "language_hints": language_hints,
            },
            "input": {},
        },
    }
    finish_task_msg = {
        "header": {
            "action": "finish-task",
            "task_id": task_id,
            "streaming": "duplex",
        },
        "payload": {"input": {}},
    }

    segments: List[Dict[str, Any]] = []
    words: List[Dict[str, Any]] = []

    def _handle_server_event(msg_text: str) -> str:
        ev = json.loads(msg_text)
        header = ev.get("header") or {}
        event_name = str(header.get("event") or "")
        if event_name == "task-failed":
            raise RuntimeError(
                f"DashScope 任务失败: code={header.get('error_code')} msg={header.get('error_message')}"
            )
        if event_name == "result-generated":
            sent = ((ev.get("payload") or {}).get("output") or {}).get("sentence") or {}
            if sent.get("sentence_end"):
                st = float(sent.get("begin_time") or 0) / 1000.0
                et = float(sent.get("end_time") or 0) / 1000.0
                txt = str(sent.get("text") or "").strip()
                if txt:
                    segments.append({"start": st, "end": max(st, et), "text": txt})
                for w in sent.get("words") or []:
                    wst = float(w.get("begin_time") or 0) / 1000.0
                    wet = float(w.get("end_time") or 0) / 1000.0
                    wtxt = str(w.get("text") or "")
                    if wtxt:
                        words.append({"word": wtxt, "start": wst, "end": max(wst, wet)})
        return event_name

    chunk_bytes = 6400  # 200ms 16kHz 16-bit mono PCM

    if importlib.util.find_spec("websockets") is not None:
        import websockets  # type: ignore

        async with websockets.connect(
            ws_url,
            additional_headers={"Authorization": f"Bearer {api_key}"},
            max_size=16 * 1024 * 1024,
        ) as ws:
            await ws.send(json.dumps(run_task_msg, ensure_ascii=False))
            started = False
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue
                ev_name = _handle_server_event(raw)
                if ev_name == "task-started":
                    started = True
                    break
            if not started:
                raise RuntimeError("DashScope WebSocket 未收到 task-started 即关闭")

            async def _sender() -> None:
                for off in range(0, len(pcm_bytes), chunk_bytes):
                    await ws.send(pcm_bytes[off : off + chunk_bytes])
                    await asyncio.sleep(0.002)
                await ws.send(json.dumps(finish_task_msg, ensure_ascii=False))

            async def _receiver() -> None:
                async for raw in ws:
                    if isinstance(raw, bytes):
                        continue
                    if _handle_server_event(raw) == "task-finished":
                        break

            await asyncio.gather(_sender(), _receiver())
        return segments, words

    if importlib.util.find_spec("aiohttp") is not None:
        import aiohttp  # type: ignore

        timeout = aiohttp.ClientTimeout(total=600)
        async with aiohttp.ClientSession(trust_env=True, timeout=timeout) as session:
            async with session.ws_connect(
                ws_url,
                headers={"Authorization": f"Bearer {api_key}"},
                max_msg_size=16 * 1024 * 1024,
            ) as ws:
                await ws.send_str(json.dumps(run_task_msg, ensure_ascii=False))
                started = False
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        if _handle_server_event(msg.data) == "task-started":
                            started = True
                            break
                    elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                        raise RuntimeError(f"DashScope WebSocket 连接异常关闭: {msg.data}")
                if not started:
                    raise RuntimeError("DashScope WebSocket 未收到 task-started 即关闭")

                async def _sender() -> None:
                    for off in range(0, len(pcm_bytes), chunk_bytes):
                        await ws.send_bytes(pcm_bytes[off : off + chunk_bytes])
                        await asyncio.sleep(0.002)
                    await ws.send_str(json.dumps(finish_task_msg, ensure_ascii=False))

                async def _receiver() -> None:
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            if _handle_server_event(msg.data) == "task-finished":
                                break
                        elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                            break

                await asyncio.gather(_sender(), _receiver())
        return segments, words

    raise RuntimeError(
        "使用 dashscope 云端后端需要安装 aiohttp 或 websockets：\n"
        "请运行：pip install websockets （或 pip install aiohttp）"
    )


def transcribe_qwen3_asr(
    media_path: Path,
    tmp_dir: Path,
    model_size: str,
    device: str,
    qwen3_dir_override: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """调用本机已部署的 Qwen3_ASR 目录中的 batch_transcribe.py 执行离线转写与逐字对齐。"""
    qinfo = detect_qwen3_asr(dir_override=qwen3_dir_override)
    if not qinfo.get("available"):
        raise RuntimeError(
            f"未检测到可用的本机 Qwen3_ASR 环境（目录={qinfo.get('qwen3_asr_dir')}, "
            f"python={qinfo.get('python_executable')}, 可用模型={qinfo.get('available_models')}）。\n"
            "请先通过 init_machine.py 检查并配置 Qwen3_ASR 目录，或改用 faster-whisper / 云端 API。"
        )

    avail = list(qinfo.get("available_models") or [])
    eff_size = model_size if model_size in avail else avail[0]
    if eff_size != model_size:
        print(f"[提示] Qwen3_ASR 未检测到 '{model_size}' 权重，已自动切换至本机已就绪模型 '{eff_size}'。")

    qwen_py = str(qinfo["python_executable"])
    batch_script = str(qinfo["batch_script"])
    cmd = [
        qwen_py,
        batch_script,
        str(media_path),
        "-o",
        str(tmp_dir),
        "--save-timestamps",
        "--model-size",
        eff_size,
        "--sleep",
        "0",
        "--force",
    ]
    if device == "cpu":
        cmd.append("--cpu")

    env = os.environ.copy()
    ff_exe = ensure_ffmpeg_in_path()
    if ff_exe:
        ff_bin = str(Path(ff_exe).resolve().parent)
        if ff_bin not in env.get("PATH", "").split(os.pathsep):
            env["PATH"] = ff_bin + os.pathsep + env.get("PATH", "")

    print(f"[qwen3-asr/{eff_size}] 调用本机 Qwen3_ASR：{batch_script}")
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Qwen3_ASR 转写失败（退出码 {proc.returncode}）")

    from scan_talking_head import parse_srt

    raw_srt = tmp_dir / f"{media_path.stem}.srt"
    raw_json = tmp_dir / f"{media_path.stem}_逐字时间.json"
    if not raw_srt.is_file() or not raw_json.is_file():
        raise RuntimeError(f"Qwen3_ASR 未能产出 .srt 与 *_逐字时间.json：{tmp_dir}")

    srt_cues = parse_srt(raw_srt)
    char_tokens = json.loads(raw_json.read_text(encoding="utf-8-sig"))
    return srt_cues, char_tokens, {
        "effective_qwen3_dir": qinfo.get("qwen3_asr_dir"),
        "effective_model_size": eff_size,
        "qwen3_python": qwen_py,
    }


def choose_auto_backend(
    whisper_python_override: Optional[str] = None,
    allow_auto_setup: bool = False,
    qwen3_dir_override: Optional[str] = None,
) -> str:
    """按 已保存初始化档案偏好 -> 环境变量凭证 -> 已装 faster-whisper -> 本机 Qwen3_ASR -> --auto-setup 顺位选择后端。"""
    profile = load_machine_profile()
    saved_backend = str((profile.get("asr") or {}).get("default_backend") or "").strip().lower()
    if saved_backend and saved_backend != "auto":
        return "faster-whisper" if saved_backend == "whisper" else saved_backend

    if os.environ.get("DASHSCOPE_API_KEY"):
        return "dashscope"
    if (
        os.environ.get("TENCENTCLOUD_SECRET_ID")
        and os.environ.get("TENCENTCLOUD_SECRET_KEY")
        and os.environ.get("TENCENTCLOUD_APPID")
    ):
        return "tencent"
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if resolve_whisper_python(explicit=whisper_python_override) is not None:
        return "faster-whisper"
    if detect_qwen3_asr(dir_override=qwen3_dir_override).get("available"):
        return "qwen3-asr"
    if allow_auto_setup:
        return "faster-whisper"

    bootstrap_path = SCRIPT_DIR / "bootstrap_faster_whisper.py"
    init_path = SCRIPT_DIR / "init_machine.py"
    raise RuntimeError(
        "未检测到可用 ASR 云端密钥，本机也尚未安装 faster-whisper 或 Qwen3_ASR。\n"
        f'建议先运行电脑初始化向导：python "{init_path}" --interactive\n'
        "或直接任选其一配置：\n"
        "  方案 1（云端 API）：配置 DASHSCOPE_API_KEY / TENCENTCLOUD_* / GROQ_API_KEY / OPENAI_API_KEY\n"
        f'  方案 2（离线 faster-whisper）：python "{bootstrap_path}" --download-model small\n'
        "  方案 3（本机 Qwen3_ASR）：传入 --backend qwen3-asr --qwen3-dir <目录>"
    )


def resolve_media_input(args: argparse.Namespace) -> Path:
    if args.input:
        p = Path(args.input)
        if not p.is_file():
            raise FileNotFoundError(f"输入媒体文件不存在: {p}")
        return p
    if args.video:
        p = Path(args.video)
        if not p.is_file():
            raise FileNotFoundError(f"输入视频文件不存在: {p}")
        return p
    if args.dir:
        d = Path(args.dir)
        if not d.is_dir():
            raise FileNotFoundError(f"输入目录不存在: {d}")
        hits = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in MEDIA_EXTS]
        if not hits:
            raise FileNotFoundError(f"目录顶层未找到音视频文件（支持 {sorted(MEDIA_EXTS)}）: {d}")
        if len(hits) > 1:
            raise RuntimeError(f"目录内有多个媒体文件，请用 --input 指定明确文件: {[h.name for h in hits]}")
        return hits[0]
    raise ValueError("必须提供 --input / --video / --dir 之一")


def _shift_items(items: Iterable[Dict[str, Any]], offset: float) -> List[Dict[str, Any]]:
    shifted: List[Dict[str, Any]] = []
    for it in items:
        item = dict(it)
        if "start" in item:
            item["start"] = float(item["start"]) + offset
        if "end" in item:
            item["end"] = float(item["end"]) + offset
        shifted.append(item)
    return shifted


def run_transcription(args: argparse.Namespace) -> Dict[str, Any]:
    ensure_ffmpeg_in_path()
    media_path = resolve_media_input(args)
    out_dir = Path(args.out_dir) if args.out_dir else media_path.parent
    stem = args.stem or media_path.stem

    existing_srt = out_dir / f"{stem}.srt"
    existing_char = out_dir / f"{stem}_逐字时间.json"
    if (
        not args.force
        and args.backend == "auto"
        and existing_srt.is_file()
        and existing_char.is_file()
        and existing_srt.stat().st_size > 10
    ):
        try:
            chars = json.loads(existing_char.read_text(encoding="utf-8-sig"))
            if isinstance(chars, list) and chars:
                return {
                    "status": "reused_existing",
                    "media_path": str(media_path),
                    "outputs": {
                        "srt": str(existing_srt),
                        "char_json": str(existing_char),
                    },
                    "counts": {"char_tokens": len(chars)},
                }
        except Exception:
            pass

    profile = load_machine_profile()
    prof_asr = profile.get("asr") or {}
    backend = (
        choose_auto_backend(
            whisper_python_override=args.whisper_python,
            allow_auto_setup=bool(args.auto_setup),
            qwen3_dir_override=args.qwen3_dir,
        )
        if args.backend == "auto"
        else args.backend
    )
    if backend == "whisper":
        backend = "faster-whisper"

    duration_sec = probe_media_duration(media_path)
    prompt_text = args.prompt if args.prompt is not None else (DEFAULT_ZH_PROMPT if args.language.startswith("zh") else "")

    all_segments: List[Dict[str, Any]] = []
    all_words: List[Dict[str, Any]] = []
    extra_meta: Dict[str, Any] = {}
    chunks_used: List[Tuple[float, float]] = [(0.0, round(duration_sec, 3))]
    pre_normalized: Optional[Tuple[List[Dict[str, Any]], List[Dict[str, Any]], bool]] = None

    with tempfile.TemporaryDirectory(prefix="th_asr_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)

        if backend == "qwen3-asr":
            model_name = args.model or str(prof_asr.get("qwen3_model_size") or "1.7b")
            q_cues, q_chars, q_info = transcribe_qwen3_asr(
                media_path=media_path,
                tmp_dir=tmp_dir,
                model_size=model_name,
                device=args.device,
                qwen3_dir_override=args.qwen3_dir,
            )
            extra_meta.update(q_info)
            pre_normalized = (q_chars, q_cues, False)

        elif backend == "faster-whisper":
            model_name = args.model or str(prof_asr.get("faster_whisper_model") or "small")
            wav_path = tmp_dir / "full_16k_mono.wav"
            print(f"[1/2] 提取单声道 16kHz WAV 用于本地离线识别...")
            extract_audio_chunk(media_path, wav_path, codec_fmt="wav")
            print(f"[2/2] 执行 faster-whisper 本地转写...")
            all_segments, all_words, fw_info = _run_faster_whisper_with_dispatch(
                audio_path=wav_path,
                model_name=model_name,
                language=args.language,
                device=args.device,
                compute_type=args.compute_type,
                hf_endpoint=args.hf_endpoint,
                initial_prompt=prompt_text,
                tmp_dir=tmp_dir,
                whisper_python_override=args.whisper_python,
                auto_setup=bool(args.auto_setup),
                pip_index=args.pip_index,
            )
            extra_meta.update(fw_info)

        elif backend in {"groq", "openai"}:
            default_model = "whisper-large-v3-turbo" if backend == "groq" else "whisper-1"
            model_name = args.model or default_model
            if backend == "groq":
                api_key = os.environ.get("GROQ_API_KEY", "").strip()
                endpoint = args.base_url or "https://api.groq.com/openai/v1/audio/transcriptions"
            else:
                api_key = os.environ.get("OPENAI_API_KEY", "").strip()
                base = (args.base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
                endpoint = base if base.endswith("/audio/transcriptions") else f"{base}/audio/transcriptions"
            if not api_key:
                key_name = "GROQ_API_KEY" if backend == "groq" else "OPENAI_API_KEY"
                raise RuntimeError(f"使用 {backend} 后端需要先设置环境变量 {key_name}")

            max_chunk = float(args.chunk_sec or 600.0)
            mids = detect_silence_midpoints(media_path) if duration_sec > max_chunk else []
            chunks_used = plan_silence_chunks(duration_sec, max_chunk, mids)
            print(f"[{backend}] 媒体时长 {duration_sec:.1f}s -> 规划 {len(chunks_used)} 个静音边界分段...")
            for idx, (cs, ce) in enumerate(chunks_used, start=1):
                mp3_path = tmp_dir / f"chunk_{idx:03d}.mp3"
                extract_audio_chunk(media_path, mp3_path, codec_fmt="mp3", start_sec=cs, end_sec=ce)
                print(f"      上传分段 {idx}/{len(chunks_used)} ({cs:.2f}s ~ {ce:.2f}s, {mp3_path.stat().st_size // 1024} KB)...")
                segs, wds = transcribe_openai_compatible_chunk(
                    mp3_path, endpoint=endpoint, api_key=api_key, model_name=model_name, language=args.language, prompt=prompt_text
                )
                all_segments.extend(_shift_items(segs, cs))
                all_words.extend(_shift_items(wds, cs))

        elif backend == "dashscope":
            model_name = args.model or "paraformer-realtime-v2"
            api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
            if not api_key:
                raise RuntimeError("使用 dashscope 后端需要先设置环境变量 DASHSCOPE_API_KEY")

            max_chunk = float(args.chunk_sec or 600.0)
            mids = detect_silence_midpoints(media_path) if duration_sec > max_chunk else []
            chunks_used = plan_silence_chunks(duration_sec, max_chunk, mids)
            lang_hints = [args.language] if args.language else ["zh", "en"]
            print(f"[dashscope/{model_name}] 媒体时长 {duration_sec:.1f}s -> 规划 {len(chunks_used)} 个分段 WebSocket 流式识别...")
            for idx, (cs, ce) in enumerate(chunks_used, start=1):
                pcm_path = tmp_dir / f"chunk_{idx:03d}.pcm"
                extract_audio_chunk(media_path, pcm_path, codec_fmt="pcm", start_sec=cs, end_sec=ce)
                print(f"      WebSocket 推送分段 {idx}/{len(chunks_used)} ({cs:.2f}s ~ {ce:.2f}s)...")
                segs, wds = asyncio.run(
                    _dashscope_ws_transcribe_async(
                        pcm_path=pcm_path, api_key=api_key, model_name=model_name, language_hints=lang_hints
                    )
                )
                all_segments.extend(_shift_items(segs, cs))
                all_words.extend(_shift_items(wds, cs))

        elif backend == "tencent":
            model_name = args.model or "16k_zh"
            appid = os.environ.get("TENCENTCLOUD_APPID", "").strip()
            secret_id = os.environ.get("TENCENTCLOUD_SECRET_ID", "").strip()
            secret_key = os.environ.get("TENCENTCLOUD_SECRET_KEY", "").strip()
            if not (appid and secret_id and secret_key):
                raise RuntimeError("使用 tencent 后端需设置 TENCENTCLOUD_APPID、TENCENTCLOUD_SECRET_ID、TENCENTCLOUD_SECRET_KEY")

            max_chunk = float(args.chunk_sec or 300.0)
            mids = detect_silence_midpoints(media_path) if duration_sec > max_chunk else []
            chunks_used = plan_silence_chunks(duration_sec, max_chunk, mids)
            print(f"[tencent/{model_name}] 媒体时长 {duration_sec:.1f}s -> 规划 {len(chunks_used)} 个极速版识别分段...")
            for idx, (cs, ce) in enumerate(chunks_used, start=1):
                mp3_path = tmp_dir / f"chunk_{idx:03d}.mp3"
                extract_audio_chunk(media_path, mp3_path, codec_fmt="mp3", start_sec=cs, end_sec=ce)
                print(f"      上传分段 {idx}/{len(chunks_used)} ({cs:.2f}s ~ {ce:.2f}s, {mp3_path.stat().st_size // 1024} KB)...")
                segs, wds = transcribe_tencent_flash_chunk(
                    mp3_path, appid=appid, secret_id=secret_id, secret_key=secret_key, engine_type=model_name, voice_format="mp3"
                )
                all_segments.extend(_shift_items(segs, cs))
                all_words.extend(_shift_items(wds, cs))

        else:
            raise ValueError(f"不支持的识别后端: {backend}")

    if pre_normalized is not None:
        char_tokens, srt_cues, estimated_words = pre_normalized
    else:
        char_tokens, srt_cues, estimated_words = normalize_asr_result(all_segments, all_words)
    meta = {
        "backend": backend,
        "model": model_name,
        "media_path": str(media_path),
        "duration_sec": round(duration_sec, 3),
        "chunks": chunks_used,
        "estimated_word_timestamps": estimated_words,
        "counts": {
            "raw_segments": len(all_segments),
            "raw_words": len(all_words),
            "srt_cues": len(srt_cues),
            "char_tokens": len(char_tokens),
        },
        **extra_meta,
    }
    outputs = write_transcription_outputs(
        out_dir=out_dir,
        stem=stem,
        srt_cues=srt_cues,
        char_tokens=char_tokens,
        meta=meta,
        write_txt=bool(args.write_txt),
        write_meta=bool(args.write_meta),
    )
    return {
        "status": "transcribed",
        "backend": backend,
        "model": model_name,
        "estimated_word_timestamps": estimated_words,
        "outputs": outputs,
        "counts": meta["counts"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="口播音视频双引擎转写工具（云端 API / 本地 faster-whisper）")
    parser.add_argument("--input", "-i", help="输入视频或音频路径")
    parser.add_argument("--video", help="等同 --input")
    parser.add_argument("--dir", help="自动探测目录顶层唯一音视频文件")
    parser.add_argument("--out-dir", help="输出目录（默认同源媒体目录）")
    parser.add_argument("--stem", help="输出文件名前缀（默认同源媒体文件名）")
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "faster-whisper", "whisper", "qwen3-asr", "dashscope", "tencent", "groq", "openai"],
        help="转写后端：auto 优先复用已有逐字稿 -> 已保存初始化配置 -> 云端 Key -> faster-whisper -> Qwen3_ASR",
    )
    parser.add_argument("--model", help="模型名覆盖（如 small / medium / large-v3-turbo / 1.7b / 0.6b / paraformer-realtime-v2 / whisper-large-v3-turbo / 16k_zh）")
    parser.add_argument("--qwen3-dir", help="本机 Qwen3_ASR 目录覆盖（默认读取配置档案或自动探测）")
    parser.add_argument("--language", default="zh", help="主语言代码（默认 zh）")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="faster-whisper 推理设备")
    parser.add_argument("--compute-type", default="auto", help="faster-whisper 计算精度（auto / float16 / int8）")
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com", help="未设置环境变量 HF_ENDPOINT 时的默认 HuggingFace 镜像")
    parser.add_argument("--whisper-python", help="显式指定已安装 faster-whisper 的 Python 解释器路径（默认自动发现当前或隔离环境）")
    parser.add_argument("--auto-setup", action="store_true", help="缺少 faster-whisper 时自动通过国内镜像创建外部隔离环境安装后再执行")
    parser.add_argument("--pip-index", help="配合 --auto-setup 使用的 PyPI 镜像地址（默认清华源）")
    parser.add_argument("--base-url", help="openai / groq 后端自定义接口地址")
    parser.add_argument("--prompt", help="Whisper 系列引导提示词（zh 默认引导简体中文与标点）")
    parser.add_argument("--chunk-sec", type=float, help="云端分段最长秒数（默认 Groq/DashScope=600s, Tencent=300s）")
    parser.add_argument("--write-txt", action="store_true", help="额外导出纯文本 .txt（默认仅输出兼容度最高的 .srt 与 *_逐字时间.json 两个文件）")
    parser.add_argument("--write-meta", action="store_true", help="额外导出 *_transcribe_meta.json 调试信息")
    parser.add_argument("--force", action="store_true", help="强制重新识别并覆盖已有 .srt 与 *_逐字时间.json")
    parser.add_argument("--_fw-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_audio", help=argparse.SUPPRESS)
    parser.add_argument("--_raw-out", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._fw_worker:
        if not args._audio or not args._raw_out:
            raise ValueError("内部 worker 模式缺少 --_audio 或 --_raw-out")
        segs, wds, info = transcribe_faster_whisper(
            audio_path=Path(args._audio),
            model_name=args.model or "small",
            language=args.language,
            device=args.device,
            compute_type=args.compute_type,
            hf_endpoint=args.hf_endpoint,
            initial_prompt=args.prompt or "",
        )
        Path(args._raw_out).write_text(
            json.dumps({"segments": segs, "words": wds, "info": info}, ensure_ascii=False),
            encoding="utf-8",
        )
        return

    result = run_transcription(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

