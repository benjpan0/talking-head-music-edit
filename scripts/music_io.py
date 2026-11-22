"""Local analysis input and content-addressed audio feature caching.

Only NumPy, the standard library and a local FFmpeg executable are required.
The input report and source audio are never modified. Cache generations are
immutable so rebuilding a damaged cache is also safe with Windows memmaps open.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import uuid
import zipfile

import numpy as np


SAMPLE_RATE = 48000
N_FFT = 4096
HOP = 480
FFT_BATCH_FRAMES = 256
FEATURE_VERSION = "v01-48k-stereo-fft4096-hop480-3"


@dataclass
class Analysis:
    json_path: Path
    audio_path: Path
    title: str
    duration: float
    beats: np.ndarray
    downbeats: np.ndarray
    section_edges: np.ndarray
    sections: list[dict]
    warnings: list[str]
    cut_points: list[dict] = field(default_factory=list)


@dataclass
class Features:
    audio: np.ndarray
    sr: int
    times: np.ndarray
    chroma: np.ndarray
    spectral: np.ndarray
    rms_db: np.ndarray
    flux: np.ndarray
    duration: float
    cache_hit: bool
    audio_sha256: str
    cache_path: Path
    meta: dict


def _number(value, field):
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _times(values, scale, duration, field):
    if not isinstance(values, list):
        raise ValueError(f"{field} must be a JSON array")
    result = np.asarray([_number(value, field) * scale for value in values],
                        dtype=np.float64)
    # Reports can round the final boundary by a millisecond.
    if np.any(result < 0) or np.any(result > duration + 0.002):
        raise ValueError(f"{field} contains a time outside the audio duration")
    return np.unique(np.clip(result, 0.0, duration))


def _resolve_audio(report, json_path, audio_override):
    if audio_override is not None:
        override = Path(audio_override).expanduser().resolve()
        if not override.is_file():
            raise FileNotFoundError(f"audio override does not exist: {override}")
        return override

    candidates = []
    source = report.get("source", {})
    file_info = report.get("file", {})
    batch_source = report.get("batch_source", {})
    for declared in (source.get("audio_path"), file_info.get("path"),
                     file_info.get("audio_path"), batch_source.get("path")):
        if declared:
            declared_path = Path(str(declared)).expanduser()
            if declared_path.is_absolute():
                candidates.append(declared_path)
            else:
                candidates.extend((json_path.parent / declared_path,
                                   json_path.parent.parent / declared_path))
    if file_info.get("name"):
        name = Path(str(file_info["name"]))
        candidates.extend((json_path.parent.parent / name, json_path.parent / name))
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_file():
            return candidate
    checked = "; ".join(str(path) for path in candidates) or "no audio path in report"
    raise FileNotFoundError(
        f"Source audio is missing. Pass audio_override (CLI: --audio). Checked: {checked}")


def load_analysis(path, audio_override=None):
    """Read the millisecond editing report and normalize operative times to seconds.

    Original section metadata is retained; clip sections gain ``start`` and
    ``end`` in seconds in addition to their original ``*_ms`` values.
    """
    json_path = Path(path).expanduser().resolve()
    with json_path.open("r", encoding="utf-8-sig") as handle:
        report = json.load(handle)
    if (not isinstance(report, dict) or report.get("schema") != "tingdong_edit_report"
            or report.get("time_unit") != "ms"):
        raise ValueError("请提供_剪辑数据.json（tingdong_edit_report，毫秒）；不再接受原始秒制分析JSON")
    scale = 0.001
    duration = _number(report.get("duration_ms"), "duration_ms") * scale
    grid = report.get("beat_grid", {})
    beats = grid.get("beats_ms", [])
    downbeats = grid.get("downbeats_ms", [])
    start_field, end_field = "start_ms", "end_ms"
    if duration <= 0:
        raise ValueError("Audio duration must be positive")

    beat_array = _times(beats, scale, duration, "beats")
    downbeat_array = _times(downbeats, scale, duration, "downbeats")
    raw_sections = report.get("sections", [])
    if not isinstance(raw_sections, list):
        raise ValueError("sections must be an array")
    sections = []
    edges = [0.0, duration]
    for index, raw_section in enumerate(raw_sections):
        if not isinstance(raw_section, dict):
            raise ValueError(f"sections[{index}] must be an object")
        start = _number(raw_section.get(start_field), f"sections[{index}].start") * scale
        end = _number(raw_section.get(end_field), f"sections[{index}].end") * scale
        if start < 0 or end <= start or end > duration + 0.002:
            raise ValueError(f"sections[{index}] has invalid boundaries")
        section = dict(raw_section)
        section.update(start=min(start, duration), end=min(end, duration))
        sections.append(section)
        edges.extend((section["start"], section["end"]))

    raw_cuts = report.get("cut_points", [])
    cut_points = []
    if isinstance(raw_cuts, list):
        for item in raw_cuts:
            if isinstance(item, dict) and item.get("time_ms") is not None:
                try:
                    t = _number(item.get("time_ms"), "cut_points.time_ms") * scale
                except ValueError:
                    continue
                if 0.0 <= t <= duration + 0.002:
                    cp = dict(item)
                    cp["time_s"] = min(max(0.0, t), duration)
                    cut_points.append(cp)

    audio_path = _resolve_audio(report, json_path, audio_override)
    title = str(report.get("title") or report.get("file", {}).get("name")
                or audio_path.name)
    warnings = []
    if not len(beat_array):
        warnings.append("Analysis contains no beat grid; beat-aligned choices are unavailable.")
    if not len(downbeat_array):
        warnings.append("Analysis contains no downbeats; phrase alignment needs listening.")
    return Analysis(json_path, audio_path, title, duration, beat_array, downbeat_array,
                    np.unique(edges).astype(np.float64), sections, warnings, cut_points)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ffmpeg_binary(ffmpeg):
    if ffmpeg is not None:
        found = shutil.which(str(ffmpeg))
        if found:
            return found
        candidate = Path(ffmpeg).expanduser()
        if candidate.is_file():
            return str(candidate.resolve())
        raise FileNotFoundError(f"FFmpeg executable not found: {ffmpeg}")
    found = shutil.which("ffmpeg")
    if found:
        return found
    local = Path(r"D:\Program\ffmpeg\bin\ffmpeg.exe")
    if local.is_file():
        return str(local)
    raise FileNotFoundError("FFmpeg executable not found; pass ffmpeg=path")


def _close_memmap(array):
    mapping = getattr(array, "_mmap", None)
    if mapping is not None:
        mapping.close()


def _decode_audio(source, directory, ffmpeg):
    raw_path = directory / "decoded.f32le"
    command = [_ffmpeg_binary(ffmpeg), "-nostdin", "-v", "error", "-i", str(source),
               "-map", "0:a:0", "-vn", "-sn", "-dn", "-ac", "2", "-ar",
               str(SAMPLE_RATE), "-acodec", "pcm_f32le", "-f", "f32le", str(raw_path)]
    result = subprocess.run(command, stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                            check=False)
    if result.returncode:
        message = result.stderr.decode("utf-8", errors="replace")[-3000:]
        raise RuntimeError(f"FFmpeg audio decode failed: {message.strip()}")
    byte_count = raw_path.stat().st_size
    if byte_count == 0 or byte_count % 8:
        raise RuntimeError("FFmpeg audio decode produced empty or incomplete stereo frames")
    frames = byte_count // 8
    raw = np.memmap(raw_path, dtype="<f4", mode="r", shape=(frames, 2))
    output = np.lib.format.open_memmap(directory / "audio.npy", mode="w+",
                                       dtype=np.float32, shape=(frames, 2))
    try:
        for start in range(0, frames, 1024 * 1024):
            block = raw[start:start + 1024 * 1024]
            if not np.isfinite(block).all():
                raise RuntimeError("FFmpeg audio decode produced nonfinite samples")
            output[start:start + len(block)] = block
        output.flush()
    finally:
        _close_memmap(output)
        _close_memmap(raw)
    raw_path.unlink()
    return frames


def _unit_rows(values):
    denominator = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(denominator, 1e-20)


def _extract_features(audio_path, output_path):
    audio = np.load(audio_path, mmap_mode="r", allow_pickle=False)
    try:
        n_samples = len(audio)
        n_frames = (n_samples + HOP - 1) // HOP
        times = np.arange(n_frames, dtype=np.float64) * (HOP / SAMPLE_RATE)
        chroma = np.zeros((n_frames, 12), dtype=np.float32)
        spectral = np.zeros((n_frames, 24), dtype=np.float32)
        rms_db = np.empty(n_frames, dtype=np.float32)
        flux = np.zeros(n_frames, dtype=np.float32)
        window = np.hanning(N_FFT).astype(np.float32)
        frequencies = np.fft.rfftfreq(N_FFT, 1.0 / SAMPLE_RATE)
        band_edges = np.geomspace(40.0, SAMPLE_RATE / 2, 25)
        spectral_masks = [(frequencies >= left) & (frequencies < right)
                          for left, right in zip(band_edges[:-1], band_edges[1:])]
        spectral_masks[-1][-1] = True
        pitched = (frequencies >= 50.0) & (frequencies <= 5000.0)
        note_classes = np.zeros(len(frequencies), dtype=np.int64)
        note_classes[pitched] = np.rint(
            69.0 + 12 * np.log2(frequencies[pitched] / 440.0)).astype(np.int64) % 12
        chroma_masks = [pitched & (note_classes == note) for note in range(12)]
        previous = None

        for first in range(0, n_frames, FFT_BATCH_FRAMES):
            count = min(FFT_BATCH_FRAMES, n_frames - first)
            block_start = first * HOP - N_FFT // 2
            block_length = (count - 1) * HOP + N_FFT
            source_start = max(0, block_start)
            source_end = min(n_samples, block_start + block_length)
            block = np.zeros(block_length, dtype=np.float32)
            energy = np.zeros(block_length, dtype=np.float32)
            source = np.asarray(audio[source_start:source_end])
            offset = source_start - block_start
            block[offset:offset + len(source)] = source.mean(axis=1)
            energy[offset:offset + len(source)] = np.mean(source * source, axis=1)
            frames = np.lib.stride_tricks.sliding_window_view(block, N_FFT)[::HOP]
            energy_frames = np.lib.stride_tricks.sliding_window_view(energy, N_FFT)[::HOP]
            magnitude = np.abs(np.fft.rfft(frames * window, axis=1))
            destination = slice(first, first + count)
            rms = np.sqrt(np.mean(energy_frames, axis=1))
            rms_db[destination] = 20 * np.log10(np.maximum(rms, 1e-8))
            chroma_raw = np.column_stack(
                [magnitude[:, mask].sum(axis=1) for mask in chroma_masks])
            spectral_raw = np.column_stack(
                [magnitude[:, mask].mean(axis=1) for mask in spectral_masks])
            chroma[destination] = _unit_rows(chroma_raw)
            spectral[destination] = _unit_rows(np.log1p(spectral_raw))
            normalized = magnitude / np.maximum(magnitude.sum(axis=1, keepdims=True), 1e-20)
            if previous is not None:
                flux[first] = np.maximum(normalized[0] - previous, 0).sum()
            if count > 1:
                flux[first + 1:first + count] = np.maximum(
                    normalized[1:] - normalized[:-1], 0).sum(axis=1)
            previous = normalized[-1].copy()

        np.savez_compressed(output_path, times=times, chroma=chroma, spectral=spectral,
                            rms_db=rms_db, flux=flux)
    finally:
        _close_memmap(audio)


def _finite_audio(audio):
    return all(np.isfinite(audio[start:start + 1024 * 1024]).all()
               for start in range(0, len(audio), 1024 * 1024))


def _read_cache(key_path, audio_sha256):
    """Return validated numeric data or None. Invalid generations stay inspectable."""
    audio = None
    try:
        pointer = json.loads((key_path / "current.json").read_text(encoding="utf-8"))
        generation = pointer["generation"]
        if not isinstance(generation, str) or not re.fullmatch(r"generation-[0-9a-f]{32}", generation):
            return None
        directory = key_path / generation
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if (metadata["audio_sha256"] != audio_sha256
                or metadata["feature_version"] != FEATURE_VERSION
                or metadata["sample_rate"] != SAMPLE_RATE
                or metadata["n_fft"] != N_FFT or metadata["hop"] != HOP):
            return None
        for filename in ("audio.npy", "features.npz"):
            if _sha256(directory / filename) != metadata["file_sha256"][filename]:
                return None
        audio = np.load(directory / "audio.npy", mmap_mode="r", allow_pickle=False)
        if (audio.dtype != np.float32 or audio.ndim != 2 or audio.shape[1] != 2
                or len(audio) <= 0 or len(audio) != metadata["num_samples"]
                or not _finite_audio(audio)):
            raise ValueError("Invalid cached audio")
        count = (len(audio) + HOP - 1) // HOP
        with np.load(directory / "features.npz", allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in ("times", "chroma", "spectral", "rms_db", "flux")}
        shapes = {"times": (count,), "chroma": (count, 12), "spectral": (count, 24),
                  "rms_db": (count,), "flux": (count,)}
        for name, values in arrays.items():
            if (values.shape != shapes[name] or values.dtype.kind != "f"
                    or not np.isfinite(values).all()):
                raise ValueError(f"Invalid cached {name}")
        if not np.allclose(arrays["times"], np.arange(count) * HOP / SAMPLE_RATE,
                           rtol=0, atol=1e-8):
            raise ValueError("Invalid cached time grid")
        return audio, arrays, metadata, directory
    except (OSError, ValueError, TypeError, KeyError, EOFError, zipfile.BadZipFile):
        if audio is not None:
            _close_memmap(audio)
        return None


def _write_json_atomic(path, value):
    temporary = path.with_name(f".{path.name}-{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _analysis_snapshot(analysis, directory):
    raw = analysis.json_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    snapshots = directory / "analysis_snapshots"
    snapshots.mkdir(exist_ok=True)
    destination = snapshots / f"{digest}.json"
    # A damaged snapshot is replaced; report/audio inputs remain read-only.
    if not destination.is_file() or _sha256(destination) != digest:
        temporary = snapshots / f".{digest}-{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("xb") as handle:
                handle.write(raw)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return digest, destination


def load_features(analysis, cache_dir, ffmpeg=None):
    """Decode once and reuse validated acoustic features for any target duration.

    ``cache_path`` is the active immutable generation directory. Both source
    content and version determine its cache key. Hit checks hash cached files;
    a hit never requires FFmpeg. Chroma and log 24-band spectral rows have unit
    L2 norm, with zero vectors for silence. Times mark centered window positions.
    """
    source_sha = _sha256(analysis.audio_path)
    cache_root = Path(cache_dir).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    key_path = cache_root / f"{source_sha}-{FEATURE_VERSION}"
    key_path.mkdir(exist_ok=True)
    cached = _read_cache(key_path, source_sha)
    cache_hit = cached is not None
    if cached is None:
        generation = f"generation-{uuid.uuid4().hex}"
        destination = key_path / generation
        with tempfile.TemporaryDirectory(prefix=".building-", dir=key_path) as work:
            staged = Path(work)
            num_samples = _decode_audio(analysis.audio_path, staged, ffmpeg)
            _extract_features(staged / "audio.npy", staged / "features.npz")
            if _sha256(analysis.audio_path) != source_sha:
                raise RuntimeError("Source audio changed during decode; retry with a stable file")
            metadata = {
                "feature_version": FEATURE_VERSION, "audio_sha256": source_sha,
                "source_audio": str(analysis.audio_path), "sample_rate": SAMPLE_RATE,
                "n_fft": N_FFT, "hop": HOP, "num_samples": num_samples,
                "duration": num_samples / SAMPLE_RATE, "channels": 2,
                "window_centered": True, "fft_batch_frames": FFT_BATCH_FRAMES,
                "chroma_normalization": "L2 per frame, C=0",
                "spectral_normalization": "log1p magnitude in 24 logarithmic bands, L2 per frame",
                "spectral_band_edges_hz": np.geomspace(40.0, SAMPLE_RATE / 2, 25).tolist(),
                "flux": "positive L1-normalized magnitude differences; first frame zero",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "file_sha256": {name: _sha256(staged / name)
                                for name in ("audio.npy", "features.npz")},
            }
            _write_json_atomic(staged / "metadata.json", metadata)
            # No existing generation or source file can be overwritten here.
            staged.rename(destination)
        _write_json_atomic(key_path / "current.json", {"generation": generation})
        cached = _read_cache(key_path, source_sha)
        if cached is None:
            raise RuntimeError("Newly built audio feature cache did not validate")

    audio, arrays, metadata, directory = cached
    try:
        analysis_sha, snapshot = _analysis_snapshot(analysis, directory)
    except BaseException:
        _close_memmap(audio)
        raise
    metadata = dict(metadata, analysis_json=str(analysis.json_path),
                    analysis_sha256=analysis_sha, analysis_snapshot=str(snapshot))
    actual_duration = len(audio) / SAMPLE_RATE
    if abs(actual_duration - analysis.duration) > 0.05:
        metadata["duration_warning"] = (
            f"Report duration {analysis.duration:.6f}s differs from decoded audio "
            f"{actual_duration:.6f}s; decoded samples are authoritative.")
    return Features(audio, SAMPLE_RATE, arrays["times"], arrays["chroma"],
                    arrays["spectral"], arrays["rms_db"], arrays["flux"],
                    actual_duration, cache_hit, source_sha, directory, metadata)
