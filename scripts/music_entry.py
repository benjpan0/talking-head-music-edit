"""Conservative acoustic entrances; no melodic phrase recognition is claimed."""
from dataclasses import dataclass
import math

import numpy as np


@dataclass
class EntryPoint:
    frame: int
    fade_frames: int
    penalty: float
    evidence: dict
    warnings: list


def entry_candidates(analysis, features, duration, *, mode='auto', fade_in_ms=None):
    """Return bounded source entrances using existing downbeats and cached RMS.

    A quiet interval followed by rising energy is acoustic evidence only. It
    does not establish a pickup, melody boundary, harmony, or complete phrase.
    Pre-roll belongs to the source interval and must count toward the target.
    """
    def finite_number(value):
        return (not isinstance(value, (bool, np.bool_)) and
                isinstance(value, (int, float, np.integer, np.floating)) and
                math.isfinite(value))

    sr_value = features.sr
    if not finite_number(sr_value) or sr_value <= 0 or int(sr_value) != sr_value:
        raise ValueError('入口分析需要有效的整数采样率')
    sr = int(sr_value)
    n = len(features.audio)
    if not finite_number(duration) or duration <= 0 or duration > n/sr+1e-9:
        raise ValueError('入口目标时长必须为有限正数且不长于音频')
    target = round(duration*sr)
    if target < 1 or mode not in ('auto', 'original', 'fade'):
        raise ValueError('入口目标样本数或模式无效')
    explicit_fade = None
    if fade_in_ms is not None:
        if not finite_number(fade_in_ms) or fade_in_ms < 0 or fade_in_ms/1000 >= duration:
            raise ValueError('开头淡入必须为有限非负数并短于目标时长')
        explicit_fade = round(fade_in_ms*sr/1000)
        if explicit_fade >= target:
            raise ValueError('开头淡入换算为样本后必须短于目标时长')

    times = np.asarray(features.times, dtype=float)
    rms = np.asarray(features.rms_db, dtype=float)
    if (times.ndim != 1 or not len(times) or rms.shape != times.shape or
            not np.isfinite(times).all() or not np.isfinite(rms).all() or
            times[0] < 0 or np.any(np.diff(times) <= 0)):
        raise ValueError('入口分析需要有限、递增且与RMS对应的特征时间')
    beats = np.asarray(analysis.beats, dtype=float)
    downs = np.asarray(analysis.downbeats, dtype=float)
    if (beats.ndim != 1 or downs.ndim != 1 or
            not np.isfinite(beats).all() or not np.isfinite(downs).all()):
        raise ValueError('入口拍点和小节标记必须是有限的一维时间')
    beats = np.unique(beats[(beats >= 0) & (beats < n/sr)])
    downs = np.unique(downs[(downs >= 0) & (downs < n/sr)])
    beat_duration = float(np.median(np.diff(beats))) if len(beats) > 1 else .5
    lookback = max(.04, min(1.2, 2*beat_duration))
    hop = float(np.median(np.diff(times))) if len(times) > 1 else .01
    anticlick = min(round(.01*sr), max(0, target-1))
    base_warnings = []
    if not len(downs):
        base_warnings.append('缺少可用小节起点，入口仅保留原曲0秒；未用普通拍点虚构起句')
    if len(beats) < 2:
        base_warnings.append('缺少可估计拍长的拍点，有限前导窗口采用默认值，不据此制造节拍')

    # Zero preserves whatever opening the recording actually contains.
    threshold = max(-60.0, float(np.percentile(rms, 90))-35)
    audible = np.flatnonzero(rms > threshold)
    zero_activity = float(times[audible[0]]) if len(audible) else None
    raw_points = [(0, 0, 0.0, '原曲起点', 'original_source_start',
                   zero_activity, {}, list(base_warnings))]

    if mode != 'original':
        anchors = downs[(downs > 0) & (downs <= (n-target)/sr+lookback)]
        # Keep early choices and spread remaining work across the whole song.
        if len(anchors) > 63:
            indices = np.unique(np.r_[np.arange(12),
                                     np.linspace(12, len(anchors)-1, 51).astype(int)])
            anchors = anchors[indices]
        for down in anchors:
            anchor = round(float(down)*sr)
            lo = max(0.0, anchor/sr-lookback)
            local = (times >= lo) & (times <= anchor/sr+.15)
            if not local.any():
                continue
            level = float(np.percentile(rms[local], 80))
            quiet_limit = min(-35.0, level-16)
            window = (times >= lo) & (times < anchor/sr)
            quiet = window & (rms <= quiet_limit)
            changes = np.diff(np.r_[False, quiet, False].astype(np.int8))
            quiet_runs = []
            for start, stop in zip(np.flatnonzero(changes == 1),
                                   np.flatnonzero(changes == -1)):
                length = times[stop-1]-times[start]+hop
                if length < .03-1e-9:
                    continue
                # The gap must be followed by an actual energy rise before
                # (or very close to) the supplied beat, not ongoing silence.
                after_gap = ((times >= times[stop-1]+hop*.5) &
                             (times <= anchor/sr+.08))
                strong_rise = after_gap & (rms >= quiet_limit+6)
                following = np.flatnonzero(after_gap & (rms > quiet_limit))
                if not strong_rise.any() or not len(following):
                    continue
                activity = float(times[following[0]])
                quiet_runs.append((float(length), -float(times[start]), activity))
            extra = dict(entry_quiet_threshold_dbfs=round(quiet_limit, 3),
                         entry_local_level_dbfs=round(level, 3))
            warnings = list(base_warnings)
            if quiet_runs:
                # A longer gap is stronger separation evidence than a tiny
                # dip between notes in the possible lead-in itself.
                gap_length, _, activity = max(quiet_runs)
                pre_roll = min(.03, gap_length/2)
                frame = max(0, math.floor((activity-pre_roll)*sr))
                default_fade = min(anticlick, max(0, round(activity*sr)-frame))
                kind, penalty = '保留声学前导的拍头候选', .06
                extra['entry_quiet_gap_seconds'] = round(gap_length, 6)
                warnings.append('静区及随后起音仅支持声学前导候选，未识别旋律弱起或完整乐句')
            else:
                lead = min(.35, beat_duration*.5, anchor/sr)
                frame = max(0, anchor-round(lead*sr))
                default_fade = anchor-frame
                activity = anchor/sr
                kind, penalty = '持续声部渐入候选', .28
                warnings.append('未找到清楚的前导静区，持续声部入口采用渐入；完整起句与更早引导音未核实')
            if frame > n-target or frame >= anchor:
                continue
            raw_points.append((frame, anchor, penalty, kind,
                               'acoustic_only_not_phrase_verified', activity,
                               dict(extra, entry_auto_fade_frames=int(default_fade)), warnings))

    result = []
    seen = set()
    for frame, anchor, penalty, kind, status, activity, extra, warnings in raw_points:
        if frame in seen:
            continue
        seen.add(frame)
        protected = max(0, round(activity*sr)-frame) if activity is not None else None
        automatic = int(extra.pop('entry_auto_fade_frames', anticlick))
        if explicit_fade is not None:
            fade = explicit_fade
            if protected is not None and fade > protected:
                warnings.append('显式开头淡入覆盖了估计起音或重音，可能削弱力度；这是用户参数而非自然起句证明')
        elif mode == 'fade':
            fade = min(round(.5*sr), max(0, target-1))
            if protected is not None:
                if protected == 0:
                    fade = min(fade, anticlick)
                    warnings.append('文件起点已存在声音，默认渐入缩短为防爆音长度以少削弱起音')
                else:
                    fade = min(fade, protected)
        else:
            fade = automatic
        # Default auto/fade entrances cannot extend beyond the requested cue.
        fade = min(int(fade), target-1)
        evidence = dict(entry_kind=kind,
                        entry_anchor_seconds=round(anchor/sr, 6),
                        entry_lead_seconds=round((anchor-frame)/sr, 6),
                        entry_fade_ms=round(fade*1000/sr, 6),
                        entry_boundary_status=status,
                        source_start_is_original=frame == 0,
                        entry_mode=mode,
                        entry_fade_overridden=explicit_fade is not None,
                        entry_first_activity_seconds=None if activity is None else round(activity, 6),
                        **extra)
        result.append(EntryPoint(int(frame), int(fade), float(penalty), evidence, warnings))
    return result
