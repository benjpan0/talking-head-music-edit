"""Bounded candidate planning; exact timing is not musical approval.

The legacy planner uses at most one forward splice; the renderer also accepts
multi-stage plans with a separate overlap length at every forward join.
Neither path manufactures silence or stretches audio to force a duration.
"""
from dataclasses import dataclass
from functools import lru_cache
import math
import numpy as np
from music_entry import entry_candidates


@dataclass
class EditPlan:
    segments: tuple
    crossfade_frames: int
    end_fade_frames: int
    score: float
    strategy: str
    evidence: dict
    warnings: list
    start_fade_frames: int = 0
    join_crossfade_frames: tuple = ()


def validate_request(duration, sample_rate, crossfade_ms, audio_duration):
    for name, value in [('duration', duration), ('sample_rate', sample_rate),
                        ('crossfade_ms', crossfade_ms), ('audio_duration', audio_duration)]:
        if not isinstance(value, (int, float, np.integer, np.floating)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name}必须是有限的正数')
    if duration < 1:
        raise ValueError('当前版本目标时长至少1秒')
    if duration > audio_duration+1e-9:
        raise ValueError('目标长于原音频；当前版本不自动延长或循环')
    if crossfade_ms/1000 >= duration/2:
        raise ValueError('交叉淡化必须短于目标时长的一半')


def plan_overlaps(plan):
    count = len(plan.segments)-1
    if count < 0:
        raise ValueError('计划至少需要一个原音频片段')
    explicit = tuple(plan.join_crossfade_frames)
    overlaps = explicit if explicit else (plan.crossfade_frames,)*count
    if len(overlaps) != count or any(not isinstance(n, (int, np.integer)) or n < 0 for n in overlaps):
        raise ValueError('每个接点必须有有效的交叉淡化样本数')
    return tuple(int(n) for n in overlaps)


def plan_timeline(plan):
    overlaps = plan_overlaps(plan)
    placements, offset = [], 0
    for index, (start, end) in enumerate(plan.segments):
        if index:
            offset -= overlaps[index-1]
        placements.append((int(offset), int(offset+end-start)))
        offset += end-start
    return tuple(placements)


def render_plan(audio, plan, target_frames, gain_db=0.0):
    if np.ndim(audio) != 2 or audio.shape[1] != 2 or not np.isfinite(gain_db):
        raise ValueError('需要有限增益与双声道音频')
    if not isinstance(target_frames, (int, np.integer)) or target_frames <= 0:
        raise ValueError('目标样本数无效')
    if not plan.segments:
        raise ValueError('计划至少需要一个原音频片段')
    cf = plan.crossfade_frames
    if not isinstance(cf, (int, np.integer)) or cf < 0:
        raise ValueError('交叉淡化样本数无效')
    chunks = []
    for start, stop in plan.segments:
        if not all(isinstance(x, (int, np.integer)) for x in (start, stop)) or not 0 <= start < stop <= len(audio):
            raise ValueError('原音频区间越界或顺序错误')
        chunks.append(audio[start:stop])
    overlaps = plan_overlaps(plan)
    for index, overlap in enumerate(overlaps):
        previous, following = plan.segments[index], plan.segments[index+1]
        if following[0] < previous[1]:
            raise ValueError('仅支持向前且源区间不重叠的计划')
        if overlap == 0 and following[0] != previous[1]:
            raise ValueError('零交叉淡化仅用于原曲连续演奏，不能硬拼跳接')
        if overlap > min(len(chunks[index]), len(chunks[index+1])):
            raise ValueError('交叉淡化长于可用片段')
    for index, chunk in enumerate(chunks):
        left = overlaps[index-1] if index else 0
        right = overlaps[index] if index < len(overlaps) else 0
        if left+right > len(chunk):
            raise ValueError('同一片段两端的交叉淡化窗口重叠')
    effective = sum(map(len, chunks))-sum(overlaps)
    if effective != target_frames:
        raise ValueError(f'计划长度{effective}不等于目标{target_frames}；禁止补零或截尾')
    if not isinstance(plan.end_fade_frames, (int, np.integer)) or not 0 <= plan.end_fade_frames <= target_frames:
        raise ValueError('尾部淡出区间无效')
    first_unblended = len(chunks[0])-(overlaps[0] if overlaps else 0)
    if (not isinstance(plan.start_fade_frames, (int, np.integer)) or
            not 0 <= plan.start_fade_frames <= min(first_unblended, target_frames-plan.end_fade_frames)):
        raise ValueError('开头淡入区间无效，或与内部交叉淡化/尾淡出重叠')
    last_unblended = len(chunks[-1])-(overlaps[-1] if overlaps else 0)
    if plan.end_fade_frames > last_unblended:
        raise ValueError('末尾淡出与内部交叉淡化重叠')
    parts = []
    for index, chunk in enumerate(chunks):
        left = overlaps[index-1] if index else 0
        right = overlaps[index] if index < len(overlaps) else 0
        if left:
            weights = np.linspace(0, 1, left, dtype=np.result_type(audio.dtype, np.float32))[:, None]
            parts.append(chunks[index-1][-left:]*(1-weights)+chunk[:left]*weights)
        stop = len(chunk)-right if right else len(chunk)
        if stop > left:
            parts.append(chunk[left:stop])
    out = np.concatenate(parts).copy()
    if gain_db:
        out = out*10**(float(gain_db)/20)
    if plan.start_fade_frames:
        n = plan.start_fade_frames
        out[:n] *= np.linspace(0, 1, n)[:, None]
    if plan.end_fade_frames:
        n = plan.end_fade_frames
        out[-n:] *= np.linspace(1, 0, n)[:, None]
    if not np.isfinite(out).all():
        raise ValueError('输出包含无效数值')
    return out


def _cosine(a, b):
    den = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.clip(1-np.dot(a, b)/den, 0, 2)) if den > 1e-12 else 1.0


def plan_candidates(analysis, features, duration, *, crossfade_ms=500, limit=3, strategy='auto',
                    entry_mode='auto', fade_in_ms=None):
    validate_request(duration, features.sr, crossfade_ms, features.duration)
    if (strategy not in ('auto', 'opening', 'ending') or
            isinstance(limit, (bool, np.bool_)) or not isinstance(limit, (int, np.integer)) or not 1 <= limit <= 3):
        raise ValueError('策略或候选数量无效')
    sr, n = features.sr, len(features.audio)
    target = round(duration*sr)
    cf = round(crossfade_ms*sr/1000)
    if cf < 1:
        raise ValueError('交叉淡化不足一个采样帧')
    times = np.asarray(features.times)
    rms = np.asarray(features.rms_db)
    flux = np.asarray(features.flux)
    active = (rms > -50) & np.isfinite(rms)
    flux_scale = max(float(np.percentile(flux[active], 75)) if active.any() else 0, 1e-8)
    beats = np.unique(np.rint(np.asarray(analysis.beats)*sr).astype(int))
    beats = beats[(beats >= 0) & (beats < n)]
    downs = np.unique(np.rint(np.asarray(analysis.downbeats)*sr).astype(int))
    downs = downs[(downs >= 0) & (downs < n)]
    edges = np.asarray(analysis.section_edges, dtype=float)
    beat_duration = float(np.median(np.diff(beats))/sr) if len(beats) > 1 else .5
    if not len(beats):
        raise ValueError('没有可用拍点；当前工具不凭空生成节拍')

    def profile(arr, t, lo, hi):
        i = int(np.searchsorted(times, t+lo))
        j = int(np.searchsorted(times, t+hi))
        i = min(max(i, 0), len(times)-1); j = min(max(j, i+1), len(times))
        return np.mean(arr[i:j], axis=0)

    # Phase labels are measured from existing downbeat markers, not fixed BPM.
    down_beat_indices = np.unique([int(np.argmin(np.abs(beats-d))) for d in downs]) if len(downs) else np.array([], dtype=int)
    bar_beats = int(round(np.median(np.diff(down_beat_indices)))) if len(down_beat_indices) > 1 else 0
    def phase(frame):
        idx = int(np.argmin(np.abs(beats-frame)))
        if not len(down_beat_indices) or not 2 <= bar_beats <= 12:
            return None
        preceding = down_beat_indices[down_beat_indices <= idx]
        if not len(preceding):
            return None
        previous = int(preceding[-1])
        following = down_beat_indices[down_beat_indices > previous]
        if len(following) and int(following[0])-previous != bar_beats:
            return None
        return (idx-previous) % bar_beats

    @lru_cache(maxsize=20000)
    def pair_cost(a, b):
        ta, tb = a/sr, b/sr
        cc = .6*_cosine(profile(features.chroma, ta, -.6, -.08), profile(features.chroma, tb, -.6, -.08))
        cc += .4*_cosine(profile(features.chroma, ta, .08, .55), profile(features.chroma, tb, .08, .55))
        sc = _cosine(profile(features.spectral, ta, -.2, .2), profile(features.spectral, tb, -.2, .2))
        level = abs(float(profile(rms, ta, -.15, .15))-float(profile(rms, tb, -.15, .15)))
        off = np.arange(-min(.3, beat_duration/2), min(.3, beat_duration/2)+.001, .02)
        oc = _cosine(np.interp(ta+off, times, flux), np.interp(tb+off, times, flux))
        # Repetition evidence covers a sequence, not the crossfade window alone.
        context = min(2.5, max(.8, beat_duration*max(bar_beats, 2)))
        grid = np.linspace(-context, -.05, 24)
        ca = np.column_stack([np.interp(ta+grid, times, features.chroma[:, k]) for k in range(12)]).ravel()
        cb = np.column_stack([np.interp(tb+grid, times, features.chroma[:, k]) for k in range(12)]).ravel()
        sequence_cost = _cosine(ca, cb)
        # Check the full requested overlap, including the new one-second setting.
        overlap_grid = np.linspace(-cf/(2*sr), cf/(2*sr), 64)
        overlap_onset_cost = _cosine(np.interp(ta+overlap_grid, times, flux),
                                     np.interp(tb+overlap_grid, times, flux))
        cost = .35*cc+.15*sc+.15*min(level/12, 1)+.10*oc+.15*sequence_cost+.10*overlap_onset_cost
        return cost, dict(chroma_cost=round(cc, 5), spectral_cost=round(sc, 5),
                          level_difference_db=round(level, 3), onset_cost=round(oc, 5),
                          repeated_context_seconds=round(context, 3), repeated_context_cost=round(sequence_cost, 5),
                          full_overlap_onset_cost=round(overlap_onset_cost, 5),
                          repetition_status='acoustic_similarity_only')

    @lru_cache(maxsize=30000)
    def endpoint(frame):
        t = frame/sr
        last = float(profile(rms, t, -.12, -.015))
        current_flux = float(profile(flux, t, -.25, 0))
        prev = float(np.max(rms[(times >= max(0, t-1.5)) & (times < t)])) if np.any((times >= max(0, t-1.5)) & (times < t)) else last
        near_file_end = n-frame <= max(.025*sr, 2)
        future = rms[(times >= t) & (times <= features.duration)]
        quiet_rest = last < -43 and (not len(future) or float(np.max(future)) < -35)
        if near_file_end or quiet_rest:
            kind, penalty, fade = '原曲尾部', 0.0, 0
        else:
            edge_distance = float(np.min(np.abs(edges[1:-1]-t))) if len(edges) > 2 else float('inf')
            decayed = prev-last >= 7 and current_flux < flux_scale*.55
            if not decayed:
                return None
            # This is explicitly a trial stop, never an asserted cadence.
            kind, penalty, fade = '内部短收束候选', .30, round(min(.35, beat_duration*.6, duration*.1)*sr)
        recent = (times >= max(0, t-min(15, duration))) & (times < t) & (rms > -40) & (flux > flux_scale*.7)
        last_activity = float(times[np.flatnonzero(recent)[-1]]) if recent.any() else max(0, t-duration)
        tail_estimate = max(0, t-last_activity)
        penalty += max(0, tail_estimate/duration-.22)*1.3
        return penalty, fade, dict(endpoint_kind=kind, endpoint_level_dbfs=round(last, 2),
                                   estimated_low_activity_tail_seconds=round(tail_estimate, 3),
                                   tail_estimate_status='heuristic_not_last_note_detection',
                                   endpoint_nearest_beat_offset_seconds=round(float(np.min(np.abs(beats-frame)))/sr, 4))

    first = int(beats[0])
    entries = entry_candidates(analysis, features, duration, mode=entry_mode, fade_in_ms=fade_in_ms)
    min_side = max(cf, round(.75*sr))
    plans = []

    def add(entry, a=None, b=None):
        s = entry.frame
        removed = 0 if a is None else b-a
        e = int(s+target+removed)
        if e > n or s < 0 or e <= s:
            return
        ending = endpoint(e)
        if ending is None:
            return
        end_penalty, end_fade, end_evidence = ending
        opening = s <= first+max(2, beat_duration*3)*sr
        if strategy == 'opening' and not opening:
            return
        if strategy == 'ending' and end_evidence['endpoint_kind'] != '原曲尾部':
            return
        warnings = ['尚未试听，声学评分不代表乐句或和声已经自然', *entry.warnings]
        if end_evidence['endpoint_kind'] != '原曲尾部':
            warnings.append('使用原曲内部短收束及显式淡出，未确认终止式')
        if end_evidence['estimated_low_activity_tail_seconds'] > max(2, duration*.25):
            warnings.append(f"尾部低活动区估计约{end_evidence['estimated_low_activity_tail_seconds']:.1f}秒，可能偏长；不是已识别的纯尾音")
        if a is None:
            segments = ((int(s), e),)
            label = '连续取段'
            cost, pair_evidence = .15, {}
        else:
            pa, pb = phase(a), phase(b)
            if pa is None or pb is None or pa != pb:
                return
            cost, pair_evidence = pair_cost(int(a), int(b))
            if cost > .35 or pair_evidence['full_overlap_onset_cost'] > .8 or (opening and pair_evidence['repeated_context_cost'] > .30):
                return
            left, right = cf//2, cf-cf//2
            segments = ((int(s), int(a+left)), (int(b-right), e))
            if min(stop-start for start, stop in segments) < min_side:
                return
            if b-right < a+left:
                return
            if end_evidence['endpoint_kind'] == '原曲尾部':
                label = '保留首尾删减' if s == 0 else ('前部取材保留结尾' if opening else '保留结尾删减')
            else:
                label = '前部相似段删减' if opening else '内部相似段删减'
            pair_evidence.update(anchor_a_seconds=round(a/sr, 6), anchor_b_seconds=round(b/sr, 6),
                                 same_beat_phase=True, phase=pa,
                                 phase_status='agreement_with_supplied_markers_not_independent_musical_verification',
                                 jump_seconds=round((b-a)/sr, 6))
        first_unblended = segments[0][1]-segments[0][0]-(cf if a is not None else 0)
        if entry.fade_frames > min(first_unblended, target-end_fade):
            return
        body = target/sr-end_evidence['estimated_low_activity_tail_seconds']
        score = cost+end_penalty+entry.penalty+(.08 if not opening else 0)+(.18 if body < min(4, duration*.35) else 0)
        evidence = dict(source_start_seconds=round(s/sr, 6), source_end_seconds=round(e/sr, 6),
                        source_duration_seconds=features.duration, target_seconds=target/sr,
                        **pair_evidence, **end_evidence, **entry.evidence)
        plans.append(EditPlan(segments, cf, end_fade, float(score), label, evidence, warnings,
                              start_fade_frames=entry.fade_frames))

    for entry in entries:
        s = entry.frame
        add(entry)
        eligible_a = beats[(beats >= s+min_side-cf//2) & (beats <= s+target-min_side+cf//2)]
        for a in eligible_a:
            possible_b = beats[(beats > a+max(cf, beat_duration*sr)) & (beats <= n-min_side+cf//2)]
            # The derived endpoint must be a measured low-activity or structural candidate.
            for b in possible_b:
                e = int(s+target+b-a)
                if e > n:
                    break
                add(entry, int(a), int(b))
    if not plans:
        raise ValueError('没有找到同时满足目标时长、入口、节拍相位和出口筛选的计划；未输出硬截或补静音版本')
    plans.sort(key=lambda p: p.score)
    chosen = []
    for p in plans:
        if any(p.segments == q.segments for q in chosen):
            continue
        if any(p.strategy == q.strategy and abs(p.segments[0][0]-q.segments[0][0]) < sr*2 and
               abs(p.segments[-1][1]-q.segments[-1][1]) < sr*2 for q in chosen):
            continue
        chosen.append(p)
        if len(chosen) == limit:
            break
    return chosen
