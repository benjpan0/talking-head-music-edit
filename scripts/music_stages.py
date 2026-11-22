"""Conservative same-song stage candidates and a joint forward sequence search.

Stage names describe editing intentions, not verified musical phrases. Reference
durations are soft preferences unless an exact total is explicitly requested.
"""
from dataclasses import dataclass
from functools import lru_cache
import math
import re

import numpy as np

from music_engine import EditPlan, _cosine, plan_overlaps, render_plan, validate_request
from music_entry import entry_candidates


_NAMES = {'calm': '平静引入', 'rise': '起势', 'climax': '高潮',
          'sustain': '持续', 'release': '回落', 'ending': '收尾'}
_ALIASES = {
    'calm': 'calm', '平静': 'calm', '引入': 'calm', '平静引入': 'calm', '安静': 'calm',
    '开始': 'calm', '开场': 'calm', '铺垫': 'calm',
    'rise': 'rise', '起势': 'rise', '推进': 'rise', '起来': 'rise', '上升': 'rise',
    'climax': 'climax', '高潮': 'climax',
    'sustain': 'sustain', '持续': 'sustain', '维持': 'sustain',
    'release': 'release', '回落': 'release', '舒缓': 'release',
    'ending': 'ending', '收尾': 'ending', '结尾': 'ending', '结束': 'ending',
}
_TRANSITION_ALIASES = {'过渡', 'transition'}


def parse_stages(text):
    if not isinstance(text, str):
        raise ValueError('阶段格式应为 平静:6,起势:4,高潮:4,收尾:6')
    parts = re.split(r'[,，;；]', text.strip())
    if not 2 <= len(parts) <= 8:
        raise ValueError('多阶段需要2至8个阶段；候选数量不是阶段数量')
    raw_items = []
    for index, part in enumerate(parts, 1):
        pair = re.split(r'[:：]', part.strip())
        if len(pair) != 2:
            raise ValueError(f'阶段{index}格式错误，应为 阶段名:参考秒数')
        raw_name = pair[0].strip().lower()
        role = '__transition__' if raw_name in _TRANSITION_ALIASES else _ALIASES.get(raw_name)
        if role is None:
            raise ValueError(f'阶段{index}不支持的角色：{pair[0]}；支持开始、平静、起势、过渡、高潮、持续、回落、结尾')
        try:
            duration = float(pair[1])
        except ValueError as exc:
            raise ValueError(f'阶段{index}参考秒数无效') from exc
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f'阶段{index}参考秒数必须是有限正数')
        raw_items.append((role, duration))
    climax_indices = [idx for idx, (role, _) in enumerate(raw_items) if role == 'climax']
    result = []
    for idx, (role, duration) in enumerate(raw_items):
        if role == '__transition__':
            if any(c > idx for c in climax_indices) and not any(c < idx for c in climax_indices):
                role = 'rise'
            elif any(c < idx for c in climax_indices):
                role = 'release'
            else:
                role = 'rise' if idx < len(raw_items)/2 else 'release'
        result.append(dict(role=role, label=_NAMES[role], preferred_duration_s=duration))
    return result


@dataclass
class _Candidate:
    start: int
    end: int
    score: float
    evidence: dict
    warnings: list
    entry: object = None
    end_fade: int = 0


def plan_stages(analysis, features, stages, *, crossfade_ms=500, limit=2,
                entry_mode='auto', fade_in_ms=None, strategy='auto', exact_duration=None):
    """Find compatible forward excerpts, keeping natural nodes and soft timings.

    An exact total only filters real node combinations; no arbitrary first/last
    sample is moved to force a result. A missing plan is an explicit failure.
    """
    if not isinstance(stages, list) or not 2 <= len(stages) <= 8:
        raise ValueError('多阶段需要2至8个已解析阶段')
    for stage in stages:
        value = stage.get('preferred_duration_s')
        if (stage.get('role') not in _NAMES or isinstance(value, bool) or
                not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0):
            raise ValueError('阶段角色或参考时长无效')
    if isinstance(limit, bool) or not isinstance(limit, (int, np.integer)) or not 1 <= limit <= 3:
        raise ValueError('候选数量必须为1至3，不代表阶段数量')
    if strategy not in ('auto', 'opening', 'ending') or entry_mode not in ('auto', 'original', 'fade'):
        raise ValueError('阶段入口或出口策略无效')
    sr, n = int(features.sr), len(features.audio)
    reference = sum(s['preferred_duration_s'] for s in stages)
    if not math.isfinite(reference):
        raise ValueError('阶段参考总时长无效')
    validate_request(max(1.001, min(reference, features.duration)), sr, crossfade_ms, features.duration)
    exact_frames = None
    if exact_duration is not None:
        validate_request(exact_duration, sr, crossfade_ms, features.duration)
        exact_frames = round(exact_duration*sr)
    cf = round(crossfade_ms*sr/1000)
    if cf < 1:
        raise ValueError('交叉淡化不足一个采样帧')
    left_handle, right_handle = cf//2, cf-cf//2
    times = np.asarray(features.times, dtype=float)
    rms = np.asarray(features.rms_db, dtype=float)
    flux = np.asarray(features.flux, dtype=float)
    if (times.ndim != 1 or len(times) < 3 or rms.shape != times.shape or flux.shape != times.shape or
            not np.isfinite(times).all() or not np.isfinite(rms).all() or not np.isfinite(flux).all() or
            np.any(np.diff(times) <= 0)):
        raise ValueError('阶段分析需要有限、递增且长度一致的RMS/起音特征')
    beats = np.unique(np.rint(np.asarray(analysis.beats)*sr).astype(np.int64))
    beats = beats[(beats >= 0) & (beats < n)]
    downs = np.unique(np.rint(np.asarray(analysis.downbeats)*sr).astype(np.int64))
    downs = downs[(downs >= 0) & (downs < n)]
    if len(beats) < 2 or len(downs) < 2:
        raise ValueError('多阶段搜索需要实际拍点和小节标记，未虚构阶段边界')
    beat_s = float(np.median(np.diff(beats)))/sr
    # Prefer downbeats, supplied section edges, and detected silent stop.
    edges = np.rint(np.asarray(analysis.section_edges)*sr).astype(np.int64)
    active_threshold = max(-55., float(np.percentile(rms, 90))-35.)
    activity = rms > active_threshold
    if not activity.any():
        raise ValueError('原音频缺少足够活动，不能生成阶段发展')
    active_flux = np.where(activity, flux, 0.0)
    flux_scale = max(float(np.percentile(active_flux[activity], 75)), 1e-8)
    power = np.power(10.0, np.clip(rms, -120.0, 20.0)/10.0)
    win = max(1, min(200, len(times)//2 if len(times) < 20 else len(times)//10))
    kernel = np.ones(win)/win
    win_power_db = 10.0*np.log10(np.maximum(np.convolve(power, kernel, mode='valid'), 1e-10))
    win_flux = np.convolve(active_flux, kernel, mode='valid')
    win_act = np.convolve(activity.astype(float), kernel, mode='valid') > .2
    ordered_power = np.sort(win_power_db[win_act]) if win_act.any() else np.sort(win_power_db)
    ordered_flux = np.sort(win_flux[win_act]) if win_act.any() else np.sort(win_flux)
    tail_quiet_th = min(-48.0, active_threshold-2.0)
    loud_frames = np.flatnonzero(rms > tail_quiet_th)
    if len(loud_frames):
        silent_stop_s = min(features.duration, float(times[int(loud_frames[-1])])+.15)
        silent_stop_frame = min(n, max(1, round(silent_stop_s*sr)))
    else:
        silent_stop_frame = n
    nodes = np.unique(np.r_[0, downs, edges, silent_stop_frame, n])
    nodes = nodes[(nodes >= 0) & (nodes <= n)].astype(int)
    spec_arr = np.asarray(features.spectral, dtype=float)
    mean_spec = np.mean(spec_arr[activity], axis=0)
    cspec_arr = spec_arr - mean_spec
    prefixes = {}
    for name, values in [('rms', rms), ('power', power), ('flux', active_flux), ('active', activity.astype(float)),
                         ('chroma', np.asarray(features.chroma)), ('spectral', spec_arr), ('cspec', cspec_arr)]:
        prefixes[name] = np.concatenate((np.zeros((1,)+values.shape[1:]), np.cumsum(values, axis=0)), axis=0)

    def average(name, start_s, end_s):
        a = min(max(int(np.searchsorted(times, start_s)), 0), len(times)-1)
        b = min(max(int(np.searchsorted(times, end_s)), a+1), len(times))
        return (prefixes[name][b]-prefixes[name][a])/(b-a)

    def power_db(start_s, end_s):
        return 10.0*math.log10(max(float(average('power', start_s, end_s)), 1e-10))

    @lru_cache(maxsize=50000)
    def stats(start, end):
        a, b = start/sr, end/sr
        third = (b-a)/3
        levels = [10.0*math.log10(max(float(average('power', a+i*third, a+(i+1)*third)), 1e-10)) for i in range(3)]
        onsets = [float(average('flux', a+i*third, a+(i+1)*third))/flux_scale for i in range(3)]
        mean_power_db = 10.0*math.log10(max(float(average('power', a, b)), 1e-10))
        mean_flux = float(average('flux', a, b))
        rank_power = float(np.searchsorted(ordered_power, mean_power_db))/len(ordered_power)
        rank_flux = float(np.searchsorted(ordered_flux, mean_flux))/len(ordered_flux)
        rank = .65*rank_power+.35*rank_flux
        return dict(mean_level_dbfs=round(mean_power_db, 3), relative_level=round(rank, 4),
                    level_thirds_dbfs=[round(v,3) for v in levels],
                    onset_thirds_relative=[round(v,4) for v in onsets],
                    level_progression_db=round(levels[-1]-levels[0], 3),
                    onset_progression=round(onsets[-1]-onsets[0], 4),
                    active_fraction=round(float(average('active', a,b)),4),
                    status='acoustic_stage_candidate_not_phrase_or_cadence_verification')

    @lru_cache(maxsize=2048)
    def ending(end):
        t = end/sr
        last = float(average('rms', max(0,t-.18),t))
        onset = float(average('flux', max(0,t-.3),t))/flux_scale
        following = (times >= t) & (times <= t+.65)
        restart = bool(np.any(following & (rms > last+8) & (active_flux > flux_scale*.35)))
        future_act = bool(np.any((times >= t) & activity))
        file_end = n-end <= max(2,round(.025*sr))
        quiet_stop = (not future_act) and (last <= min(-38.0, active_threshold+2.0))
        audible_before = (times < t) & (rms > tail_quiet_th)
        last_audible = float(times[np.flatnonzero(audible_before)[-1]]) if audible_before.any() else None
        silence_tail_s = None if last_audible is None else max(0.0, t-last_audible)
        if exact_frames is None and silence_tail_s is not None and silence_tail_s > .60:
            return None
        measured = (times < t) & activity & (active_flux > flux_scale*.35)
        last_activity = float(times[np.flatnonzero(measured)[-1]]) if measured.any() else None
        ref_t = min(t, last_activity+.25) if last_activity is not None else t
        recent = (times >= max(0,ref_t-1.8)) & (times < ref_t)
        prior_peak = float(np.max(rms[recent])) if recent.any() else last
        if not (file_end or quiet_stop) and (last > -38 or prior_peak-last < 6 or onset > .55 or restart):
            return None
        kind = '原曲无声截停尾候选' if (quiet_stop and not file_end) else ('原曲文件尾候选' if file_end else '内部衰减收束候选')
        fade = 0 if file_end else (round(.05*sr) if quiet_stop else round(.15*sr))
        return dict(endpoint_kind=kind,
                    endpoint_level_dbfs=round(last,3), recent_decay_db=round(prior_peak-last,3),
                    endpoint_onset_relative=round(onset,4), last_activity_seconds=last_activity,
                    estimated_low_activity_tail_seconds=None if last_activity is None else round(t-last_activity,3),
                    cadence_status='not_verified'), fade

    entries = entry_candidates(analysis, features, min(stages[0]['preferred_duration_s'], features.duration),
                               mode=entry_mode, fade_in_ms=fade_in_ms)
    entries_by_start = {entry.frame: entry for entry in entries}
    down_indices = np.unique([int(np.argmin(np.abs(beats-d))) for d in downs])
    bar_size = int(round(np.median(np.diff(down_indices)))) if len(down_indices)>1 else 0

    @lru_cache(maxsize=2048)
    def phase(frame):
        i = int(np.argmin(np.abs(beats-frame)))
        if abs(int(beats[i])-frame) > max(.08*sr, beat_s*sr*.22) or not 2 <= bar_size <= 12:
            return None
        previous = down_indices[down_indices <= i]
        if not len(previous):
            return None
        return (i-int(previous[-1])) % bar_size

    @lru_cache(maxsize=50000)
    def join(a,b):
        if a == b:
            return 0., dict(kind='原曲连续演奏', crossfade_ms=0)
        if b-right_handle < a+left_handle or a+left_handle > n or b-right_handle < 0:
            return None
        pa,pb = phase(a),phase(b)
        if pa is None or pb is None or pa != pb:
            return None
        ta,tb = a/sr,b/sr
        chroma_cost = .5*_cosine(average('chroma',ta-.5,ta-.05),average('chroma',tb-.5,tb-.05))
        chroma_cost += .5*_cosine(average('chroma',ta+.05,ta+.5),average('chroma',tb+.05,tb+.5))
        spectral_cost = _cosine(average('spectral',ta-.25,ta+.25),average('spectral',tb-.25,tb+.25))
        win_cs = _cosine(average('cspec',ta-.35,ta+.35),average('cspec',tb-.35,tb+.35))
        cross_cs = _cosine(average('cspec',ta-1.0,ta-.05),average('cspec',tb+.05,tb+1.0))
        pre_cs = _cosine(average('cspec',ta-1.0,ta-.05),average('cspec',tb-1.0,tb-.05))
        timbre_cost = .45*cross_cs+.35*win_cs+.20*pre_cs
        offsets = np.linspace(-cf/(2*sr),cf/(2*sr),48)
        onset_cost = _cosine(np.interp(ta+offsets,times,active_flux),np.interp(tb+offsets,times,active_flux))
        level_difference = abs(power_db(ta-.8,ta-.05)-power_db(tb+.05,tb+.8))
        cost = .30*chroma_cost+.35*timbre_cost+.15*onset_cost+.20*min(level_difference/10.0,1.0)
        if (cost > .52 or chroma_cost > .95 or onset_cost > .92 or
                win_cs > .80 or cross_cs > .75 or (win_cs > .65 and pre_cs > .90) or level_difference > 6.5):
            return None
        return float(cost+.28), dict(kind='向前跳接候选', crossfade_ms=cf/sr*1000,
                                    same_supplied_beat_phase=True, phase=int(pa),
                                    chroma_cost=round(chroma_cost,5), spectral_cost=round(spectral_cost,5),
                                    timbre_cost=round(timbre_cost,5), cross_timbre_cost=round(cross_cs,5),
                                    overlap_onset_cost=round(onset_cost,5), level_difference_db=round(level_difference,3),
                                    status='compatibility_heuristic_not_musical_approval')

    libraries = []
    min_core = max(cf/sr+.05, min(1., beat_s*2))
    for number,stage in enumerate(stages):
        preferred,role = stage['preferred_duration_s'],stage['role']
        min_stage_len = min_core if exact_frames is not None else max(min_core, min(preferred*.45, preferred-1.5))
        starts = sorted(entries_by_start) if number == 0 else nodes[:-1]
        candidates = []
        for start in starts:
            if number == 0 and strategy == 'opening' and start > max(2,beat_s*3)*sr+int(beats[0]):
                continue
            for end in nodes[nodes > start]:
                length = (end-start)/sr
                if length < min_stage_len:
                    continue
                data = stats(int(start),int(end))
                if data['active_fraction'] < .08:
                    continue
                level,delta,fd = data['relative_level'],data['level_progression_db'],data['onset_progression']
                thirds,onset_thirds = data['level_thirds_dbfs'],data['onset_thirds_relative']
                up = delta >= 1.2 and thirds[1] >= thirds[0]-1.5 and thirds[2] >= thirds[1]-1.5
                onset_up = fd >= .15 and onset_thirds[1] >= onset_thirds[0]-.15 and onset_thirds[2] >= onset_thirds[1]-.15
                down = delta <= -1.2 and thirds[1] <= thirds[0]+1.5 and thirds[2] <= thirds[1]+1.5
                onset_down = fd <= -.15 and onset_thirds[1] <= onset_thirds[0]+.15 and onset_thirds[2] <= onset_thirds[1]+.15
                gentle_down = delta <= -.35 and thirds[1] <= thirds[0]+.2 and thirds[2] <= thirds[1]+.2
                endpoint = ending(int(end)) if number == len(stages)-1 or role == 'ending' else None
                end_fade,extra,warnings = 0,{},[]
                if role == 'calm':
                    if level > .68:
                        continue
                    role_cost = level*.65+max(0,delta-3)*.01
                elif role == 'rise':
                    if not (up or onset_up):
                        continue
                    role_cost = .25-min(max(delta/10,fd),1)*.15
                    extra['rise_basis'] = '持续电平推进' if up else '持续起音活动推进'
                elif role == 'climax':
                    if level < .55 or delta < -4.0:
                        continue
                    role_cost = (1-level)*1.15
                    extra['climax_basis'] = '小节尺度功率与起音活动加权高能量候选，非已确认的全曲或旋律高潮'
                elif role == 'release':
                    if not (down or onset_down or gentle_down):
                        continue
                    role_cost = (.25-min(max(-delta/10,-fd),1)*.15) if (down or onset_down) else (.22-delta*.05)
                    extra['release_basis'] = '持续电平或起音活动回落'
                elif role == 'ending':
                    if endpoint is None or not (down or onset_down or endpoint[0]['recent_decay_db'] >= 7):
                        continue
                    last = endpoint[0]['last_activity_seconds']
                    if last is None or start/sr > last-min(.75,preferred*.15):
                        continue
                    role_cost = .12 if endpoint[0]['endpoint_kind'] == '内部衰减收束候选' else .02
                    warnings.append('收尾仅有文件尾/无声截停或声学衰减依据，未确认和声终止式或完整乐句')
                else:
                    role_cost = abs(delta)*.015
                if number == len(stages)-1:
                    if endpoint is None or (strategy == 'ending' and endpoint[0]['endpoint_kind'] not in ('原曲文件尾候选', '原曲无声截停尾候选')):
                        continue
                    extra.update(endpoint[0]); end_fade = endpoint[1]
                if number == 0:
                    entry = entries_by_start[int(start)]
                    if entry.fade_frames >= end-start:
                        continue
                    entry_cost = entry.penalty+.65*start/max(n,1)
                    extra.update(entry.evidence)
                    warnings.extend(entry.warnings)
                else:
                    entry,entry_cost = None,0.
                duration_cost = .95*abs(math.log(length/preferred))
                evidence = dict(data, **extra, boundary_basis='supplied_downbeat_or_section_boundary_and_acoustic_entry',
                                reference_duration_s=preferred)
                candidates.append(_Candidate(int(start),int(end),float(role_cost+duration_cost+entry_cost),
                                             evidence,warnings,entry,end_fade))
        if not candidates:
            raise ValueError(f'阶段{number+1}“{stage["label"]}”没有符合当前声学条件的候选；未用音量标签或硬截代替')
        candidates.sort(key=lambda c:c.score)
        # Keep several options per source entrance so one low-cost region cannot
        # erase all globally feasible earlier/later combinations.
        counts,diverse = {},[]
        for candidate in candidates:
            bucket = candidate.start//max(1,round(2*sr))
            if counts.get(bucket,0) >= 8:
                continue
            counts[bucket] = counts.get(bucket,0)+1
            diverse.append(candidate)
            if len(diverse) >= 320:
                break
        libraries.append(diverse)

    # A bounded global path search evaluates role fit, soft duration and all
    # joins together; independent per-stage audio exports are never concatenated.
    beam = [(c.score,[c],[]) for c in libraries[0]]
    for number,library in enumerate(libraries[1:],1):
        role = stages[number]['role']
        prev_role = stages[number-1]['role']
        next_beam = []
        for cost,path,joins in beam:
            previous = path[-1]
            for candidate in library:
                if candidate.start < previous.end:
                    continue
                if role == 'release' and prev_role == 'climax':
                    if candidate.evidence['relative_level'] > previous.evidence['relative_level']+.02:
                        continue
                connection = join(previous.end,candidate.start)
                if connection is None:
                    continue
                connection_cost,connection_evidence = connection
                left_overlap = cf if joins and joins[-1]['kind'] != '原曲连续演奏' else 0
                right_overlap = cf if connection_evidence['kind'] != '原曲连续演奏' else 0
                # Actual middle length includes a half handle at each jumping
                # side; this check protects its unblended core and entrance.
                previous_actual = previous.end-previous.start+(right_handle if left_overlap else 0)+(left_handle if right_overlap else 0)
                if previous_actual < left_overlap+right_overlap:
                    continue
                if len(path)==1 and previous.entry.fade_frames > previous_actual-right_overlap:
                    continue
                next_beam.append((cost+candidate.score+connection_cost,path+[candidate],joins+[connection_evidence]))
        if not next_beam:
            raise ValueError(f'阶段{number}到{number+1}之间没有兼容且向前的接点，未退化为硬拼或独立段落拼接')
        next_beam.sort(key=lambda item:item[0])
        # Preserve diversity in the most recent source region as well as score.
        beam,counts = [],{}
        for item in next_beam:
            last=item[1][-1]
            bucket=last.start//max(1,round(2*sr))
            if counts.get(bucket,0)>=8:
                continue
            counts[bucket]=counts.get(bucket,0)+1
            beam.append(item)
            if len(beam)>=160:
                break

    plans = []
    for cost,path,connections in beam:
        total = sum(c.end-c.start for c in path)
        if exact_frames is not None and total != exact_frames:
            continue
        overlaps = tuple(cf if c['kind']!='原曲连续演奏' else 0 for c in connections)
        segments = tuple((c.start-(right_handle if i and overlaps[i-1] else 0),
                          c.end+(left_handle if i<len(overlaps) and overlaps[i] else 0))
                         for i,c in enumerate(path))
        if (segments[0][0]<0 or segments[-1][1]>n or
                any(segments[i+1][0]<segments[i][1] for i in range(len(segments)-1))):
            continue
        if path[-1].end_fade > segments[-1][1]-segments[-1][0]-(overlaps[-1] if overlaps else 0):
            continue
        stage_rows,offset = [],0
        warnings = ['多阶段为声学功能候选，尚未试听；未识别或保证完整旋律句、自然起势和和声终止式']
        for stage,candidate in zip(stages,path):
            length=candidate.end-candidate.start
            row = dict(role=stage['role'],label=stage['label'],preferred_duration_s=stage['preferred_duration_s'],
                       actual_duration_s=length/sr,deviation_s=length/sr-stage['preferred_duration_s'],
                       source_core_s=[candidate.start/sr,candidate.end/sr],
                       output_core_s=[offset/sr,(offset+length)/sr],
                       source_core_frames=[candidate.start,candidate.end],
                       output_core_frames=[offset,offset+length],evidence=candidate.evidence,
                       warnings=candidate.warnings)
            stage_rows.append(row); offset+=length
            warnings.extend(candidate.warnings)
        if abs(total/sr-reference)>.001:
            warnings.append(f'阶段时长按自然候选调整：参考合计{reference:g}秒，建议实际{total/sr:.3f}秒')
        evidence=dict(output_frames=int(total),reference_duration_s=reference,stages=stage_rows,
                      joins=connections,exact_total_requested=exact_duration is not None,
                      source_start_is_original=segments[0][0]==0,
                      source_duration_seconds=features.duration,
                      duration_status='exact_total' if exact_duration is not None else 'soft_stage_preferences',
                      search_method='bounded_global_forward_beam',search_beam_width=160,
                      search_limitations='bounded search; no plan does not prove the song has no possible edit')
        total_cost = float(cost)+(.65*abs(math.log(max(total/sr,.1)/reference)) if exact_frames is None else 0.)
        plans.append(EditPlan(segments,cf,path[-1].end_fade,total_cost,'多阶段联合重组',evidence,
                              list(dict.fromkeys(warnings)),start_fade_frames=path[0].entry.fade_frames,
                              join_crossfade_frames=overlaps))
    if not plans:
        reason = '与精确总时长一致的自然边界组合' if exact_duration is not None else '合法的多接点计划'
        raise ValueError(f'没有找到{reason}；未移动任意样本凑秒数、补静音或变速')
    plans.sort(key=lambda p:p.score)

    def _coalesced_spans(plan):
        spans = []
        for stage, overlap in zip(plan.evidence.get('stages', []), [0]+list(plan.join_crossfade_frames)):
            s0, s1 = stage['source_core_frames']
            if spans and not overlap and spans[-1][1] == s0:
                spans[-1] = (spans[-1][0], s1)
            else:
                spans.append((s0, s1))
        return spans

    selected=[]
    for plan in plans:
        if any(plan.segments==other.segments for other in selected):
            continue
        p_spans = _coalesced_spans(plan)
        if any(len(p_spans)==len(o_spans) and
               all(abs(ps[0]-os[0]) < .8*sr and abs(ps[1]-os[1]) < .8*sr
                   for ps, os in zip(p_spans, o_spans))
               for other in selected for o_spans in (_coalesced_spans(other),)):
            continue
        selected.append(plan)
        if len(selected)==limit:
            break
    if len(selected) < limit:
        for plan in plans:
            if any(plan.segments==other.segments for other in selected):
                continue
            p_stages = plan.evidence.get('stages', [])
            if any(len(p_stages)==len(o_stages) and
                   all(abs(ps['source_core_frames'][0]-os['source_core_frames'][0]) < .5*sr and
                       abs(ps['source_core_frames'][1]-os['source_core_frames'][1]) < .5*sr
                       for ps, os in zip(p_stages, o_stages))
                   for other in selected for o_stages in (other.evidence.get('stages', []),)):
                continue
            selected.append(plan)
            if len(selected)==limit:
                break
    if len(selected) < limit:
        for plan in plans:
            if any(plan.segments==other.segments for other in selected):
                continue
            selected.append(plan)
            if len(selected)==limit:
                break
    return selected
