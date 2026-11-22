"""Create an editing plan or finished audio, using disposable per-run cache."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import numpy as np

from music_io import load_analysis, load_features, _close_memmap
from music_engine import plan_candidates, render_plan, validate_request, plan_overlaps, plan_timeline
from music_stages import parse_stages, plan_stages


def build_parser():
    p = argparse.ArgumentParser(description='输出剪辑方案或成品音乐，结果保存到原音频目录')
    p.add_argument('analysis_json')
    p.add_argument('--mode', choices=('plan', 'audio'), required=True)
    p.add_argument('--duration', type=float, help='普通模式必填的精确秒数；多阶段中给出时锁定总长')
    p.add_argument('--stages', help='阶段及参考秒数，如 平静:6,起势:4,高潮:4,收尾:6')
    p.add_argument('--output', help='原音频目录的独立结果子目录；默认自动命名')
    p.add_argument('--audio')
    p.add_argument('--cache', default=str(Path(tempfile.gettempdir())/'talking-head-music-edit'))
    p.add_argument('--strategy', choices=('auto', 'opening', 'ending'), default='auto')
    p.add_argument('--candidates', type=int, choices=(1, 2, 3), default=2)
    p.add_argument('--crossfade-ms', type=float, default=500)
    p.add_argument('--entry-mode', choices=('auto', 'original', 'fade'), default='auto')
    p.add_argument('--fade-in-ms', type=float)
    return p


def prepare_output(output, analysis_source, audio_source):
    """Validate only; publish creates the directory after generation succeeds."""
    output = Path(output).expanduser().resolve()
    sources = [Path(analysis_source).resolve(), Path(audio_source).resolve()]
    if output.parent != sources[1].parent or output in sources or output in [p.parent for p in sources]:
        raise ValueError('输出必须是原音频目录的独立子目录，不能是原目录、输入文件或分析目录')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('输出目录已有文件；请指定新目录，不覆盖或清理旧结果')
    return output


def _default_output(analysis, duration, mode, multistage=False):
    title = re.sub(r'[^\w\s-]', '_', analysis.audio_path.stem, flags=re.UNICODE).strip(' _') or '音乐'
    title = re.sub(r'\s+', ' ', title)[:80]
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    kind = '剪辑方案' if mode == 'plan' else '音乐'
    seconds = f'{duration:g}'.replace('.', 'p')
    timing = f'参考{seconds}秒_多阶段' if multistage else f'{seconds}秒'
    return analysis.audio_path.parent/f'{title}_{timing}_{kind}_{stamp}_{uuid.uuid4().hex[:8]}'


def compact_summary(record):
    keys = ('status', 'mode', 'planning_mode', 'tool_version', 'output_directory', 'source_audio',
            'sample_rate', 'target_frames', 'requested_duration_s', 'reference_duration_s',
            'duration_is_exact', 'crossfade_ms',
            'entry_mode', 'gain_db', 'cache_cleaned', 'elapsed_seconds', 'error', 'cleanup_error',
            'remaining_cache_directory', 'plan_markdown', 'plan_json', 'editing_suggestions_markdown')
    result = {key: record[key] for key in keys if key in record}
    for key in ('requested_duration_s', 'reference_duration_s', 'crossfade_ms', 'gain_db'):
        if key in result and not math.isfinite(float(result[key])):
            result[key] = None
    for key in ('error', 'cleanup_error'):
        if key in result:
            result[key] = str(result[key])[:1000]
    result['source_warnings'] = [str(w)[:180] for w in record.get('source_warnings', [])[:4]]
    result['candidates'] = []
    fields = ('wav', 'mp3', 'strategy', 'score', 'duration_s', 'source_segments_s',
              'output_segments_s', 'actual_crossfade_ms', 'crossfade_output_s',
              'join_crossfade_ms', 'crossfade_regions_s', 'target_frames', 'stage_summary',
              'entry_fade_ms', 'end_fade_ms', 'entry_fade_output_s', 'end_fade_output_s',
              'editing_suggestion_count')
    for candidate in record.get('candidates', [])[:3]:
        item = {key: candidate[key] for key in fields if key in candidate}
        item['warnings'] = [str(w)[:180] for w in candidate.get('warnings', [])[:4]]
        result['candidates'].append(item)
    def size():
        return len(json.dumps(result, ensure_ascii=False, allow_nan=False).encode('utf-8'))
    if size() > 8000:
        for candidate in result['candidates']:
            for key in ('wav', 'mp3'):
                if key in candidate:
                    candidate[key] = Path(candidate[key]).name
        for key in ('plan_markdown', 'plan_json', 'editing_suggestions_markdown'):
            if key in result:
                result[key] = Path(result[key]).name
        result['file_paths_relative_to_output_directory'] = True
    if size() > 8000:
        for candidate in result['candidates']:
            if candidate.get('stage_summary'):
                candidate['stage_summary'] = [
                    {key: stage[key] for key in ('label', 'output_core_s')}
                    for stage in candidate['stage_summary']]
        result['stage_summary_reduced'] = True
    while size() > 8000:
        warnings = [result['source_warnings']]+[c['warnings'] for c in result['candidates']]
        populated = [items for items in warnings if items]
        if not populated:
            break
        max(populated, key=lambda items: len(str(items))).pop()
    if size() > 8000:
        result = {key: result[key] for key in ('status', 'mode', 'planning_mode', 'duration_is_exact',
                                             'cache_cleaned', 'requested_duration_s') if key in result}
        result['candidate_durations_s'] = [candidate.get('duration_s')
                                          for candidate in record.get('candidates', [])[:3]]
        result.update(output_directory=str(record.get('output_directory', ''))[:750],
                      summary_reduced_for_path_length=True)
        for key in ('error', 'cleanup_error'):
            if key in record:
                result[key] = str(record[key])[:500]
        if 'remaining_cache_directory' in record:
            result['remaining_cache_directory'] = str(record['remaining_cache_directory'])[:750]
    return result


def _binary(name):
    found = shutil.which(name)
    if not found:
        fallback = Path(r'D:\Program\ffmpeg\bin')/(name+'.exe')
        if fallback.is_file():
            found = str(fallback)
    if not found:
        raise FileNotFoundError(f'找不到{name}，请将现有工具加入PATH')
    return found


def _process(args, data=None):
    p = subprocess.run(args, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if p.returncode:
        raise RuntimeError(p.stderr.decode('utf-8', errors='replace')[-2000:])
    return p


def _export(audio, sr, target, stem, output, ffmpeg, ffprobe):
    wav, mp3 = output/(stem+'.wav'), output/(stem+'.mp3')
    _process([ffmpeg, '-v', 'error', '-n', '-f', 'f32le', '-ar', str(sr), '-ac', '2', '-i', 'pipe:0',
              '-c:a', 'pcm_s24le', '-metadata', 'title='+stem, str(wav)], audio.astype('<f4').tobytes())
    _process([ffmpeg, '-v', 'error', '-n', '-i', str(wav), '-c:a', 'libmp3lame', '-b:a', '320k', str(mp3)])
    probe = json.loads(_process([ffprobe, '-v', 'error', '-select_streams', 'a:0',
                                '-show_entries', 'stream=duration_ts,time_base,sample_rate,channels',
                                '-of', 'json', str(wav)]).stdout)
    stream = probe['streams'][0]
    if int(stream['duration_ts']) != target or stream['time_base'] != f'1/{sr}':
        raise RuntimeError('导出后样本数与目标不一致')
    _process([ffmpeg, '-v', 'error', '-i', str(mp3), '-f', 'null', os.devnull])
    return dict(wav=str(wav), mp3=str(mp3), duration_s=target/sr,
                verified_wav_frames=target, mp3_full_decode=True,
                sample_peak_dbfs=round(20*np.log10(float(np.max(np.abs(audio)))+1e-15), 3))


def _candidate_record(plan, sr, target):
    source = [[a/sr, b/sr] for a, b in plan.segments]
    overlaps = plan_overlaps(plan)
    timeline = plan_timeline(plan)
    placement = [[a/sr, b/sr] for a, b in timeline]
    regions = [[timeline[i+1][0]/sr, timeline[i][1]/sr]
               for i, overlap in enumerate(overlaps) if overlap]
    stages = [{key: stage[key] for key in ('label', 'preferred_duration_s', 'actual_duration_s',
                                         'deviation_s', 'output_core_s')}
              for stage in plan.evidence.get('stages', [])]
    return dict(strategy=plan.strategy, score=round(float(plan.score), 6),
                duration_s=target/sr, target_frames=target, source_segments_s=source,
                output_segments_s=placement, stage_summary=stages,
                actual_crossfade_ms=max(overlaps, default=0)/sr*1000,
                join_crossfade_ms=[overlap/sr*1000 for overlap in overlaps],
                crossfade_regions_s=regions,
                crossfade_output_s=regions[0] if len(regions) == 1 else None,
                entry_fade_ms=plan.start_fade_frames/sr*1000,
                end_fade_ms=plan.end_fade_frames/sr*1000,
                entry_fade_output_s=[0, plan.start_fade_frames/sr] if plan.start_fade_frames else None,
                end_fade_output_s=[(target-plan.end_fade_frames)/sr, target/sr] if plan.end_fade_frames else None,
                warnings=list(plan.warnings), plan=asdict(plan))


def _editing_suggestions(analysis, plan, sr, target):
    """Map key editing points (stage transitions, structural cuts, emphasis peaks) to the edited timeline."""
    overlaps, timeline = plan_overlaps(plan), plan_timeline(plan)
    avoid_regions = []

    def avoid(start, end, label, reason):
        if end > start:
            avoid_regions.append(dict(start_frame=start, end_frame=end,
                                      start_s=start/sr, end_s=end/sr,
                                      label=label, reason=reason))

    for index, overlap in enumerate(overlaps):
        if overlap:
            avoid(timeline[index+1][0], timeline[index][1], f'接缝{index+1}交叉淡化区',
                  '两段音乐叠加，原拍点不一定对应成品清晰重音，需试听后决定画面落点。')
    avoid(0, plan.start_fade_frames, '开头淡入区',
          '起音电平经过淡入，不把原拍点直接当作可靠重音。')
    avoid(target-plan.end_fade_frames, target, '结尾淡出区',
          '适合画面收束或字幕停留，避免密集切换；不据此认定音乐可提前截断。')

    events = []
    stages = plan.evidence.get('stages', [])
    joins = plan.evidence.get('joins', [])
    for idx in range(1, len(stages)):
        prev_label = stages[idx-1]['label']
        curr_label = stages[idx]['label']
        out_f = int(stages[idx]['output_core_frames'][0])
        src_f = int(stages[idx]['source_core_frames'][0])
        if not 0 < out_f < target:
            continue
        seg_idx = min(idx+1, len(plan.segments))
        # Find exact segment containing out_f on output timeline
        for s_i, ((s_start, s_end), (p_start, p_end)) in enumerate(zip(plan.segments, timeline), 1):
            if p_start <= out_f < p_end:
                seg_idx = s_i
                src_f = s_start + (out_f - p_start)
                break
        is_splice = idx-1 < len(joins) and joins[idx-1].get('kind') != '原曲连续演奏'
        join_desc = '跨段拼接点（含交叉淡化）' if is_splice else '原曲连续自然过渡'
        marker = dict(source_time_s=src_f/sr, source_frame=src_f,
                      segment_index=seg_idx, kind='stage_transition')
        events.append(dict(
            frame=out_f, rank=0,
            label='阶段切换点',
            suggested_use=f'阶段切换（{prev_label} → {curr_label}）：{join_desc}，适合画面转场或叙事阶段切换。',
            marker=marker,
        ))

    raw_cuts = getattr(analysis, 'cut_points', []) or []
    cut_times_by_kind = {'picture_cut': [], 'picture_emphasis': []}
    for item in raw_cuts:
        t = item.get('time_s')
        tp = item.get('type')
        if t is not None and tp in cut_times_by_kind:
            cut_times_by_kind[tp].append(float(t))

    struct_times = sorted({float(t) for t in analysis.section_edges if 0 < t < analysis.duration}
                          | set(cut_times_by_kind['picture_cut']))
    emphasis_times = sorted(set(cut_times_by_kind['picture_emphasis']))

    marker_sets = (
        ('section_boundary', struct_times, 1, '段落结构切点',
         '原曲段落结构切点：音乐进入新段落或音色变化，适合切换大场景或画面段落。'),
        ('emphasis', emphasis_times, 2, '核心重音/能量峰值点',
         '音乐核心重音/能量峰值点：适合关键动作落点、视觉冲击或重点特写。'),
    )
    segment_has_important = {i: False for i in range(1, len(plan.segments)+1)}
    for ev in events:
        segment_has_important[ev['marker']['segment_index']] = True

    for index, (start, end) in enumerate(plan.segments, 1):
        offset = timeline[index-1][0]
        for kind, times, rank, label, use in marker_sets:
            for time_s in times:
                source_frame = round(float(time_s)*sr)
                if not start <= source_frame < end:
                    continue
                output_frame = offset+source_frame-start
                if not 0 < output_frame < target:
                    continue
                if any(r['start_frame'] <= output_frame < r['end_frame']
                       for r in avoid_regions):
                    continue
                segment_has_important[index] = True
                marker = dict(source_time_s=source_frame/sr, source_frame=source_frame,
                              segment_index=index, kind=kind)
                events.append(dict(frame=output_frame, rank=rank, label=label,
                                   suggested_use=use, marker=marker))

    # Fallback: if a long segment has no stage transition, section boundary or emphasis point,
    # pick representative downbeats spaced at least 6.0s apart.
    min_gap_frames = round(6.0*sr)
    for index, (start, end) in enumerate(plan.segments, 1):
        if segment_has_important[index]:
            continue
        offset = timeline[index-1][0]
        last_picked = None
        for time_s in analysis.downbeats:
            source_frame = round(float(time_s)*sr)
            if not start <= source_frame < end:
                continue
            output_frame = offset+source_frame-start
            if not 0 < output_frame < target:
                continue
            if any(r['start_frame'] <= output_frame < r['end_frame']
                   for r in avoid_regions):
                continue
            if last_picked is not None and output_frame - last_picked < min_gap_frames:
                continue
            last_picked = output_frame
            marker = dict(source_time_s=source_frame/sr, source_frame=source_frame,
                          segment_index=index, kind='downbeat_key')
            events.append(dict(
                frame=output_frame, rank=3, label='重点小节起点',
                suggested_use='阶段内部核心小节起点：可作为主要画面切换或动作落点候选。',
                marker=marker,
            ))

    # Merge nearby important markers within 600ms, prioritizing stage transitions and structural cuts.
    tolerance = round(0.600*sr)
    groups = []
    for event in sorted(events, key=lambda e: (e['frame'], e['rank'])):
        if not groups or event['frame']-groups[-1][0]['frame'] > tolerance:
            groups.append([])
        groups[-1].append(event)
    points = []
    for group in groups:
        anchor = min(group, key=lambda e: (e['rank'], e['frame']))
        points.append(dict(output_time_s=anchor['frame']/sr, output_frame=anchor['frame'],
                           label=anchor['label'], suggested_use=anchor['suggested_use'],
                           source_markers=[e['marker'] for e in group]))
    notes = [
        '全部位置基于剪后音乐的计划/WAV采样时间轴，从成品0秒计时；不是原曲时间或视频帧号。',
        '仅保留重要剪辑点（阶段切换点、段落结构切点、核心重音/能量峰值点），已过滤密集常规拍点；600毫秒内相近标记已合并。',
        '淡化区内的常规标记不列为推荐点；其余点仅为画面卡点候选，需结合成品试听。',
        '画面剪辑点不等于音乐可安全截断的位置；没有视频帧率时不换算视频帧号。',
    ]
    empty_reason = (None if points else
                    '保留片段内没有可映射的重要结构或重音剪辑点，未编造推荐点。')
    return dict(time_axis='output', time_unit='s', sample_rate=sr, output_frames=target,
                method='important_markers_only', merge_tolerance_ms=600,
                points=points, avoid_regions=avoid_regions, notes=notes, empty_reason=empty_reason)


def _stage_markdown(candidate, include_evidence=False):
    stages = candidate['plan']['evidence'].get('stages', [])
    text = f'\n实际成品时长：{candidate["duration_s"]:.6f}秒，{candidate["target_frames"]}采样帧。\n'
    if not stages:
        return text
    text += '\n### 阶段建议时间表\n\n阶段秒数是参考；核心区间是成品的阶段分配，取材区间另含交叉淡化余量。\n\n'
    text += '| 阶段 | 参考秒数 | 建议秒数 | 偏差 | 成品核心区间 | 原曲核心区间 |\n|---|---|---|---|---|---|\n'
    for stage in stages:
        output, source = stage['output_core_s'], stage['source_core_s']
        text += (f'| {stage["label"]} | {stage["preferred_duration_s"]:g} | '
                 f'{stage["actual_duration_s"]:.6f} | {stage["deviation_s"]:+.6f} | '
                 f'{output[0]:.6f}—{output[1]:.6f} | {source[0]:.6f}—{source[1]:.6f} |\n')
    if include_evidence:
        for stage in stages:
            text += '\n'+stage['label']+'的候选依据：\n\n    '
            text += json.dumps(stage['evidence'], ensure_ascii=False, indent=2).replace('\n', '\n    ')+'\n'
            for warning in stage['warnings']:
                text += '\n- '+str(warning)+'\n'
    return text


def _suggestions_markdown(candidate):
    suggestions = candidate['editing_suggestions']
    text = '\n### 新音乐剪辑点建议\n\n'
    text += '\n'.join('- '+note for note in suggestions['notes'])+'\n'
    if suggestions['points']:
        text += '\n| 新音乐位置（秒） | 标记 | 建议用途 | 原曲位置 / 取材片段 |\n|---|---|---|---|\n'
        for point in suggestions['points']:
            sources = sorted({(m['segment_index'], m['source_time_s'])
                              for m in point['source_markers']})
            source_text = '；'.join(f'{time_s:.6f}秒 / 片段{index}' for index, time_s in sources)
            text += (f'| {point["output_time_s"]:.6f} | {point["label"]} | '
                     f'{point["suggested_use"]} | {source_text} |\n')
    else:
        text += '\n'+suggestions['empty_reason']+'\n'
    if suggestions['avoid_regions']:
        text += '\n建议先试听再决定卡点的区域（成品时间，含起点、不含终点）：\n\n'
        for region in suggestions['avoid_regions']:
            text += (f'- {region["start_s"]:.6f}—{region["end_s"]:.6f}秒，'
                     f'{region["label"]}：{region["reason"]}\n')
    return text


def _audio_suggestions_markdown(record):
    now = record['created']
    text = f'''---
created: {now}
modified: {now}
tags: [音乐剪辑, 画面卡点]
status: 完成
type: 报告
project: 音乐自动剪辑
---

# 新音乐剪辑点建议

原音频：{record['source_audio']}
分析JSON：{record['analysis_json']}
参考/指定时长：{record['requested_duration_s']:g}秒；是否锁定总长：{record['duration_is_exact']}；{record['sample_rate']}Hz。
各候选的实际时长、采样帧数和阶段建议分别列出。
以下按候选分别列出；对齐画面请以对应WAV为准。MP3播放器的时长显示可能受编码填充影响。
本文件给出画面卡点候选，未代替用户试听，也不表示音乐可在这些点直接切断。
'''
    for warning in record['source_warnings']:
        text += '\n- 原分析警告：'+str(warning)+'\n'
    for number, candidate in enumerate(record['candidates'], 1):
        text += f'\n## 候选{number:02d}：{candidate["strategy"]}\n\n'
        text += f'WAV：{Path(candidate["wav"]).name}\n\nMP3：{Path(candidate["mp3"]).name}\n'
        text += _stage_markdown(candidate)
        for index, (source, placed) in enumerate(zip(candidate['source_segments_s'], candidate['output_segments_s']), 1):
            text += (f'\n片段{index}：原曲{source[0]:.6f}—{source[1]:.6f}秒，'
                     f'放在成品{placed[0]:.6f}—{placed[1]:.6f}秒。\n')
        for warning in candidate['warnings']:
            text += '\n- '+str(warning)+'\n'
        text += _suggestions_markdown(candidate)
    return text


def _plan_markdown(record):
    now = record['created']
    text = f'''---
created: {now}
modified: {now}
tags: [音乐剪辑, 剪辑方案]
status: 完成
type: 报告
project: 音乐自动剪辑
---

# 音乐剪辑方案

原音频：{record['source_audio']}
原音频SHA256：{record['source_audio_sha256']}
分析JSON：{record['analysis_json']}
分析SHA256：{record['analysis_sha256']}
参考/指定时长：{record['requested_duration_s']:g}秒；是否锁定总长：{record['duration_is_exact']}；{record['sample_rate']}Hz。
各候选实际时长、采样帧数及阶段建议分别列出。
入口模式：{record['entry_mode']}；固定增益：{record['gain_db']:.4f}dB。
配置内部交叉淡化：{record['crossfade_ms']:g}毫秒；连续段实际交叉淡化为0。

本次只生成方案，没有渲染或导出音乐。方案来自实际音频分析与接点搜索，音乐性仍需试听核对。
区间含起点、不含终点，保持原速。源取材时间与成品放置时间是两条轴；重叠处前段线性淡出、后段线性淡入。
固定增益保留动态，是RMS及样本峰值约束下的电平调整，不是LUFS归一化。
'''
    for number, candidate in enumerate(record['candidates'], 1):
        text += f'\n## 候选{number}：{candidate["strategy"]}\n\n'
        text += _stage_markdown(candidate, include_evidence=True)
        text += '| 片段 | 原曲取材区间 | 成品放置区间 |\n|---|---|---|\n'
        for index, (source, placed) in enumerate(zip(candidate['source_segments_s'], candidate['output_segments_s']), 1):
            text += f'| {index} | {source[0]:.6f}—{source[1]:.6f}秒 | {placed[0]:.6f}—{placed[1]:.6f}秒 |\n'
        text += f'\n逐接点交叉淡化毫秒数：{candidate["join_crossfade_ms"]}；全部成品混合区间：{candidate["crossfade_regions_s"]}。\n'
        text += f'开头淡入：{candidate["entry_fade_ms"]:g}毫秒，成品范围：{candidate["entry_fade_output_s"]}。\n'
        text += f'结尾淡出：{candidate["end_fade_ms"]:g}毫秒，成品范围：{candidate["end_fade_output_s"]}。\n'
        text += '\n'+'\n'.join('- '+str(w) for w in candidate['warnings'])+'\n'
        text += '\n接点和出口声学依据（不保证完整乐句或终止式）：\n\n    '
        evidence = {key: value for key, value in candidate['plan']['evidence'].items() if key != 'stages'}
        text += json.dumps(evidence, ensure_ascii=False, indent=2).replace('\n', '\n    ')+'\n'
        text += _suggestions_markdown(candidate)
    return text+'\n所有具体样本区间及原始计划保存在同目录剪辑方案.json；无需读取本次结束后删除的缓存。\n'


def _publish(staged, output):
    """Only remove files created by this publication if copying fails."""
    created, owns_directory = [], False
    try:
        try:
            output.mkdir(exist_ok=False)
            owns_directory = True
        except FileExistsError:
            if not output.is_dir() or any(output.iterdir()):
                raise FileExistsError('发布时结果目录已被占用，未覆盖已有文件')
        for source in sorted(staged.iterdir()):
            if not source.is_file():
                raise ValueError('暂存交付物必须是普通文件')
            target = output/source.name
            with target.open('xb') as destination:
                created.append(target)
                with source.open('rb') as stream:
                    shutil.copyfileobj(stream, destination)
        return created
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        if owns_directory and output.is_dir() and not any(output.iterdir()):
            output.rmdir()
        raise


def _cleanup_run_cache(run_cache, cache_base):
    actual, parent = run_cache.resolve(), cache_base.resolve()
    if actual.parent != parent or not actual.name.startswith('run-') or run_cache.is_symlink():
        raise ValueError('拒绝删除不属于本次任务的缓存路径')
    if run_cache.exists():
        shutil.rmtree(run_cache)


def main(argv=None):
    args = build_parser().parse_args(argv)
    start = time.perf_counter()
    features = run_cache = cache_base = None
    result = dict(status='未生成可用结果', mode=args.mode,
                  requested_duration_s=args.duration if args.duration is not None else 0.0,
                  crossfade_ms=args.crossfade_ms, entry_mode=args.entry_mode, candidates=[])
    code = 2
    try:
        analysis = load_analysis(args.analysis_json, args.audio)
        stages = parse_stages(args.stages) if args.stages is not None else None
        if stages is None and args.duration is None:
            raise ValueError('请给出--duration精确秒数，或--stages阶段及参考秒数')
        reference = sum(stage['preferred_duration_s'] for stage in stages) if stages else args.duration
        requested = args.duration if args.duration is not None else reference
        result.update(requested_duration_s=requested, reference_duration_s=reference,
                      duration_is_exact=args.duration is not None)
        validate_request(args.duration if args.duration is not None else min(reference, analysis.duration),
                         48000, args.crossfade_ms, analysis.duration)
        if args.fade_in_ms is not None and (not math.isfinite(args.fade_in_ms) or
                                           not 0 <= args.fade_in_ms < requested*1000):
            raise ValueError('开头淡入毫秒数必须有限、大于等于0且短于目标时长')
        output = prepare_output(args.output or _default_output(analysis, requested, args.mode,
                                                               multistage=bool(stages)),
                                analysis.json_path, analysis.audio_path)
        result['output_directory'] = str(output)
        cache_base = Path(args.cache).expanduser().resolve()
        if cache_base == output or cache_base.is_relative_to(output):
            raise ValueError('缓存根目录不能是结果目录或其内部目录')
        cache_base.mkdir(parents=True, exist_ok=True)
        run_cache = Path(tempfile.mkdtemp(prefix='run-', dir=cache_base)).resolve()
        staged = run_cache/'deliverables'
        staged.mkdir()
        ffmpeg = _binary('ffmpeg')
        features = load_features(analysis, run_cache, ffmpeg=ffmpeg)
        if stages:
            plans = plan_stages(analysis, features, stages, crossfade_ms=args.crossfade_ms,
                                limit=args.candidates, strategy=args.strategy,
                                entry_mode=args.entry_mode, fade_in_ms=args.fade_in_ms,
                                exact_duration=args.duration)
        else:
            plans = plan_candidates(analysis, features, args.duration, crossfade_ms=args.crossfade_ms,
                                    limit=args.candidates, strategy=args.strategy,
                                    entry_mode=args.entry_mode, fade_in_ms=args.fade_in_ms)
        active = features.rms_db[features.rms_db > -50]
        gain_db = float(-20-np.median(active)) if len(active) else 0.0
        gain_db = min(12, gain_db, -3-20*np.log10(float(np.max(np.abs(features.audio)))+1e-15))
        candidates = []
        ffprobe = _binary('ffprobe') if args.mode == 'audio' else None
        for index, plan in enumerate(plans, 1):
            target = int(plan.evidence['output_frames']) if stages else round(args.duration*features.sr)
            if target <= 0 or plan_timeline(plan)[-1][1] != target:
                raise ValueError('候选实际时间轴与总样本数不一致，未发布结果')
            if args.duration is not None and target != round(args.duration*features.sr):
                raise ValueError('候选不满足用户明确指定的精确总长，未发布结果')
            candidate = _candidate_record(plan, features.sr, target)
            candidate['editing_suggestions'] = _editing_suggestions(analysis, plan, features.sr, target)
            candidate['editing_suggestion_count'] = len(candidate['editing_suggestions']['points'])
            if args.mode == 'audio':
                audio = render_plan(features.audio, plan, target, gain_db=gain_db)
                stem = f'候选{index:02d}_{target/features.sr:g}秒_{plan.strategy}'
                candidate.update(_export(audio, features.sr, target, stem, staged, ffmpeg, ffprobe))
                for key in ('wav', 'mp3'):
                    candidate[key] = str(output/Path(candidate[key]).name)
            candidates.append(candidate)
        now = datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M')
        if not candidates:
            raise ValueError('未找到可交付的音乐候选，未发布空结果目录')
        frame_counts = {candidate['target_frames'] for candidate in candidates}
        result.update(created=now, tool_version='0.5', status='剪辑方案已生成' if args.mode == 'plan' else '待试听',
                      source_audio=str(analysis.audio_path), source_audio_sha256=features.audio_sha256,
                      analysis_json=str(analysis.json_path),
                      analysis_sha256=hashlib.sha256(analysis.json_path.read_bytes()).hexdigest(),
                      target_frames=next(iter(frame_counts)) if len(frame_counts) == 1 else None,
                      sample_rate=features.sr, gain_db=round(gain_db, 4),
                      planning_mode='stages' if stages else 'duration', stage_request=stages,
                      gain_method='fixed active-median RMS target -20dBFS with sample peak cap -3dBFS; not LUFS',
                      source_warnings=list(analysis.warnings), candidates=candidates,
                      verification=dict(human_auditioned=False, user_approved=False),
                      cache_policy='per-run temporary cache removed after completion or failure')
        if args.mode == 'plan':
            result.update(plan_markdown=str(output/'剪辑方案.md'), plan_json=str(output/'剪辑方案.json'))
            (staged/'剪辑方案.md').write_text(_plan_markdown(result), encoding='utf-8')
            (staged/'剪辑方案.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        else:
            result['editing_suggestions_markdown'] = str(output/'剪辑点建议.md')
            (staged/'剪辑点建议.md').write_text(_audio_suggestions_markdown(result), encoding='utf-8')
        _publish(staged, output)
        code = 0
    except (ValueError, FileNotFoundError, FileExistsError, OSError, RuntimeError) as error:
        result.update(status='未生成可用结果', error=str(error), candidates=[])
    finally:
        cleanup_error = None
        try:
            if features is not None:
                _close_memmap(features.audio)
            features = None
            gc.collect()
            if run_cache is not None:
                _cleanup_run_cache(run_cache, cache_base)
        except Exception as error:
            cleanup_error = str(error)
        result['cache_cleaned'] = cleanup_error is None
        if cleanup_error is not None:
            result['cleanup_error'] = cleanup_error
            if run_cache is not None:
                result['remaining_cache_directory'] = str(run_cache)
            if code == 0:
                result['status'] = '结果已生成但缓存清理失败'
            code = 2
        result['elapsed_seconds'] = round(time.perf_counter()-start, 3)
    print(json.dumps(compact_summary(result), ensure_ascii=False, allow_nan=False))
    return code


if __name__ == '__main__':
    if sys.platform.startswith('win'):
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, 'reconfigure'):
                stream.reconfigure(encoding='utf-8')
    raise SystemExit(main())
