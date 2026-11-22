"""口播原始转写稿与用户参考原稿对照及无损纠错回写工具（仅输出确认版 .srt 与 *_逐字时间.json 两个文件）。"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

from scan_talking_head import parse_srt  # noqa: E402
from transcribe_talking_head import extract_core_units, format_srt_ms  # noqa: E402


def _locate_cue_token_ranges(
    cues: Sequence[Dict[str, Any]],
    char_tokens: Sequence[Dict[str, Any]],
) -> List[Tuple[int, int]]:
    """
    将 SRT 条目按时间窗与核心字符序列，一一映射为 char_tokens 中的连续切片范围 [start_idx, end_idx)。
    确保未修改片段的时间戳保持绝对原样，修改片段精确锁定音频起止边界。
    """
    ranges: List[Tuple[int, int]] = []
    n_tokens = len(char_tokens)
    cursor = 0

    for idx, cue in enumerate(cues):
        c_start = float(cue["start"])
        c_end = float(cue["end"])
        next_start = float(cues[idx + 1]["start"]) if idx + 1 < len(cues) else (c_end + 10.0)
        units = extract_core_units(str(cue["text"]))

        while cursor < n_tokens and float(char_tokens[cursor].get("end_time", 0.0)) < c_start - 0.35:
            cursor += 1

        if not units:
            ranges.append((cursor, cursor))
            continue

        # 收窄搜索窗口：截至下一句开始前或合理富余字符量，防跨句错配
        time_limit = max(c_end + 0.35, min(next_start + 0.15, c_end + 1.5))
        cand_end = cursor
        while cand_end < n_tokens and (
            float(char_tokens[cand_end].get("start_time", 0.0)) <= time_limit
            or cand_end - cursor < len(units)
        ):
            cand_end += 1
            if cand_end - cursor >= len(units) + 12 and float(char_tokens[cand_end - 1].get("start_time", 0.0)) > c_end + 0.35:
                break

        window_texts = [str(t.get("text", "")) for t in char_tokens[cursor:cand_end]]
        if not window_texts:
            ranges.append((cursor, cursor))
            continue

        sm = difflib.SequenceMatcher(None, window_texts, units, autojunk=False)
        blocks = [b for b in sm.get_matching_blocks() if b.size > 0]
        if blocks:
            first_b = blocks[0]
            last_b = blocks[-1]
            rel_start = max(0, first_b.a - first_b.b)
            missing_tail = len(units) - (last_b.b + last_b.size)
            rel_end = min(len(window_texts), last_b.a + last_b.size + max(0, missing_tail))
            abs_start = cursor + rel_start
            abs_end = max(abs_start, cursor + rel_end)
        else:
            abs_start = cursor
            abs_end = min(n_tokens, cursor + len(units))

        ranges.append((abs_start, abs_end))
        cursor = abs_end

    return ranges


def _map_old_slice_to_units(
    old_slice: Sequence[Dict[str, Any]],
    old_units: Sequence[str],
    cue_start: float,
    cue_end: float,
) -> List[Tuple[float, float]]:
    """将 char_tokens 片段的时间精确对齐到原 SRT 句子的每一个核心单元上。"""
    if not old_units:
        return []
    if len(old_slice) == len(old_units):
        return [
            (
                float(t.get("start_time", cue_start)),
                float(t.get("end_time", t.get("start_time", cue_end))),
            )
            for t in old_slice
        ]

    if not old_slice:
        dur = max(0.05, cue_end - cue_start)
        step = dur / len(old_units)
        return [(round(cue_start + i * step, 4), round(cue_start + (i + 1) * step, 4)) for i in range(len(old_units))]

    slice_texts = [str(t.get("text", "")) for t in old_slice]
    sm = difflib.SequenceMatcher(None, slice_texts, list(old_units), autojunk=False)
    unit_times: List[Optional[Tuple[float, float]]] = [None] * len(old_units)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                unit_times[j1 + k] = (
                    float(old_slice[i1 + k].get("start_time", cue_start)),
                    float(old_slice[i1 + k].get("end_time", cue_end)),
                )
        elif tag == "replace" and (i2 - i1) > 0 and (j2 - j1) > 0:
            s_t = float(old_slice[i1].get("start_time", cue_start))
            e_t = float(old_slice[i2 - 1].get("end_time", cue_end))
            step = max(0.0, e_t - s_t) / (j2 - j1)
            for k in range(j2 - j1):
                unit_times[j1 + k] = (s_t + k * step, s_t + (k + 1) * step if k < (j2 - j1 - 1) else e_t)

    # 填补缺失插槽
    span_s = float(old_slice[0].get("start_time", cue_start))
    span_e = max(span_s, float(old_slice[-1].get("end_time", cue_end)))
    filled: List[Tuple[float, float]] = []
    for idx in range(len(old_units)):
        if unit_times[idx] is not None:
            filled.append(unit_times[idx])  # type: ignore[arg-type]
            continue
        prev_e = filled[-1][1] if filled else span_s
        next_s = span_e
        for nxt in range(idx + 1, len(old_units)):
            if unit_times[nxt] is not None:
                next_s = unit_times[nxt][0]  # type: ignore[index]
                break
        filled.append((prev_e, max(prev_e, next_s)))
    return filled


def rewrite_cue_tokens(
    old_text: str,
    new_text: str,
    old_slice: Sequence[Dict[str, Any]],
    cue_start: float,
    cue_end: float,
) -> List[Dict[str, Any]]:
    """
    对照修改前后同一句字幕，无损继承未改动字的原音频时间戳，并对替换字在原错误区间内精确对齐：
    - 等长错字替换（如 记录着 -> 记录者）：100% 原时间戳逐字保留；
    - 变长错字替换（如 哎哎 -> AI）：严格锁定被替换汉字区间 [start, end] 线性重分配。
    """
    old_units = extract_core_units(old_text)
    new_units = extract_core_units(new_text)
    if not new_units:
        return []

    old_unit_times = _map_old_slice_to_units(old_slice, old_units, cue_start, cue_end)
    if not old_unit_times:
        dur = max(0.05, cue_end - cue_start)
        step = dur / len(new_units)
        return [
            {
                "text": u,
                "start_time": round(cue_start + i * step, 4),
                "end_time": round(cue_start + (i + 1) * step, 4),
            }
            for i, u in enumerate(new_units)
        ]

    sm = difflib.SequenceMatcher(None, old_units, new_units, autojunk=False)
    out: List[Dict[str, Any]] = []
    span_s = old_unit_times[0][0]
    span_e = max(span_s, old_unit_times[-1][1])

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(j2 - j1):
                st, et = old_unit_times[i1 + k]
                out.append({"text": new_units[j1 + k], "start_time": round(st, 4), "end_time": round(max(st, et), 4)})
        elif tag == "replace":
            old_len = i2 - i1
            new_len = j2 - j1
            if old_len == new_len:
                for k in range(new_len):
                    st, et = old_unit_times[i1 + k]
                    out.append({"text": new_units[j1 + k], "start_time": round(st, 4), "end_time": round(max(st, et), 4)})
            else:
                seg_s = old_unit_times[i1][0]
                seg_e = max(seg_s, old_unit_times[i2 - 1][1])
                step = max(0.0, seg_e - seg_s) / new_len
                for k in range(new_len):
                    st = seg_s + k * step
                    et = seg_s + (k + 1) * step if k < new_len - 1 else seg_e
                    out.append({"text": new_units[j1 + k], "start_time": round(st, 4), "end_time": round(max(st, et), 4)})
        elif tag == "insert":
            new_len = j2 - j1
            anchor_s = out[-1]["end_time"] if out else (old_unit_times[i1][0] if i1 < len(old_unit_times) else span_s)
            anchor_e = old_unit_times[i1][0] if i1 < len(old_unit_times) else span_e
            if anchor_e < anchor_s:
                anchor_e = anchor_s
            step = (anchor_e - anchor_s) / new_len
            for k in range(new_len):
                st = anchor_s + k * step
                et = anchor_s + (k + 1) * step if k < new_len - 1 else anchor_e
                out.append({"text": new_units[j1 + k], "start_time": round(st, 4), "end_time": round(max(st, et), 4)})
        elif tag == "delete":
            continue

    return out


def inspect_reference_differences(
    cues: Sequence[Dict[str, Any]],
    reference_text: str,
) -> Dict[str, Any]:
    """
    纯只读对比“用户原口播稿”与“音频实际识别 SRT”：
    合并中间夹杂 <=3 字相等的碎裂差异（如 董[北]董 -> 东[北]东），附上前后上下文，
    同时自动略过原稿大段未念或即兴扩写区块，供 AI 助手精确生成 corrections.json。
    """
    ref_clean = re.sub(r"\s+", "", reference_text)
    ref_chars = list(ref_clean)
    suspects: List[Dict[str, Any]] = []

    asr_stream: List[Tuple[str, int]] = []
    for cue in cues:
        for ch in extract_core_units(str(cue["text"])):
            for c in ch:
                asr_stream.append((c, int(cue["id"])))

    ref_core = [c for ch in extract_core_units(ref_clean) for c in ch]
    asr_core = [x[0] for x in asr_stream]
    cue_by_id = {int(c["id"]): c for c in cues}

    sm = difflib.SequenceMatcher(None, asr_core, ref_core, autojunk=False)
    opcodes = sm.get_opcodes()

    # 将相距 <=3 个相同字符的局部差异区块合并为完整词语差异（例：记录着 <-> 记录者）
    merged_spans: List[Tuple[int, int, int, int]] = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            continue
        if merged_spans:
            pi1, pi2, pj1, pj2 = merged_spans[-1]
            if (i1 - pi2) <= 3 and (j1 - pj2) <= 3 and (i2 - pi1) <= 18 and (j2 - pj1) <= 18:
                merged_spans[-1] = (pi1, i2, pj1, j2)
                continue
        merged_spans.append((i1, i2, j1, j2))

    for i1, i2, j1, j2 in merged_spans:
        asr_frag = "".join(asr_core[i1:i2])
        ref_frag = "".join(ref_core[j1:j2])
        if not asr_frag or not ref_frag:
            continue
        if max(len(asr_frag), len(ref_frag)) > 16 and abs(len(asr_frag) - len(ref_frag)) > 6:
            continue
        touched_cue_ids = sorted({asr_stream[idx][1] for idx in range(i1, min(i2, len(asr_stream)))})
        ctx_asr = "".join(asr_core[max(0, i1 - 6) : min(len(asr_core), i2 + 6)])
        ctx_ref = "".join(ref_core[max(0, j1 - 6) : min(len(ref_core), j2 + 6)])
        suspects.append(
            {
                "cue_ids": touched_cue_ids,
                "asr_fragment": asr_frag,
                "reference_fragment": ref_frag,
                "context_asr": ctx_asr,
                "context_reference": ctx_ref,
                "cue_texts": {str(cid): cue_by_id[cid]["text"] for cid in touched_cue_ids if cid in cue_by_id},
            }
        )

    return {
        "total_cues": len(cues),
        "asr_core_char_count": len(asr_core),
        "reference_core_char_count": len(ref_core),
        "reference_preview": "".join(ref_chars[:120]),
        "candidate_typo_regions": suspects,
    }


def apply_corrections(
    srt_path: Path,
    char_json_path: Path,
    corrections: Dict[str, Any],
    out_srt_path: Optional[Path] = None,
    out_char_json_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    将 corrections.json（cue_fixes + global_replacements）同步无损回写至 .srt 与 *_逐字时间.json。
    严格保留非 1:1 照念的音频语序、重录和即兴词句，只输出确认版 .srt 与 *_逐字时间.json 两个文件。
    """
    cues = parse_srt(srt_path)
    char_tokens: List[Dict[str, Any]] = json.loads(char_json_path.read_text(encoding="utf-8-sig"))
    cue_ranges = _locate_cue_token_ranges(cues, char_tokens)

    cue_fixes_raw = corrections.get("cue_fixes") or {}
    cue_fixes: Dict[int, str] = {int(k): str(v) for k, v in cue_fixes_raw.items()}
    global_reps: List[Tuple[str, str]] = [
        (str(pair[0]), str(pair[1]))
        for pair in (corrections.get("global_replacements") or [])
        if isinstance(pair, (list, tuple)) and len(pair) == 2 and str(pair[0])
    ]

    new_cues: List[Dict[str, Any]] = []
    new_char_tokens: List[Dict[str, Any]] = []
    changed_cues: List[Dict[str, Any]] = []

    cursor = 0
    for idx, cue in enumerate(cues):
        seg_start_idx, seg_end_idx = cue_ranges[idx]
        if seg_start_idx > cursor:
            new_char_tokens.extend(char_tokens[cursor:seg_start_idx])

        old_slice = char_tokens[seg_start_idx:seg_end_idx]
        cursor = max(cursor, seg_end_idx)

        cue_id = int(cue["id"])
        old_text = str(cue["text"])
        new_text = old_text
        for src_pat, dst_pat in global_reps:
            new_text = new_text.replace(src_pat, dst_pat)
        if cue_id in cue_fixes:
            new_text = cue_fixes[cue_id].strip()

        if new_text != old_text:
            updated_slice = rewrite_cue_tokens(
                old_text=old_text,
                new_text=new_text,
                old_slice=old_slice,
                cue_start=float(cue["start"]),
                cue_end=float(cue["end"]),
            )
            new_char_tokens.extend(updated_slice)
            changed_cues.append(
                {
                    "id": cue_id,
                    "old_text": old_text,
                    "new_text": new_text,
                    "old_token_count": len(old_slice),
                    "new_token_count": len(updated_slice),
                }
            )
        else:
            new_char_tokens.extend(old_slice)

        new_cues.append({**cue, "text": new_text})

    if cursor < len(char_tokens):
        new_char_tokens.extend(char_tokens[cursor:])

    target_srt = out_srt_path or srt_path
    target_chars = out_char_json_path or char_json_path
    target_srt.parent.mkdir(parents=True, exist_ok=True)
    target_chars.parent.mkdir(parents=True, exist_ok=True)

    srt_blocks = [
        f"{c['id']}\n{format_srt_ms(float(c['start']))} --> {format_srt_ms(float(c['end']))}\n{c['text']}"
        for c in new_cues
    ]
    target_srt.write_text("\n\n".join(srt_blocks) + ("\n" if srt_blocks else ""), encoding="utf-8")
    target_chars.write_text(json.dumps(new_char_tokens, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "status": "corrected",
        "outputs": {
            "srt": str(target_srt),
            "char_json": str(target_chars),
        },
        "summary": {
            "total_cues": len(new_cues),
            "changed_cue_count": len(changed_cues),
            "char_token_count_before": len(char_tokens),
            "char_token_count_after": len(new_char_tokens),
        },
        "changed_cues": changed_cues,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="对照参考口播原稿核查并无损回写确认版 .srt 与 *_逐字时间.json")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_inspect = sub.add_parser("inspect", help="只读比对原口播稿与 ASR .srt，列出候选同音错字片段")
    p_inspect.add_argument("--srt", required=True, help="原始 ASR .srt 路径")
    p_inspect.add_argument("--ref", required=True, help="用户提供的原本口播内容 .txt/.md 路径")

    p_apply = sub.add_parser("apply", help="将纠错补丁无损回写至确认版 .srt 与 *_逐字时间.json（仅输出2个文件）")
    p_apply.add_argument("--srt", required=True, help="原始 .srt 路径")
    p_apply.add_argument("--chars", required=True, help="原始 *_逐字时间.json 路径")
    p_apply.add_argument("--corrections", required=True, help="纠错补丁 JSON（含 cue_fixes / global_replacements）")
    p_apply.add_argument("--out-srt", help="输出确认版 .srt 路径（默认原地更新 --srt）")
    p_apply.add_argument("--out-chars", help="输出确认版 *_逐字时间.json 路径（默认原地更新 --chars）")

    args = parser.parse_args()

    if args.cmd == "inspect":
        cues = parse_srt(Path(args.srt))
        ref_txt = Path(args.ref).read_text(encoding="utf-8-sig")
        res = inspect_reference_differences(cues, ref_txt)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return

    corr_data = json.loads(Path(args.corrections).read_text(encoding="utf-8-sig"))
    res = apply_corrections(
        srt_path=Path(args.srt),
        char_json_path=Path(args.chars),
        corrections=corr_data,
        out_srt_path=Path(args.out_srt) if args.out_srt else None,
        out_char_json_path=Path(args.out_chars) if args.out_chars else None,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
