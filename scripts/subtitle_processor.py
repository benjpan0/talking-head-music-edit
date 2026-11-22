"""中文字幕友好化处理引擎：标点清洗、实拍字级时间精确拆条、防闪屏、阅读延尾。"""

from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

TRAILING_STRIP_PUNCT = set("，。；：、,.;:")
KEEP_TRAILING_PUNCT = set("？！…?!")
CLOSING_WRAPPER_CHARS = set("”’」』）】")
SPLIT_INTERNAL_PUNCT = set("，；,;")
ENUM_CHAR = "、"
CORE_ALNUM_RE = re.compile(r"[A-Za-z0-9]+(?:[.\-+][A-Za-z0-9]+)*")

DEFAULT_DROP_FILLERS = {"呃", "嗯", "啊", "哦", "唉"}

# 在这些高频连接/介词/动词短语【前】拆分最符合中文短语阅读节奏
# 注意：绝不放孤立“为”，避免把“作为 / 成为 / 因为 / 为了”强行劈成两半
PRE_BREAK_WORDS = [
    ("或者说", 3.4),
    ("而且", 3.6),
    ("但是", 3.6),
    ("不过", 3.5),
    ("所以", 3.6),
    ("因此", 3.5),
    ("因为", 3.5),
    ("为了", 3.2),
    ("作为", 2.8),
    ("成为", 2.4),
    ("然后", 3.2),
    ("或者", 3.2),
    ("还是", 2.8),
    ("并且", 3.4),
    ("以及", 3.2),
    ("如果", 3.2),
    ("只要", 3.0),
    ("那么", 2.8),
    ("还想", 2.6),
    ("还要", 2.6),
    ("还能", 2.6),
    ("还会", 2.6),
    ("即将", 2.4),
    ("可以", 2.6),
    ("能够", 2.6),
    ("必须", 2.6),
    ("就是", 2.4),
    ("不是", 2.6),
    ("而是", 3.2),
    ("用来", 2.8),
    ("才能", 2.4),
    ("都会", 2.6),
    ("去", 2.0),
    ("把", 2.2),
    ("对", 1.8),
    ("给", 1.8),
    ("从", 1.8),
]

# 在这些话题/时间/语气助词【后】拆分最自然
POST_BREAK_WORDS = [
    ("的时候", 3.6),
    ("的话", 3.4),
    ("之后", 3.3),
    ("之前", 3.3),
    ("以后", 3.2),
    ("以前", 3.2),
    ("来说", 3.2),
    ("的话呢", 3.5),
    ("呢", 2.6),
    ("嘛", 2.6),
    ("啊", 2.5),
    ("吧", 2.5),
    ("了", 1.8),
]

# 通用不可拆保护词表（用户专属人名/品牌/专业词统一从本机 profile 的 glossary.terms 动态合并）
DEFAULT_PROTECTED_TERMS = [
    "背景音乐",
    "短视频",
    "生产力",
    "生活方式",
    "艺术品",
    "音乐库",
    "剪辑点",
    "这一期",
    "就这样子",
    "非常之",
    "十分之",
    "找出来",
    "做出来",
    "导出来",
    "说实在",
    "消费者",
    "制作工艺",
    "上个世纪",
    "几秒钟",
]


def _load_profile_glossary() -> Tuple[List[str], List[List[str]]]:
    """从本机配置档案读取用户常用词汇表（glossary.terms 与 glossary.replacements）。"""
    try:
        from portable_paths import load_machine_profile

        profile = load_machine_profile() or {}
        glossary = profile.get("glossary") or {}
        terms = [str(t).strip() for t in (glossary.get("terms") or []) if str(t).strip()]
        raw_rep = glossary.get("replacements") or {}
        replacements: List[List[str]] = []
        if isinstance(raw_rep, dict):
            for k, v in raw_rep.items():
                ks, vs = str(k).strip(), str(v).strip()
                if ks and vs:
                    replacements.append([ks, vs])
        return terms, replacements
    except Exception:
        return [], []

# 结构化中文词汇保护正则（不依赖 jieba 即能杜绝量词、合成词、成语、叠词被腰斩）
LEXICAL_PROTECT_PATTERNS = [
    r"(?:作|成|因|以|认|行)为",
    r"为了|为止",
    r"[一二三四五六七八九十两几这那每某]\s*[种个双条层件事样点段套款位家群名张本篇首支]",
    r"[这那每某]个(?:软件|地方|事情|问题|场景|节点|功能|作品)",
    r"第[一二三四五六七八九十百千]+[种个双条层件事次遍位名]?",
    r"([\u4e00-\u9fff])\1([\u4e00-\u9fff])\2",  # AABB 叠词如 清清楚楚
    r"(?:即将|已经|正在|可以|能够|必须|应该|需要|开始|继续|懂得|学会|想要|觉得|发现|找到|出现|产生|变成|代表|证明|形成|替代|打造|塑造|介绍|展示|呈现|表达|结合|探讨|站在)",
    r"(?:品牌|产品|功能|个性|境界|能力|工艺|手段|壁垒|外形|外表|材质|光影|氛围|气质|印记|魅力|价值|意义|方法|技巧|经验|习惯|节奏|情绪|感觉|故事|内容|画面|镜头|声音|音乐|配乐|音效|旋律|软件|工具|平台|网站|素材|版权|老师|朋友|大家|观众|高手|对手|自己|橡胶|帅哥|美女|旅游|登山|健身|艺术|精神|生活|风格|世界|时代|时间|问题)",
    r"(?:什么|怎么|为什么|哪些|这样|那样|这里|那里|现在|以前|以后|明天|今天|很多|许多|非常|特别|尤其|简单|普通|初级|高级|最高|最大|最初|唯一|真正|真实|所谓|重要|关键|核心|主要|基本|美好|独特|漂亮|容易)",
]

# 切分左侧末尾禁止悬挂的“悬空单字”（防 你的广告还 / 塑造一 / 即将成 这类断句）
DANGLING_LEFT_END_CHARS = set("还再也都只更最很挺太被把向往从与及而两几这那每某作将能会")


def visual_hanzi_width(text: str) -> float:
    """估算屏幕单行视觉宽度（汉字/全角=1.0，ASCII/半角≈0.45）。"""
    width = 0.0
    for ch in text:
        if ord(ch) < 128:
            width += 0.45
        else:
            width += 1.0
    return round(width, 2)


def is_core_token_char(ch: str) -> bool:
    """判断字符是否属于 forced-aligner 对齐的核心文字（汉字或字母数字）。"""
    return ("\u4e00" <= ch <= "\u9fff") or ch.isalnum()


def normalize_char_spans(tokens: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    展开并平滑修复 forced-aligner 逐字时间中的 0.000s 伪边界，保持严格非递减且不超前下一发声。
    每个输出元素保留原始与修正时间：{"ch", "start", "end"}。
    """
    expanded: List[Dict[str, Any]] = []
    for tok in tokens:
        text = str(tok.get("text", ""))
        st = float(tok.get("start", tok.get("start_time", 0.0)))
        et = float(tok.get("end", tok.get("end_time", st)))
        if not text:
            continue
        # 若一个 token 含多个字符（例如英文单词 AI / 123），记录按视觉长度分摊或整词边界
        if len(text) == 1 or text.isascii():
            expanded.append({"ch": text, "start": st, "end": max(st, et)})
        else:
            dur = max(0.0, et - st)
            step = dur / len(text)
            for idx, ch in enumerate(text):
                c_st = st + idx * step
                c_et = st + (idx + 1) * step if idx < len(text) - 1 else et
                expanded.append({"ch": ch, "start": c_st, "end": max(c_st, c_et)})

    n = len(expanded)
    for i in range(n):
        if expanded[i]["end"] - expanded[i]["start"] < 0.005:
            # 借用后一个有效非重叠字起点，上限给 0.06s，保证单字不呈 0 时长但也不吃进未来静音
            if i + 1 < n and expanded[i + 1]["start"] > expanded[i]["start"]:
                expanded[i]["end"] = min(expanded[i + 1]["start"], expanded[i]["start"] + 0.08)
            else:
                expanded[i]["end"] = expanded[i]["start"] + 0.04
    return expanded


def strip_trailing_punctuation(text: str) -> str:
    """
    按规范清洗句末标点：
    - 剥离句末 ，。；：、 与对应半角符号；
    - 保留 ？！… 及配对引号/书名号内外的合理结尾；
    - 若句末为 。” / 。） 这类情况，同时剥掉被包裹的句末句号/逗号。
    """
    chars = list(text.strip())
    while chars:
        last = chars[-1]
        if last in TRAILING_STRIP_PUNCT or last.isspace():
            chars.pop()
            continue
        if last in CLOSING_WRAPPER_CHARS and len(chars) >= 2 and chars[-2] in TRAILING_STRIP_PUNCT:
            chars.pop(-2)
            continue
        break
    return "".join(chars)


def _match_text_to_char_spans(
    clean_text: str,
    candidate_spans: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    把一个清洗/纠错后的字幕字符串每个字符（或保留完整英文 token）绑定到连续时间槽位。
    首先抽取 candidate_spans 的 [min_start, max_end]，按核心字符序列对齐；若因 typo 改写导致字符数增减，
    优先用 difflib 序列对齐真实音节时间，其余差值字在局部锚点之间精确分配，外边界 100% 锁定原区间。
    """
    import difflib

    spans = list(candidate_spans)
    if not spans:
        return []

    raw_units: List[Tuple[str, float, float]] = [(s["ch"], float(s["start"]), float(s["end"])) for s in spans]
    overall_start = raw_units[0][1]
    overall_end = max(u[2] for u in raw_units)

    # 把待绑定文本拆分为单元：连续 ASCII 字母数字合并为一个单元（对齐 forced-aligner 常见行为），中文逐字
    target_units: List[str] = []
    idx = 0
    while idx < len(clean_text):
        m = CORE_ALNUM_RE.match(clean_text, idx)
        if m:
            target_units.append(m.group(0))
            idx = m.end()
        else:
            target_units.append(clean_text[idx])
            idx += 1

    core_target_indices = [i for i, u in enumerate(target_units) if any(is_core_token_char(c) for c in u)]
    if not core_target_indices:
        return []

    raw_core = [u[0] for u in raw_units]
    tgt_core = [target_units[i] for i in core_target_indices]

    # 等长直接一一对应（最快、零误差）
    assigned_times: List[Tuple[float, float]] = [(overall_start, overall_end)] * len(core_target_indices)
    if len(raw_core) == len(tgt_core):
        for j in range(len(tgt_core)):
            assigned_times[j] = (raw_units[j][1], raw_units[j][2])
    else:
        sm = difflib.SequenceMatcher(a="".join(raw_core), b="".join(tgt_core))
        # 先映射核心字串每个单字符的时间，再聚合到 tgt_core 单元
        raw_char_times: List[Tuple[float, float]] = []
        for tok, st, et in raw_units:
            if len(tok) <= 1:
                raw_char_times.append((st, et))
            else:
                step = (et - st) / len(tok)
                for k in range(len(tok)):
                    raw_char_times.append((st + k * step, st + (k + 1) * step))

        tgt_flat = "".join(tgt_core)
        tgt_char_times: List[Optional[Tuple[float, float]]] = [None] * len(tgt_flat)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for off in range(i2 - i1):
                    tgt_char_times[j1 + off] = raw_char_times[i1 + off]
            elif tag == "replace" and j2 > j1 and i2 > i1:
                st = raw_char_times[i1][0]
                et = raw_char_times[i2 - 1][1]
                step = (et - st) / (j2 - j1)
                for off in range(j2 - j1):
                    tgt_char_times[j1 + off] = (st + off * step, st + (off + 1) * step)
            elif tag == "insert" and j2 > j1:
                left_t = raw_char_times[i1 - 1][1] if i1 > 0 else overall_start
                right_t = raw_char_times[i1][0] if i1 < len(raw_char_times) else overall_end
                span = max(0.04 * (j2 - j1), right_t - left_t)
                step = span / (j2 - j1)
                for off in range(j2 - j1):
                    tgt_char_times[j1 + off] = (left_t + off * step, left_t + (off + 1) * step)

        # 填补残余 None 并映射回 tgt_core
        for k in range(len(tgt_char_times)):
            if tgt_char_times[k] is None:
                prev_t = tgt_char_times[k - 1][1] if k > 0 and tgt_char_times[k - 1] else overall_start
                tgt_char_times[k] = (prev_t, min(overall_end, prev_t + 0.05))

        cursor = 0
        for j, unit in enumerate(tgt_core):
            end_cur = cursor + len(unit)
            st = tgt_char_times[cursor][0]  # type: ignore[index]
            et = tgt_char_times[end_cur - 1][1]  # type: ignore[index]
            assigned_times[j] = (st, max(st + 0.01, et))
            cursor = end_cur

    # 把核心单元的时间附加标点一起组装回去，用于按完整展示单位切分
    core_pos_map = {ci: assigned_times[idx_k] for idx_k, ci in enumerate(core_target_indices)}
    bound_units: List[Dict[str, Any]] = []
    for i, u in enumerate(target_units):
        if i in core_pos_map:
            st, et = core_pos_map[i]
            bound_units.append({"text": u, "start": st, "end": et, "is_punct": False})
        else:
            # 标点依附前后时间，不占独立语音时长
            prev_end = bound_units[-1]["end"] if bound_units else overall_start
            bound_units.append({"text": u, "start": prev_end, "end": prev_end, "is_punct": True})

    return bound_units


def _split_clauses_by_internal_punct(
    text: str,
    bound_units: List[Dict[str, Any]],
    rules: Dict[str, Any],
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """
    先沿句内逗号/分号拆分，并对顿号“、”按视觉宽度决定转成空格还是继续拆条。
    严格保留外边界与内部字符时间绑定，不切开配对引号/书名号。
    """
    if not bound_units:
        return []

    enum_max = float(rules.get("enum_single_line_max_hanzi", 12))
    split_puncts = set(rules.get("split_internal_punctuation", ["，", "；", ",", ";"]))

    # 按 bound_units 序列遍历，跳过配对引号内标点
    clauses: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []
    quote_depth = 0
    for u in bound_units:
        t = u["text"]
        if t in ("“", "《", "（"):
            quote_depth += 1
        elif t in ("”", "》", "）") and quote_depth > 0:
            quote_depth -= 1

        if quote_depth == 0 and t in split_puncts:
            if cur:
                clauses.append(cur)
                cur = []
            continue
        cur.append(u)
    if cur:
        clauses.append(cur)

    # 顿号处理：若该子句含“、”，先看将“、”替换为空格并剥尾标点后长度是否 <= enum_max
    result: List[Tuple[str, List[Dict[str, Any]]]] = []
    for cl in clauses:
        raw_cl = strip_trailing_punctuation("".join(u["text"] for u in cl))
        if not raw_cl:
            continue
        # 重新裁掉末尾因为剥标点去掉的单元
        trimmed_cl = _trim_units_to_text(cl, raw_cl)
        if ENUM_CHAR not in raw_cl:
            result.append((raw_cl, trimmed_cl))
            continue

        spaced = raw_cl.replace(ENUM_CHAR, " ")
        if visual_hanzi_width(spaced) <= enum_max:
            new_units = []
            for u in trimmed_cl:
                nu = dict(u)
                if nu["text"] == ENUM_CHAR:
                    nu["text"] = " "
                new_units.append(nu)
            result.append((spaced, new_units))
        else:
            # 先按顿号拆为并列项，再适度合并<=enum_max的相邻项（用空格隔开）
            parts: List[List[Dict[str, Any]]] = []
            buf: List[Dict[str, Any]] = []
            for u in trimmed_cl:
                if u["text"] == ENUM_CHAR:
                    if buf:
                        parts.append(buf)
                        buf = []
                else:
                    buf.append(u)
            if buf:
                parts.append(buf)

            # 尝试把靠后的同构并列短项适当聚拢（例如 穿着你的鞋子去旅游 / 去登山 去健身）
            merged_groups: List[List[Dict[str, Any]]] = []
            for part in parts:
                p_txt = "".join(x["text"] for x in part).strip()
                if not p_txt:
                    continue
                if merged_groups:
                    prev_txt = "".join(x["text"] for x in merged_groups[-1])
                    # 如果两者都短且前一段没有主谓长前缀（避免把“穿着你的鞋子去旅游”与“去登山”黏出14字）
                    candidate = f"{prev_txt} {p_txt}"
                    if visual_hanzi_width(candidate) <= enum_max:
                        space_t = merged_groups[-1][-1]["end"]
                        merged_groups[-1].append({"text": " ", "start": space_t, "end": space_t, "is_punct": True})
                        merged_groups[-1].extend(part)
                        continue
                merged_groups.append(list(part))

            for grp in merged_groups:
                g_txt = "".join(x["text"] for x in grp).strip()
                if g_txt:
                    result.append((g_txt, grp))

    return result


def _trim_units_to_text(units: List[Dict[str, Any]], expected_text: str) -> List[Dict[str, Any]]:
    """把 stripped 后的 expected_text 对应到截取后的 bound_units。"""
    acc = ""
    kept: List[Dict[str, Any]] = []
    target_no_lead = expected_text.lstrip()
    started = False
    for u in units:
        if not started and u["text"].isspace():
            continue
        started = True
        if len(acc) >= len(target_no_lead):
            break
        kept.append(u)
        acc += u["text"]
    return kept


def _protected_boundaries(text: str, extra_terms: Sequence[str]) -> List[Tuple[int, int]]:
    """找出文本中所有严禁从中间割裂的字符区间 [start_idx, end_idx)。"""
    spans: List[Tuple[int, int]] = []
    # 英文数字连续体
    for m in CORE_ALNUM_RE.finditer(text):
        spans.append((m.start(), m.end()))
    # 成对引号和书名号（仅当整体较短 <= 10 字时锁定不被拆散）
    for pat in (r"《[^》]{1,10}》", r"“[^”]{1,10}”"):
        for m in re.finditer(pat, text):
            spans.append((m.start(), m.end()))
    # 结构化中文合成词 / 量词 / 叠词正则保护
    for pat in LEXICAL_PROTECT_PATTERNS:
        for m in re.finditer(pat, text):
            spans.append((m.start(), m.end()))
    # 固定术语和常见连词整体
    terms = list(DEFAULT_PROTECTED_TERMS) + list(extra_terms)
    for w, _ in PRE_BREAK_WORDS + POST_BREAK_WORDS:
        if len(w) >= 2:
            terms.append(w)
    for term in terms:
        start = 0
        while True:
            pos = text.find(term, start)
            if pos < 0:
                break
            spans.append((pos, pos + len(term)))
            start = pos + len(term)
    return spans


def _unit_char_offsets(units: Sequence[Dict[str, Any]]) -> Tuple[str, List[Tuple[int, int]]]:
    text_parts: List[str] = []
    offsets: List[Tuple[int, int]] = []
    cursor = 0
    for u in units:
        t = u["text"]
        text_parts.append(t)
        offsets.append((cursor, cursor + len(t)))
        cursor += len(t)
    return "".join(text_parts), offsets


def split_units_by_real_timestamps(
    units: List[Dict[str, Any]],
    target_min: float,
    target_max: float,
    hard_max: float,
    micro_gap_threshold: float = 0.06,
    protected_terms: Sequence[str] = (),
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """
    递归按真实字时间微气口 + 中文短语边界把超长句拆分为满足 <= hard_max 的多条字幕。
    严禁按字数均匀切分时间！
    """
    text, _ = _unit_char_offsets(units)
    text_clean = text.strip()
    if not text_clean:
        return []
    if visual_hanzi_width(text_clean) <= hard_max:
        return [(text_clean, units)]

    # 从左到右找最佳第一个切分下标 k (k 是 units 的索引，左侧为 units[:k], 右侧为 units[k:])
    prot_spans = _protected_boundaries(text, protected_terms)
    unit_starts = []
    cur_pos = 0
    for u in units:
        unit_starts.append(cur_pos)
        cur_pos += len(u["text"])

    total_w = visual_hanzi_width(text_clean)
    best_k: Optional[int] = None
    best_score = -1e9
    ideal_mid = (target_min + target_max) / 2.0

    for k in range(1, len(units)):
        # 跳过纯空格或末尾标点前的无意义断点
        if units[k]["text"] in (" ", "”", "》", "）", "？", "！", "…"):
            continue
        if units[k - 1]["text"] in (" ", "“", "《", "（"):
            continue

        char_pos = unit_starts[k]
        # 不能破坏不可拆保护区间
        if any(s < char_pos < e for s, e in prot_spans):
            continue

        left_str = "".join(u["text"] for u in units[:k]).strip()
        right_str = "".join(u["text"] for u in units[k:]).strip()
        lw = visual_hanzi_width(left_str)
        rw = visual_hanzi_width(right_str)
        if lw <= 0 or lw > hard_max or rw <= 0:
            continue

        # 打分项 1：孤儿碎片严惩（避免前面 12 字后面剩 2 个字）
        score = 0.0
        if lw < 3.0:
            score -= 6.0
        if rw < 3.0:
            score -= 8.0
        elif rw < 4.5 and total_w > hard_max + 2.0:
            score -= 2.5

        # 打分项 2：真实字间物理停顿微气口 (>= 0.06s)
        prev_speech = next((u for u in reversed(units[:k]) if not u["is_punct"]), None)
        next_speech = next((u for u in units[k:] if not u["is_punct"]), None)
        if prev_speech and next_speech:
            gap = next_speech["start"] - prev_speech["end"]
            if gap >= micro_gap_threshold:
                score += 6.0 + min(4.0, gap * 10.0)

        # 打分项 3：中文语法短语边界（连词/介词前、助词后）
        for word, bonus in PRE_BREAK_WORDS:
            if right_str.startswith(word):
                score += bonus
                break
        for word, bonus in POST_BREAK_WORDS:
            if left_str.endswith(word):
                score += bonus
                break
        # 长的定语“的”字结尾弱鼓励（但短词如“好的”“你的”不可在此切）
        if left_str.endswith("的") and lw >= 6.0 and rw >= 4.0:
            score += 1.6

        # 避免行末留下悬空虚词（如“你的广告还”、“塑造一”）
        if left_str and left_str[-1] in DANGLING_LEFT_END_CHARS:
            score -= 5.0

        # 打分项 4：接近理想宽度区间与均衡度
        if target_min <= lw <= target_max:
            score += 2.2
        score -= abs(lw - ideal_mid) * 0.25

        # 尽量让右侧也有机会满足平衡（比如总长 18 字，分成 9+9 优于 13+5）
        if total_w <= hard_max * 2:
            score -= abs(lw - rw) * 0.15

        if score > best_score:
            best_score = score
            best_k = k

    if best_k is None:
        # 兜底：选最接近 target_max 且不大于 hard_max 的最大安全点（依然只落在单字/ASCII整体界限）
        for k in range(len(units) - 1, 0, -1):
            left_str = "".join(u["text"] for u in units[:k]).strip()
            if left_str and visual_hanzi_width(left_str) <= hard_max:
                best_k = k
                break
        if best_k is None:
            return [(text_clean, units)]

    left_units = [u for u in units[:best_k] if u["text"]]
    right_units = [u for u in units[best_k:] if u["text"]]
    # 去掉交界处悬空空格
    while left_units and left_units[-1]["text"] == " ":
        left_units.pop()
    while right_units and right_units[0]["text"] == " ":
        right_units.pop(0)

    left_txt = "".join(u["text"] for u in left_units).strip()
    out: List[Tuple[str, List[Dict[str, Any]]]] = []
    if left_txt:
        out.append((left_txt, left_units))
    out.extend(
        split_units_by_real_timestamps(
            right_units,
            target_min=target_min,
            target_max=target_max,
            hard_max=hard_max,
            micro_gap_threshold=micro_gap_threshold,
            protected_terms=protected_terms,
        )
    )
    return out


def _span_timing(units: Sequence[Dict[str, Any]]) -> Tuple[float, float]:
    speech_units = [u for u in units if not u["is_punct"]]
    if not speech_units:
        speech_units = list(units)
    st = min(u["start"] for u in speech_units)
    et = max(u["end"] for u in speech_units)
    if et <= st:
        et = st + 0.05
    return st, et


def _apply_global_replacements(text: str, replacements: Sequence[Sequence[str]]) -> str:
    out = text
    for pair in replacements:
        if len(pair) >= 2 and pair[0]:
            out = out.replace(pair[0], pair[1])
    return out


def _clean_cue_text_check_filler(text: str, drop_fillers: set[str]) -> bool:
    """检查整条字幕是否仅包含语气废词或空标点。"""
    stripped = re.sub(r"[，。；：、？！…,.!?;:\s“”\"'《》（）()—·\-]+", "", text)
    if not stripped:
        return True
    if stripped in drop_fillers:
        return True
    # 如 "呃 嗯" 全部是语气词
    if all(ch in drop_fillers for ch in stripped):
        return True
    return False


def process_subtitles(
    raw_cues: Sequence[Dict[str, Any]],
    char_tokens: Sequence[Dict[str, Any]],
    keep_segments: Sequence[Dict[str, Any]],
    orientation: str,
    rules: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    将映射到保留视频片段的原始字幕条，结合字级时间戳执行完整友好化处理：
    1. 应用全局与按条错字校正；
    2. 对每条原始字幕绑定真实字级时间；
    3. 沿句内逗号/分号拆条、剥离句末冗余标点、顿号转空格或分条；
    4. 对超单行上限条目按真实字时间（气口 + 语法）精准拆条；
    5. 剔除孤立语气词条、合并 <0.45s 闪屏短词，并加 0.15~0.20s 延尾（严禁跨段或碰撞后条）。
    """
    rules = deepcopy(rules or {})
    is_portrait = orientation == "portrait"
    target_min = float(rules.get("target_min_hanzi") or (6.0 if is_portrait else 8.0))
    target_max = float(rules.get("target_max_hanzi") or (12.0 if is_portrait else 14.0))
    hard_max = float(rules.get("hard_max_hanzi") or (13.0 if is_portrait else 16.0))
    min_cue_dur = float(rules.get("min_cue_duration_sec", 0.45))
    tail_pad = float(rules.get("tail_pad_sec", 0.18))
    min_gap = float(rules.get("min_cue_gap_sec", 0.02))
    drop_fillers = set(rules.get("drop_standalone_fillers", list(DEFAULT_DROP_FILLERS)))
    profile_terms, profile_replacements = _load_profile_glossary()
    protected_terms = list(dict.fromkeys(profile_terms + list(rules.get("protected_terms", []))))

    typo_cfg = rules.get("typo_fixes", {})
    global_replacements = profile_replacements + list(typo_cfg.get("global_replacements", []))
    per_id_fixes = {str(k): v for k, v in typo_cfg.items() if k != "global_replacements"}

    norm_chars = normalize_char_spans(char_tokens)
    seg_lookup = {s["id"]: s for s in keep_segments}

    # 先构建带目标时间映射的字表，供每个 raw_cue 精准匹配字符时间
    intermediate_cues: List[Dict[str, Any]] = []
    for rc in raw_cues:
        seg_id = rc["keep_id"]
        seg = seg_lookup[seg_id]
        src_s = float(rc["source_start"])
        src_e = float(rc["source_end"])
        srt_id_str = str(rc.get("srt_id", ""))

        text = str(rc["text"]).strip()
        if srt_id_str and srt_id_str in per_id_fixes:
            text = str(per_id_fixes[srt_id_str]).strip()
        text = _apply_global_replacements(text, global_replacements)

        if _clean_cue_text_check_filler(text, drop_fillers):
            continue

        # 提取对应原片区间内的字符时间（优先严格匹配中心或起止，允许 +-0.08s 对齐余量）
        local_spans = [
            c
            for c in norm_chars
            if c["end"] > src_s - 0.08 and c["start"] < src_e + 0.08
            and (c["start"] + c["end"]) * 0.5 >= src_s - 0.06
            and (c["start"] + c["end"]) * 0.5 <= src_e + 0.06
        ]
        # 确保单调裁切在 [src_s, src_e] 且换算到 target 轴（支持 speed 倍速映射）
        spd = float(seg.get("speed") or 1.0)
        target_spans: List[Dict[str, Any]] = []
        for c in local_spans:
            cs = max(src_s, min(src_e, c["start"]))
            ce = max(cs + 0.01, min(src_e, c["end"]))
            ts = seg["target_start"] + (cs - seg["source_start"]) / spd
            te = seg["target_start"] + (ce - seg["source_start"]) / spd
            target_spans.append({"ch": c["ch"], "start": ts, "end": max(ts + 0.01, te)})

        if not target_spans:
            # 若 custom cue 无字符表匹配，用整条起止兜底构造虚拟跨度（外边界依然来自 target）
            ts = seg["target_start"] + (src_s - seg["source_start"]) / spd
            te = max(ts + 0.05, seg["target_start"] + (src_e - seg["source_start"]) / spd)
            target_spans = [{"ch": ch, "start": ts, "end": te} for ch in text if is_core_token_char(ch)]

        bound_units = _match_text_to_char_spans(text, target_spans)
        if not bound_units:
            continue

        clauses = _split_clauses_by_internal_punct(text, bound_units, rules)
        for cl_txt, cl_units in clauses:
            if _clean_cue_text_check_filler(cl_txt, drop_fillers):
                continue
            pieces = split_units_by_real_timestamps(
                cl_units,
                target_min=target_min,
                target_max=target_max,
                hard_max=hard_max,
                protected_terms=protected_terms,
            )
            for piece_txt, piece_units in pieces:
                final_txt = strip_trailing_punctuation(piece_txt)
                if not final_txt or _clean_cue_text_check_filler(final_txt, drop_fillers):
                    continue
                st, et = _span_timing(piece_units)
                st = max(seg["target_start"], min(seg["target_end"] - 0.02, st))
                et = max(st + 0.02, min(seg["target_end"], et))
                intermediate_cues.append(
                    {
                        "keep_id": seg_id,
                        "srt_id": rc.get("srt_id"),
                        "text": final_txt,
                        "start": st,
                        "end": et,
                        "raw_start": st,
                        "raw_end": et,
                        "visual_width": visual_hanzi_width(final_txt),
                        "units": piece_units,
                    }
                )

    # 按时间排序并单调消重叠
    intermediate_cues.sort(key=lambda x: (x["start"], x["end"]))

    # 闪屏消除与短条合并：针对 < min_cue_dur 的条，优先在同 keep_id 且间隙 <= 0.40s 内合并相邻条
    # 若直接合并略超 hard_max（例如“当然” + “这个软件呢还有很多功能啊”），先融合再按真实字边界重拆为均衡两条
    merged_cues: List[Dict[str, Any]] = []
    i = 0
    while i < len(intermediate_cues):
        cur = dict(intermediate_cues[i])
        dur = cur["end"] - cur["start"]
        if dur < min_cue_dur:
            # 1) 优先尝试向后合并（常见句首短衬词 "好" + "谢谢大家" / "当然" + 后句）
            if i + 1 < len(intermediate_cues):
                nxt = intermediate_cues[i + 1]
                gap_fwd = nxt["start"] - cur["end"]
                cand_fwd = f"{cur['text']} {nxt['text']}".strip()
                if nxt["keep_id"] == cur["keep_id"] and gap_fwd <= 0.40:
                    if visual_hanzi_width(cand_fwd) <= hard_max:
                        cur_merged = dict(nxt)
                        cur_merged["text"] = cand_fwd
                        cur_merged["start"] = cur["start"]
                        cur_merged["raw_start"] = cur["raw_start"]
                        cur_merged["visual_width"] = visual_hanzi_width(cand_fwd)
                        sep_t = cur["end"]
                        cur_merged["units"] = (
                            list(cur.get("units", []))
                            + [{"text": " ", "start": sep_t, "end": sep_t, "is_punct": True}]
                            + list(nxt.get("units", []))
                        )
                        merged_cues.append(cur_merged)
                        i += 2
                        continue
                    elif cur.get("units") and nxt.get("units"):
                        sep_t = cur["end"]
                        comb_units = (
                            list(cur["units"])
                            + [{"text": " ", "start": sep_t, "end": sep_t, "is_punct": True}]
                            + list(nxt["units"])
                        )
                        re_pieces = split_units_by_real_timestamps(
                            comb_units,
                            target_min=target_min,
                            target_max=target_max,
                            hard_max=hard_max,
                            protected_terms=protected_terms,
                        )
                        if len(re_pieces) >= 2:
                            re_built = []
                            seg = seg_lookup[cur["keep_id"]]
                            for rp_txt, rp_units in re_pieces:
                                rp_clean = strip_trailing_punctuation(rp_txt)
                                if not rp_clean:
                                    continue
                                r_st, r_et = _span_timing(rp_units)
                                r_st = max(seg["target_start"], min(seg["target_end"] - 0.02, r_st))
                                r_et = max(r_st + 0.02, min(seg["target_end"], r_et))
                                re_built.append(
                                    {
                                        "keep_id": cur["keep_id"],
                                        "srt_id": nxt.get("srt_id"),
                                        "text": rp_clean,
                                        "start": r_st,
                                        "end": r_et,
                                        "raw_start": r_st,
                                        "raw_end": r_et,
                                        "visual_width": visual_hanzi_width(rp_clean),
                                        "units": rp_units,
                                    }
                                )
                            if re_built and all((x["end"] - x["start"]) > dur + 0.05 for x in re_built):
                                merged_cues.extend(re_built)
                                i += 2
                                continue
            # 2) 尝试向前合并到上一条同段字幕
            if merged_cues:
                prev = merged_cues[-1]
                gap_bwd = cur["start"] - prev["end"]
                cand_bwd = f"{prev['text']} {cur['text']}".strip()
                if (
                    prev["keep_id"] == cur["keep_id"]
                    and gap_bwd <= 0.40
                    and visual_hanzi_width(cand_bwd) <= hard_max
                ):
                    prev["text"] = cand_bwd
                    prev["end"] = cur["end"]
                    prev["raw_end"] = cur["raw_end"]
                    prev["visual_width"] = visual_hanzi_width(cand_bwd)
                    i += 1
                    continue
        merged_cues.append(cur)
        i += 1

    # 最后一步：加入 0.15~0.20s 阅读延尾，并在 [seg.target_start, min(seg.target_end, next.start - min_gap)] 内截断；
    # 若仍有独立实词条略短于 min_cue_dur（如 0.35s），延尾顺便将其安全拉满至 min_cue_dur（在物理不跨段/不撞后条前提下）
    final_cues: List[Dict[str, Any]] = []
    for idx_c, c in enumerate(merged_cues):
        seg = seg_lookup[c["keep_id"]]
        hard_ceiling = seg["target_end"]
        if idx_c + 1 < len(merged_cues):
            next_start = merged_cues[idx_c + 1]["start"]
            hard_ceiling = min(hard_ceiling, next_start - min_gap)

        desired_end = max(c["end"] + tail_pad, c["start"] + min_cue_dur)
        # 首要保底不短于原始语音结尾（除非与下一个相邻条重叠，则优先按 min_gap 避让，极端情况保底 0.05s）
        safe_floor = min(c["end"], hard_ceiling)
        new_end = max(safe_floor, min(desired_end, hard_ceiling))
        if new_end <= c["start"]:
            new_end = min(seg["target_end"], c["start"] + 0.05)

        start_us = int(round(c["start"] * 1_000_000))
        end_us = int(round(new_end * 1_000_000))
        if final_cues and start_us < final_cues[-1]["end_us"]:
            start_us = final_cues[-1]["end_us"]
        if end_us <= start_us:
            end_us = start_us + 20_000

        final_cues.append(
            {
                "index": len(final_cues) + 1,
                "keep_id": c["keep_id"],
                "srt_id": c.get("srt_id"),
                "text": c["text"],
                "start_sec": round(start_us / 1_000_000.0, 6),
                "end_sec": round(end_us / 1_000_000.0, 6),
                "duration_sec": round((end_us - start_us) / 1_000_000.0, 4),
                "start_us": start_us,
                "end_us": end_us,
                "duration_us": end_us - start_us,
                "visual_width": visual_hanzi_width(c["text"]),
            }
        )

    return final_cues


def build_final_plain_text(
    raw_cues: Sequence[Dict[str, Any]],
    keep_segments: Sequence[Dict[str, Any]],
    rules: Optional[Dict[str, Any]] = None,
) -> str:
    """
    依据保留分段（K01..Kn）与纠错规则汇总成片最终纯文本台词（保留完整自然标点，平滑衔接句内切点并按句末符号自然分段）。
    """
    rules = rules or {}
    drop_fillers = set(rules.get("drop_standalone_fillers", list(DEFAULT_DROP_FILLERS)))
    _, profile_replacements = _load_profile_glossary()
    typo_cfg = rules.get("typo_fixes", {})
    global_replacements = profile_replacements + list(typo_cfg.get("global_replacements", []))
    per_id_fixes = {str(k): v for k, v in typo_cfg.items() if k != "global_replacements"}

    keep_order = {str(s["id"]): idx for idx, s in enumerate(keep_segments)}
    ordered_cues = sorted(
        enumerate(raw_cues),
        key=lambda item: (keep_order.get(str(item[1].get("keep_id", "")), 10**9), float(item[1].get("source_start", 0.0)), item[0]),
    )

    merged = ""
    for _, rc in ordered_cues:
        srt_id_str = str(rc.get("srt_id", ""))
        text = str(rc.get("text", "")).strip()
        if srt_id_str and srt_id_str in per_id_fixes:
            text = str(per_id_fixes[srt_id_str]).strip()
        text = _apply_global_replacements(text, global_replacements).strip()
        if not text or _clean_cue_text_check_filler(text, drop_fillers):
            continue
        if not merged:
            merged = text
        elif merged[-1].isascii() and merged[-1].isalnum() and text[0].isascii() and text[0].isalnum():
            merged += " " + text
        else:
            merged += text

    merged = merged.strip()
    if not merged:
        return ""
    if merged[-1] in {"，", ",", "、"}:
        merged = merged[:-1] + "。"

    # 按自然句末标点分组，每 3 句一段，避免句内切点产生生硬换行
    sents = [s.strip() for s in re.split(r"(?<=[。？！…?!])", merged) if s.strip()]
    if len(sents) <= 3:
        return "".join(sents) + "\n"

    paras = ["".join(sents[i : i + 3]) for i in range(0, len(sents), 3)]
    return "\n\n".join(paras) + "\n"


def _format_srt_ms(sec: float) -> str:
    total_ms = max(0, int(round(sec * 1000.0)))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _filter_chars_by_drop_ranges(
    chars: Sequence[Dict[str, Any]],
    drop_ranges: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not drop_ranges:
        return list(chars)
    drops = [(float(r["source_start"]), float(r["source_end"])) for r in drop_ranges]
    kept: List[Dict[str, Any]] = []
    for ch in chars:
        st = float(ch.get("start_time", ch.get("start", 0.0)))
        et = float(ch.get("end_time", ch.get("end", st)))
        mid = 0.5 * (st + et)
        if any(ds <= mid <= de for ds, de in drops):
            continue
        kept.append(ch)
    return kept


def _collect_raw_cues_for_keeps(
    src_cues: Sequence[Dict[str, Any]],
    keeps: Sequence[Dict[str, Any]],
    rules: Dict[str, Any],
) -> List[Dict[str, Any]]:
    excluded_ids = {int(x) for x in rules.get("excluded_srt_ids", [])}
    overrides = rules.get("cue_overrides") or {}
    overridden_ids = {
        int(item["srt_id"])
        for items in overrides.values()
        for item in (items or [])
        if item.get("srt_id") is not None
    }
    skip_auto_ids = excluded_ids | overridden_ids

    raw_mapped: List[Dict[str, Any]] = []
    for seg in keeps:
        sid = seg["id"]
        s_start = float(seg["source_start"])
        s_end = float(seg["source_end"])
        for ov in overrides.get(sid, []):
            raw_mapped.append(
                {
                    "keep_id": sid,
                    "srt_id": ov.get("srt_id"),
                    "text": str(ov["text"]).strip(),
                    "source_start": max(s_start, float(ov["source_start"])),
                    "source_end": min(s_end, float(ov["source_end"])),
                }
            )
        for c in src_cues:
            cid = int(c["index"])
            if cid in skip_auto_ids:
                continue
            cs, ce = float(c["start"]), float(c["end"])
            mid = 0.5 * (cs + ce)
            if s_start - 0.05 <= mid <= s_end + 0.05:
                ov_s = max(s_start, cs)
                ov_e = min(s_end, ce)
                if ov_e > ov_s + 0.01:
                    raw_mapped.append(
                        {
                            "keep_id": sid,
                            "srt_id": cid,
                            "text": str(c["text"]).strip(),
                            "source_start": ov_s,
                            "source_end": ov_e,
                        }
                    )

    for cc in rules.get("custom_cues", []):
        cs, ce = float(cc["source_start"]), float(cc["source_end"])
        mid = 0.5 * (cs + ce)
        for seg in keeps:
            if seg["source_start"] <= mid <= seg["source_end"]:
                raw_mapped.append(
                    {
                        "keep_id": seg["id"],
                        "srt_id": cc.get("srt_id"),
                        "text": str(cc["text"]).strip(),
                        "source_start": max(seg["source_start"], cs),
                        "source_end": min(seg["source_end"], ce),
                    }
                )
                break

    return raw_mapped


def build_subtitles_from_timeline(
    plan_path: Path,
    srt_path: Path,
    chars_path: Path,
    out_srt: Path,
    out_txt: Path,
    rules_path: Optional[Path] = None,
    video_track: int = 0,
) -> Dict[str, Any]:
    """依据 timeline_edit.json、确认版 .srt 与逐字时间 JSON 生成友好精剪字幕 .srt 和 100% 纯净纯文本 .txt。"""
    import json
    import sys

    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    from execute_timeline_edit import load_timeline_edit
    from scan_talking_head import parse_srt

    plan = load_timeline_edit(plan_path, check_files=False)
    proj = plan["project"]
    orientation = "portrait" if int(proj["height"]) > int(proj["width"]) else "landscape"

    video_clips = sorted(
        [c for c in plan["tracks"]["video"] if c["track"] == video_track and c["asset_type"] == "video"],
        key=lambda c: c["timeline_start"],
    )
    if not video_clips:
        raise ValueError(f"timeline_edit.json 中视频轨 track={video_track} 没有视频片段")

    keeps = [
        {
            "id": c["id"],
            "source_start": float(c["source_start"]),
            "source_end": float(c["source_end"]),
            "target_start": float(c["timeline_start"]),
            "target_end": float(c["timeline_end"]),
            "speed": float(c["speed"]),
        }
        for c in video_clips
    ]

    rules: Dict[str, Any] = {}
    if rules_path is not None:
        rules = json.loads(rules_path.read_text(encoding="utf-8-sig"))
    if rules.get("orientation") in ("portrait", "landscape"):
        orientation = str(rules["orientation"])

    src_cues = parse_srt(srt_path)
    raw_chars = json.loads(chars_path.read_text(encoding="utf-8-sig"))
    filtered_chars = _filter_chars_by_drop_ranges(raw_chars, rules.get("drop_char_ranges", []))
    raw_mapped = _collect_raw_cues_for_keeps(src_cues, keeps, rules)

    final_cues = process_subtitles(
        raw_cues=raw_mapped,
        char_tokens=filtered_chars,
        keep_segments=keeps,
        orientation=orientation,
        rules=rules,
    )
    plain_txt = build_final_plain_text(raw_mapped, keeps, rules)

    srt_blocks = [
        f"{c['index']}\n{_format_srt_ms(c['start_sec'])} --> {_format_srt_ms(c['end_sec'])}\n{c['text']}"
        for c in final_cues
    ]
    out_srt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_srt.write_text("\n\n".join(srt_blocks) + ("\n" if srt_blocks else ""), encoding="utf-8")
    out_txt.write_text(plain_txt, encoding="utf-8")

    hard_max = float(rules.get("hard_max_hanzi") or (13.0 if orientation == "portrait" else 16.0))
    over_limit = [c for c in final_cues if float(c["visual_width"]) > hard_max]
    overlaps = [
        (final_cues[i - 1]["index"], final_cues[i]["index"])
        for i in range(1, len(final_cues))
        if final_cues[i]["start_us"] < final_cues[i - 1]["end_us"]
    ]
    return {
        "orientation": orientation,
        "video_clips": len(keeps),
        "subtitle_count": len(final_cues),
        "max_visual_width": max((float(c["visual_width"]) for c in final_cues), default=0.0),
        "hard_max_allowed": hard_max,
        "over_limit_count": len(over_limit),
        "overlap_count": len(overlaps),
        "out_srt": str(out_srt.resolve()),
        "out_txt": str(out_txt.resolve()),
    }


def main() -> None:
    import argparse
    import json
    import sys

    if sys.platform.startswith("win"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="按 timeline_edit.json 与逐字时间戳生成精剪友好字幕 (.srt) 及 100% 纯净纯文本 (.txt)"
    )
    parser.add_argument("--plan", type=Path, required=True, help="timeline_edit.json 文件路径")
    parser.add_argument("--srt", type=Path, required=True, help="确认版原字幕 .srt 路径")
    parser.add_argument("--chars", type=Path, required=True, help="确认版逐字时间 JSON 路径")
    parser.add_argument("--out-srt", type=Path, required=True, help="输出的精剪字幕 .srt 路径")
    parser.add_argument("--out-txt", type=Path, required=True, help="输出的纯净口播正文 .txt 路径")
    parser.add_argument("--rules", type=Path, help="可选跨切点/删字规则 JSON 路径")
    parser.add_argument("--video-track", type=int, default=0, help="读取哪条视频轨（默认 0）")
    args = parser.parse_args()

    res = build_subtitles_from_timeline(
        plan_path=args.plan,
        srt_path=args.srt,
        chars_path=args.chars,
        out_srt=args.out_srt,
        out_txt=args.out_txt,
        rules_path=args.rules,
        video_track=args.video_track,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
