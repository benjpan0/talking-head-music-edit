"""阿杜导演（ADu Director）MEP 1 底层协议工具：帧率映射、请求校验、协作目录心跳检测与原子提交。"""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

ALLOWED_MEP_FPS = (23.976, 24, 25, 29.97, 30, 48, 50, 59.94, 60, 120)
MEP_STALE_MS = 20_000
MEP_MAX_OPS_PER_REQUEST = 100
DEFAULT_POLL_SEC = 0.5
DEFAULT_TIMEOUT_SEC = 180.0


def normalize_mep_fps(raw_fps: float) -> Any:
    """将帧率匹配至 MEP 1 白名单枚举值。"""
    for cand in ALLOWED_MEP_FPS:
        if abs(float(raw_fps) - float(cand)) < 0.05:
            return cand
    rounded = int(round(float(raw_fps)))
    if rounded in ALLOWED_MEP_FPS:
        return rounded
    return "auto"


def validate_mep_request_envelope(req: Dict[str, Any]) -> None:
    """按 MEP 1 规范在提交前做本地预校验。"""
    if req.get("mep") != 1:
        raise ValueError("MEP 请求缺少固定字段 '\"mep\": 1'")
    key = req.get("key")
    if not isinstance(key, str) or not (1 <= len(key) <= 64):
        raise ValueError("MEP 请求 key 必须为 1~64 位字符串")
    who = req.get("who")
    if not isinstance(who, str) or not (1 <= len(who) <= 24) or who != who.strip():
        raise ValueError("MEP 请求 who 必须为 1~24 位且首尾不含空格")

    kinds = [k for k in ("ops", "roughcut", "template", "look", "voice", "search", "fetch") if k in req]
    if len(kinds) != 1:
        raise ValueError(f"MEP 请求必须且仅包含一种载荷，当前检测到：{kinds}")

    kind = kinds[0]
    if kind in {"ops", "roughcut", "template"}:
        base = str(req.get("base") or "")
        if not re.fullmatch(r"[a-f0-9]{16,64}", base):
            raise ValueError(f"MEP 请求 {kind} 需要合法 16~64 位小写十六进制 base，当前为：{base!r}")

    if kind == "ops":
        ops = req["ops"]
        if not isinstance(ops, list) or not (1 <= len(ops) <= MEP_MAX_OPS_PER_REQUEST):
            raise ValueError(
                f"MEP 单批 ops 数量必须在 1~{MEP_MAX_OPS_PER_REQUEST} 之间，"
                f"当前为 {len(ops) if isinstance(ops, list) else '非列表'}"
            )


def check_mep_workspace(mep_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """检查阿杜导演协作目录在线心跳（<20s）及当前 project.json 状态。"""
    if not mep_dir.is_dir():
        raise FileNotFoundError(f"阿杜导演协作目录不存在: {mep_dir}")

    status_path = mep_dir / "status.json"
    project_path = mep_dir / "project.json"
    if not status_path.is_file() or not project_path.is_file():
        raise FileNotFoundError(f"目标目录缺少 status.json 或 project.json: {mep_dir}")

    status = json.loads(status_path.read_text(encoding="utf-8-sig"))
    now_ms = int(time.time() * 1000)
    alive_ms = int(status.get("alive") or 0)
    if now_ms - alive_ms >= MEP_STALE_MS:
        raise RuntimeError(
            f"阿杜导演宿主离线或该工程未打开（心跳距今超过 {MEP_STALE_MS // 1000} 秒），请打开工程后重试。"
        )
    if status.get("state") != "open":
        raise RuntimeError(
            f"阿杜导演当前协作单据状态为 {status.get('state')!r}（非 'open'），请重新点击「复制命令」获取新口令。"
        )

    project = json.loads(project_path.read_text(encoding="utf-8-sig"))
    return status, project


def submit_mep_request(
    mep_dir: Path,
    key: str,
    payload: Dict[str, Any],
    who: str = "talking-head-music-edit",
    req_id: Optional[str] = None,
    poll_sec: float = DEFAULT_POLL_SEC,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
) -> Dict[str, Any]:
    """先写 requests/<id>.tmp 再原子改名为 .json，并轮询 results/<id>.json 回执。"""
    rid = req_id or f"r-{uuid.uuid4().hex[:12]}"
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", rid):
        raise ValueError(f"非法 MEP 请求 id: {rid}")

    envelope: Dict[str, Any] = {
        "mep": 1,
        "key": key,
        "who": who,
        **payload,
    }
    validate_mep_request_envelope(envelope)

    req_dir = mep_dir / "requests"
    res_dir = mep_dir / "results"
    req_dir.mkdir(parents=True, exist_ok=True)

    res_file = res_dir / f"{rid}.json"
    if res_file.is_file():
        raise FileExistsError(f"回执文件已存在（防重提交）：{res_file}")

    tmp_file = req_dir / f"{rid}.tmp"
    final_file = req_dir / f"{rid}.json"
    encoded = json.dumps(envelope, ensure_ascii=False, indent=2).encode("utf-8")
    if len(encoded) > 2 * 1024 * 1024:
        raise ValueError(f"MEP 请求体积超过 2 MiB 上限：{len(encoded)} bytes")

    tmp_file.write_bytes(encoded)
    tmp_file.replace(final_file)

    start_t = time.time()
    last_working_msg = ""
    while True:
        if res_file.is_file():
            res_data = json.loads(res_file.read_text(encoding="utf-8-sig"))
            if not res_data.get("ok"):
                err = res_data.get("error") or {}
                raise RuntimeError(
                    f"阿杜导演执行失败 [{err.get('code')} | retry={err.get('retry')}]: {err.get('message')}"
                )
            return res_data

        status_path = mep_dir / "status.json"
        if status_path.is_file():
            try:
                st = json.loads(status_path.read_text(encoding="utf-8-sig"))
                refused = st.get("refused") or {}
                if refused.get("id") == rid:
                    raise RuntimeError(f"MEP 请求被拒绝 [{refused.get('code')}]: {refused.get('message')}")
                working = st.get("working") or {}
                if working.get("id") == rid and working.get("text") and working.get("text") != last_working_msg:
                    last_working_msg = str(working["text"])
                    print(f"[阿杜导演] {last_working_msg}")
            except RuntimeError:
                raise
            except Exception:
                pass

        if time.time() - start_t >= timeout_sec:
            raise TimeoutError(f"等待阿杜导演回执超时（{timeout_sec:.0f}s）：{res_file}")
        time.sleep(poll_sec)
