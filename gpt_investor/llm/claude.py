import subprocess
import threading
import json
import re
from typing import TypeVar

import json_repair
from loguru import logger
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

_PREVIEW = 200  # chars of prompt/response shown on console; full text goes to file

_lock = threading.Lock()
_total_input_tokens: int = 0
_total_output_tokens: int = 0
_total_cache_read_tokens: int = 0

_meta_lock = threading.Lock()
_last_call_meta: dict = {}


def add_token_usage(input_tokens: int, output_tokens: int, cache_read_tokens: int) -> None:
    global _total_input_tokens, _total_output_tokens, _total_cache_read_tokens
    with _lock:
        _total_input_tokens += input_tokens
        _total_output_tokens += output_tokens
        _total_cache_read_tokens += cache_read_tokens


def get_token_totals() -> dict:
    with _lock:
        return {
            "input": _total_input_tokens,
            "output": _total_output_tokens,
            "cache_read": _total_cache_read_tokens,
        }


def get_last_call_meta() -> dict:
    with _meta_lock:
        return dict(_last_call_meta)


def _parse_stream_json(stdout: str) -> dict:
    text = ""
    tool_calls: list[dict] = []
    urls: list[str] = []
    model_usage: dict = {}

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        et = ev.get("type")
        if et == "assistant":
            for block in ev.get("message", {}).get("content", []) or []:
                if block.get("type") == "tool_use":
                    tool_calls.append({
                        "name": block.get("name", ""),
                        "input": block.get("input", {}),
                    })
        elif et == "user":
            tu_result = ev.get("tool_use_result", {}) or {}
            for r in tu_result.get("results", []) or []:
                if not isinstance(r, dict):
                    continue
                content = r.get("content")
                if isinstance(content, list):
                    for c in content:
                        u = c.get("url") if isinstance(c, dict) else None
                        if u:
                            urls.append(u)
        elif et == "result":
            text = ev.get("result", "") or text
            model_usage = ev.get("modelUsage", {}) or model_usage

    return {
        "text": text.strip(),
        "tool_calls": tool_calls,
        "urls": urls,
        "model_usage": model_usage,
    }


def call_claude(
    system_prompt: str,
    user_message: str,
    model: str = "haiku",
    tools: bool = True,
    require_tools: list[str] | None = None,
    max_retries: int = 1,
) -> str:
    global _total_input_tokens, _total_output_tokens, _total_cache_read_tokens, _last_call_meta

    base_cmd = [
        "claude", "-p", user_message,
        "--system-prompt", system_prompt,
        "--model", model,
        "--permission-mode", "bypassPermissions",
        "--no-session-persistence",
        "--output-format", "stream-json",
        "--verbose",
    ]
    if tools:
        base_cmd += ["--allowed-tools", "WebSearch,WebFetch"]

    logger.info(
        "LLM call → model={} tools={} require={} sys_chars={} user_chars={}",
        model, tools, require_tools or [], len(system_prompt), len(user_message),
    )
    logger.debug("LLM REQUEST system={!r}", system_prompt)
    logger.debug("LLM REQUEST user={!r}", user_message)

    attempt = 0
    parsed: dict = {}
    while True:
        cmd = list(base_cmd)
        if attempt > 0 and require_tools:
            stricter = (
                f"YOU MUST call one of these tools before answering: {', '.join(require_tools)}. "
                "Do not answer from memory. If the tool fails, say so explicitly.\n\n"
                + system_prompt
            )
            sp_idx = cmd.index("--system-prompt") + 1
            cmd[sp_idx] = stricter
            logger.warning("retry attempt {} with stricter prompt (require_tools={})", attempt, require_tools)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired:
            logger.error("CLI TIMEOUT after 180s  model={}; returning empty", model)
            parsed = {"text": "", "tool_calls": [], "urls": [], "model_usage": {}}
            break
        if result.returncode != 0:
            stderr_tail = (result.stderr or "").strip().splitlines()[-3:]
            logger.error("CLI non-zero exit ({})  stderr tail: {}", result.returncode, stderr_tail)
        parsed = _parse_stream_json(result.stdout)

        called = {tc["name"] for tc in parsed["tool_calls"]}
        if not require_tools or called.intersection(require_tools):
            break
        attempt += 1
        if attempt > max_retries:
            break

    call_input = sum(u.get("inputTokens", 0) for u in parsed["model_usage"].values())
    call_output = sum(u.get("outputTokens", 0) for u in parsed["model_usage"].values())
    call_cache_read = sum(u.get("cacheReadInputTokens", 0) for u in parsed["model_usage"].values())

    with _lock:
        _total_input_tokens += call_input
        _total_output_tokens += call_output
        _total_cache_read_tokens += call_cache_read

    tool_counts: dict[str, int] = {}
    for tc in parsed["tool_calls"]:
        tool_counts[tc["name"]] = tool_counts.get(tc["name"], 0) + 1
    called_set = set(tool_counts.keys())
    satisfied = (not require_tools) or bool(called_set.intersection(require_tools))

    with _meta_lock:
        _last_call_meta = {
            "tool_calls": parsed["tool_calls"],
            "urls": parsed["urls"],
            "tool_counts": tool_counts,
            "retried": attempt > 0,
            "require_tools": require_tools,
            "satisfied": satisfied,
        }

    tool_summary = " ".join(f"{n}x{c}" for n, c in tool_counts.items()) or "none"
    text = parsed["text"]
    preview = (text[:_PREVIEW] + "…") if len(text) > _PREVIEW else text
    logger.info(
        "LLM resp ← {} chars  tokens(in={:,} out={:,} cache={:,})  cum(in={:,} out={:,} cache={:,})",
        len(text), call_input, call_output, call_cache_read,
        _total_input_tokens, _total_output_tokens, _total_cache_read_tokens,
    )
    logger.info(
        "tools: {}  urls={}  retried={}  satisfied={}",
        tool_summary, len(parsed["urls"]), attempt > 0, satisfied,
    )
    logger.debug("LLM RESPONSE preview={!r}", preview)
    logger.debug("LLM RESPONSE full={!r}", text)

    if require_tools and not satisfied:
        logger.warning("required tools not invoked after {} retries: {}", attempt, require_tools)

    return text


_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)


def _extract_json_blob(text: str) -> str:
    """Pull the most likely JSON payload out of an LLM response.

    Prefers fenced ```json blocks, falls back to the first {...} or [...]
    region, finally returns the whole string for json_repair to attempt.
    """
    if not text:
        return ""
    m = _FENCED_JSON_RE.search(text)
    if m:
        return m.group(1)
    start = min(
        (i for i in (text.find("{"), text.find("[")) if i != -1),
        default=-1,
    )
    if start == -1:
        return text.strip()
    end = max(text.rfind("}"), text.rfind("]"))
    if end > start:
        return text[start : end + 1]
    return text.strip()


def call_claude_structured(
    schema: type[T],
    system_prompt: str,
    user_message: str,
    model: str = "haiku",
    tools: bool = False,
    require_tools: list[str] | None = None,
    max_schema_retries: int = 1,
) -> T | None:
    """Call Claude and parse the response into `schema` (a Pydantic model).

    Appends the schema's JSON Schema to the system prompt so the model knows
    the shape expected. On parse/validation failure, retries up to
    `max_schema_retries` times with the validator error fed back in.
    Returns `None` if every attempt fails — caller decides the fallback.
    """
    schema_json = json.dumps(schema.model_json_schema(), indent=2)
    base_sys = (
        system_prompt.rstrip()
        + "\n\nYou MUST respond with a single JSON object matching this schema "
        "(no prose, no fences, no extra fields):\n"
        + schema_json
    )
    sys_now = base_sys
    last_err: str | None = None

    for attempt in range(max_schema_retries + 1):
        text = call_claude(
            sys_now,
            user_message,
            model=model,
            tools=tools,
            require_tools=require_tools,
        )
        blob = _extract_json_blob(text)
        try:
            obj = json_repair.loads(blob) if blob else None
        except Exception as e:
            last_err = f"json parse failed: {e}"
            obj = None

        if obj is not None:
            try:
                return schema.model_validate(obj)
            except ValidationError as e:
                last_err = f"schema validation failed: {e.errors(include_url=False)}"

        if attempt >= max_schema_retries:
            break
        sys_now = (
            base_sys
            + f"\n\nPrevious response was rejected: {last_err}. "
            "Return ONLY the JSON object — no markdown, no commentary."
        )
        logger.warning("[structured] {} attempt {} failed: {}", schema.__name__, attempt + 1, last_err)

    logger.error(
        "[structured] {} gave up after {} attempts; last_err={}",
        schema.__name__, max_schema_retries + 1, last_err,
    )
    return None
