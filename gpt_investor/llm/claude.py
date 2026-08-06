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


def add_token_usage(input_tokens: int, output_tokens: int, cache_read_tokens: int) -> None:
    """Add a call's token counts to the process-wide running totals.

    Thread-safe: guarded by the module lock so parallel ticker calls don't race.

    Parameters
    ----------
    input_tokens : int
        Prompt tokens billed for the call.
    output_tokens : int
        Completion tokens billed for the call.
    cache_read_tokens : int
        Tokens served from the prompt cache.
    """
    global _total_input_tokens, _total_output_tokens, _total_cache_read_tokens
    with _lock:
        _total_input_tokens += input_tokens
        _total_output_tokens += output_tokens
        _total_cache_read_tokens += cache_read_tokens


def get_token_totals() -> dict:
    """Return the cumulative token counts for this process.

    Returns
    -------
    dict
        Keys ``input``, ``output``, ``cache_read`` with their running totals.
    """
    with _lock:
        return {
            "input": _total_input_tokens,
            "output": _total_output_tokens,
            "cache_read": _total_cache_read_tokens,
        }


# --- cooperative cancel: kill in-flight CLI subprocesses ------------------
# Every LLM call is a `claude` subprocess run in a worker thread; cancel can't
# interrupt one from the async side. We register each live process so a user
# cancel can terminate it immediately instead of waiting out its 180s timeout.
_procs_lock = threading.Lock()
_active_procs: set = set()
_cancelled = threading.Event()


def cancel_inflight() -> None:
    """Signal cancel and terminate every running Claude CLI subprocess.

    Called from the UI's cancel handler. Sets a flag so no *new* call starts,
    and terminates in-flight ones so a run stops within a second or two rather
    than after the current call's 180s timeout.
    """
    _cancelled.set()
    with _procs_lock:
        procs = list(_active_procs)
    for p in procs:
        try:
            p.terminate()
        except Exception:
            pass


def reset_cancel() -> None:
    """Clear the cancel flag at the start of a fresh run."""
    _cancelled.clear()


def _run_cli(cmd: list[str], timeout: int) -> tuple[int, str, str]:
    """Run the CLI as a *killable* subprocess, registered for cancel.

    Parameters
    ----------
    cmd : list[str]
        Full command to execute.
    timeout : int
        Seconds before the process is killed and TimeoutExpired raised.

    Returns
    -------
    tuple[int, str, str]
        (returncode, stdout, stderr).
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    with _procs_lock:
        _active_procs.add(proc)
    try:
        out, err = proc.communicate(timeout=timeout)
        return proc.returncode, out, err
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raise
    finally:
        with _procs_lock:
            _active_procs.discard(proc)


def _parse_stream_json(stdout: str) -> dict:
    """Parse the CLI's stream-json NDJSON into text, tool calls, URLs, usage.

    Notes
    -----
    ``stream-json`` requires the CLI's ``--verbose`` flag or it errors out.
    ``tool_use_result.results`` is heterogeneous: dict entries carry a
    ``content`` list of ``{title, url}``, but a trailing entry is the model's
    prose summary as a bare string — hence the ``isinstance(r, dict)`` and
    ``isinstance(c, dict)`` guards before ``.get()``.

    Parameters
    ----------
    stdout : str
        Raw stdout from the ``claude`` subprocess, one JSON event per line.

    Returns
    -------
    dict
        Keys ``text`` (final result string), ``tool_calls`` (list of
        ``{name, input}``), ``urls`` (search-result URLs), ``model_usage``
        (per-model token usage dict).
    """
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
    model: str = "sonnet",
    tools: bool = True,
    require_tools: list[str] | None = None,
    max_retries: int = 1,
) -> str:
    """Invoke the Claude CLI via subprocess and return the final text.

    Billing goes through the user's Claude Code subscription, not the SDK.
    When ``require_tools`` is set and the model answered without calling any of
    them, retries up to ``max_retries`` times with a stricter system prompt.
    A 180s timeout or non-zero exit yields empty text rather than raising.

    Parameters
    ----------
    system_prompt : str
        System prompt passed via ``--system-prompt``.
    user_message : str
        User prompt passed via ``-p``.
    model : str, optional
        CLI model alias, default ``"sonnet"``.
    tools : bool, optional
        Allow WebSearch/WebFetch, default ``True``.
    require_tools : list[str] | None, optional
        Tool names that must be invoked; triggers stricter-prompt retries if
        absent. Default ``None``.
    max_retries : int, optional
        Max stricter-prompt retries when ``require_tools`` is unsatisfied,
        default ``1``.

    Returns
    -------
    str
        Final response text, or ``""`` on timeout / failure.
    """
    global _total_input_tokens, _total_output_tokens, _total_cache_read_tokens

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

        if _cancelled.is_set():
            logger.info("call_claude skipped — cancel in progress  model={}", model)
            parsed = {"text": "", "tool_calls": [], "urls": [], "model_usage": {}}
            break
        try:
            returncode, stdout, stderr = _run_cli(cmd, 180)
        except subprocess.TimeoutExpired:
            logger.error("CLI TIMEOUT after 180s  model={}; returning empty", model)
            parsed = {"text": "", "tool_calls": [], "urls": [], "model_usage": {}}
            break
        if _cancelled.is_set():
            # We terminated this process via cancel — don't treat as an error.
            logger.info("CLI terminated by cancel  model={}", model)
            parsed = {"text": "", "tool_calls": [], "urls": [], "model_usage": {}}
            break
        if returncode != 0:
            stderr_tail = (stderr or "").strip().splitlines()[-3:]
            logger.error("CLI non-zero exit ({})  stderr tail: {}", returncode, stderr_tail)
        parsed = _parse_stream_json(stdout)

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

    Parameters
    ----------
    text : str
        Raw model response, possibly wrapped in prose or code fences.

    Returns
    -------
    str
        The extracted JSON substring, or the trimmed original when no
        delimiters are found.
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
    model: str = "sonnet",
    tools: bool = False,
    require_tools: list[str] | None = None,
    max_schema_retries: int = 1,
) -> T | None:
    """Call Claude and parse the response into `schema` (a Pydantic model).

    Appends the schema's JSON Schema to the system prompt so the model knows
    the shape expected. On parse/validation failure, retries up to
    `max_schema_retries` times with the validator error fed back in.

    Parameters
    ----------
    schema : type[T]
        Pydantic model class to validate the response against.
    system_prompt : str
        System prompt; the JSON Schema is appended to it.
    user_message : str
        User prompt.
    model : str, optional
        CLI model alias, default ``"sonnet"``.
    tools : bool, optional
        Allow WebSearch/WebFetch, default ``False``.
    require_tools : list[str] | None, optional
        Tool names that must be invoked, forwarded to ``call_claude``.
        Default ``None``.
    max_schema_retries : int, optional
        Max re-prompts on parse/validation failure, default ``1``.

    Returns
    -------
    T | None
        Validated model instance, or ``None`` if every attempt fails — the
        caller decides the fallback.
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
