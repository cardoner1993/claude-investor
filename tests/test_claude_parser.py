"""Parser unit tests for gpt_investor.llm.claude._parse_stream_json.

These tests do NOT spawn the Claude CLI. They feed the parser real NDJSON
captured from a live `claude -p --output-format stream-json --verbose` run
and validate the audit fields the rest of the app relies on (tool_calls,
URLs, model_usage, final text).
"""

from gpt_investor.llm.claude import _parse_stream_json


# Captured from a real probe (Fed funds rate WebSearch). The trailing
# {"type":"result", ...} event was truncated in the original head(1) probe
# but is what the CLI emits at end-of-turn; we include a minimal one so the
# parser sees the full lifecycle.
PROBE_NDJSON = """\
{"type":"system","subtype":"init","session_id":"abc"}
{"type":"assistant","message":{"content":[{"type":"thinking","thinking":"plan"}]}}
{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t1","name":"WebSearch","input":{"query":"Fed funds rate May 2026"}}]}}
{"type":"user","message":{"content":[{"tool_use_id":"t1","type":"tool_result","content":"..."}]},"tool_use_result":{"results":[{"content":[{"title":"H.15","url":"https://www.federalreserve.gov/releases/h15/"},{"title":"FOMC","url":"https://www.federalreserve.gov/newsevents/pressreleases/monetary20260429a.htm"}]}]}}
{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t2","name":"WebFetch","input":{"url":"https://www.federalreserve.gov/releases/h15/"}}]}}
{"type":"assistant","message":{"content":[{"type":"text","text":"answer"}]}}
{"type":"result","subtype":"success","result":"Fed funds: 3.5-3.75%\\nSource: https://www.federalreserve.gov/releases/h15/","modelUsage":{"haiku":{"inputTokens":120,"outputTokens":40,"cacheReadInputTokens":55000}}}
"""


def test_parse_extracts_final_text():
    p = _parse_stream_json(PROBE_NDJSON)
    assert p["text"].startswith("Fed funds: 3.5-3.75%")


def test_parse_collects_all_tool_uses_in_order():
    p = _parse_stream_json(PROBE_NDJSON)
    names = [tc["name"] for tc in p["tool_calls"]]
    assert names == ["WebSearch", "WebFetch"]
    assert p["tool_calls"][0]["input"] == {"query": "Fed funds rate May 2026"}


def test_parse_extracts_search_result_urls():
    p = _parse_stream_json(PROBE_NDJSON)
    assert len(p["urls"]) == 2
    assert all("federalreserve.gov" in u for u in p["urls"])


def test_parse_extracts_model_usage():
    p = _parse_stream_json(PROBE_NDJSON)
    usage = p["model_usage"]["haiku"]
    assert usage["inputTokens"] == 120
    assert usage["outputTokens"] == 40
    assert usage["cacheReadInputTokens"] == 55000


def test_parse_ignores_malformed_lines():
    bad = "\nnot json\n{\"type\":\"unknown\"}\n"
    p = _parse_stream_json(bad)
    assert p["text"] == ""
    assert p["tool_calls"] == []
    assert p["urls"] == []
    assert p["model_usage"] == {}


def test_parse_handles_empty_input():
    p = _parse_stream_json("")
    assert p == {"text": "", "tool_calls": [], "urls": [], "model_usage": {}}


def test_require_tools_satisfied_when_tool_invoked():
    p = _parse_stream_json(PROBE_NDJSON)
    called = {tc["name"] for tc in p["tool_calls"]}
    assert bool(called.intersection(["WebSearch"])) is True


def test_parse_handles_mixed_type_results_array():
    # Real CLI output: tool_use_result.results is a heterogeneous list -
    # first a dict with structured `content`, then a trailing string with
    # the model's prose summary. Parser must skip the string, not crash.
    line = (
        '{"type":"user","message":{"content":[]},'
        '"tool_use_result":{"results":['
        '{"content":[{"title":"H.15","url":"https://www.federalreserve.gov/releases/h15/"}]},'
        '"Based on the search results, here is the summary..."'
        ']}}'
    )
    p = _parse_stream_json(line)
    assert p["urls"] == ["https://www.federalreserve.gov/releases/h15/"]


def test_require_tools_unsatisfied_when_no_tool_use():
    # Only a result event, no tool_use in between -> model hallucinated.
    only_result = '{"type":"result","result":"made up answer","modelUsage":{}}'
    p = _parse_stream_json(only_result)
    called = {tc["name"] for tc in p["tool_calls"]}
    assert bool(called.intersection(["WebSearch"])) is False
    assert p["text"] == "made up answer"
