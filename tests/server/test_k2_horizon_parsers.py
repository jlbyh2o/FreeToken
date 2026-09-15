"""K2-Horizon reasoning parser (three effort-selected think tags) and tool-call
detector (key/value XML) -- one-shot and streaming."""

from __future__ import annotations

import json

import pytest

from freetoken.server.function_call_parser import (
    Function,
    FunctionCallParser,
    K2HorizonDetector,
    Tool,
)
from freetoken.server.reasoning_parser import K2HorizonReasoningParser, ReasoningParser

CLOSERS = ("</ifm|think>", "</ifm|think_fast>", "</ifm|think_faster>")


def _stream(parser, text: str, step: int) -> tuple[str, str]:
    reasoning, normal = "", ""
    for i in range(0, len(text), step):
        r = parser.parse_streaming_increment(text[i : i + step])
        reasoning += r.reasoning_text
        normal += r.normal_text
    r = parser.flush()
    return reasoning + r.reasoning_text, normal + r.normal_text


# ---------------------------------------------------------------------------
# Reasoning parser
# ---------------------------------------------------------------------------
def test_reasoning_registered():
    assert ReasoningParser.ReasoningParserEnum["k2_horizon"] is K2HorizonReasoningParser


@pytest.mark.parametrize("closer", CLOSERS)
def test_reasoning_every_effort_level(closer):
    # The template pre-opens the tag, so the model emits only the closer -- and which
    # closer depends on the request's reasoning_effort.
    r = K2HorizonReasoningParser().detect_and_parse(f"weighing it up{closer}The answer is 4.")
    assert r.reasoning_text == "weighing it up"
    assert r.normal_text == "The answer is 4."


def test_reasoning_empty_block():
    r = K2HorizonReasoningParser().detect_and_parse("</ifm|think_faster>4")
    assert r.reasoning_text == ""
    assert r.normal_text == "4"


def test_reasoning_truncated_without_closer():
    r = K2HorizonReasoningParser().detect_and_parse("still thinking")
    assert r.reasoning_text == "still thinking"
    assert r.normal_text == ""


def test_reasoning_tool_block_ends_reasoning():
    # A turn that skips the closer and runs into a tool block ends reasoning there,
    # and the block is preserved for the tool-call parser.
    text = "need a tool<ifm|tool_calls>\n<ifm|tool_call>ping\n</ifm|tool_call>\n</ifm|tool_calls>"
    r = K2HorizonReasoningParser().detect_and_parse(text)
    assert r.reasoning_text == "need a tool"
    assert r.normal_text.startswith("<ifm|tool_calls>")


@pytest.mark.parametrize("closer", CLOSERS)
@pytest.mark.parametrize("step", [1, 3, 7])
def test_reasoning_streaming_matches_one_shot(closer, step):
    # A step of 1 splits every marker; the parser must hold back partials of all
    # three closers, not just the one it eventually binds to.
    text = f"chain of thought{closer}final answer"
    reasoning, normal = _stream(K2HorizonReasoningParser(), text, step)
    one = K2HorizonReasoningParser().detect_and_parse(text)
    assert reasoning.strip() == one.reasoning_text.strip() == "chain of thought"
    assert normal.strip() == one.normal_text.strip() == "final answer"


def test_reasoning_closer_lookalike_is_not_split_early():
    # </ifm|think_fast> must not be mistaken for </ifm|think> plus stray text.
    r = K2HorizonReasoningParser().detect_and_parse("a</ifm|think_fast>b")
    assert (r.reasoning_text, r.normal_text) == ("a", "b")


# ---------------------------------------------------------------------------
# Tool-call detector
# ---------------------------------------------------------------------------
def _tools() -> list[Tool]:
    return [
        Tool(
            type="function",
            function=Function(
                name="get_weather",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "days": {"type": "integer"},
                        "opts": {"type": "object"},
                    },
                },
            ),
        ),
        Tool(type="function", function=Function(name="ping", parameters={"type": "object", "properties": {}})),
    ]


_XML = (
    "Let me check.<ifm|tool_calls>\n"
    "<ifm|tool_call>get_weather\n"
    "<ifm|arg_key>city</ifm|arg_key>\n<ifm|arg_value>Sheffield</ifm|arg_value>\n"
    "<ifm|arg_key>days</ifm|arg_key>\n<ifm|arg_value>3</ifm|arg_value>\n"
    "<ifm|arg_key>opts</ifm|arg_key>\n<ifm|arg_value>{\"units\": \"c\"}</ifm|arg_value>\n"
    "</ifm|tool_call>\n"
    "<ifm|tool_call>ping\n</ifm|tool_call>\n"
    "</ifm|tool_calls>"
)


def test_detector_registered():
    assert FunctionCallParser.ToolCallParserEnum["k2_horizon"] is K2HorizonDetector


def test_detector_one_shot_types_from_schema():
    result = K2HorizonDetector().detect_and_parse(_XML, _tools())
    assert result.normal_text == "Let me check."
    assert [c.name for c in result.calls] == ["get_weather", "ping"]
    # the wire carries every value as text; the schema decides the JSON type
    assert json.loads(result.calls[0].parameters) == {
        "city": "Sheffield",
        "days": 3,
        "opts": {"units": "c"},
    }
    assert json.loads(result.calls[1].parameters) == {}


def test_detector_skips_xml_typed_arg_type():
    text = (
        "<ifm|tool_calls>\n<ifm|tool_call>get_weather\n"
        "<ifm|arg_key>city</ifm|arg_key>\n<ifm|arg_type>string</ifm|arg_type>\n"
        "<ifm|arg_value>Leeds</ifm|arg_value>\n</ifm|tool_call>\n</ifm|tool_calls>"
    )
    result = K2HorizonDetector().detect_and_parse(text, _tools())
    assert json.loads(result.calls[0].parameters) == {"city": "Leeds"}


def test_detector_no_tool_call_is_all_content():
    result = K2HorizonDetector().detect_and_parse("just prose", _tools())
    assert result.normal_text == "just prose"
    assert result.calls == []


@pytest.mark.parametrize("step", [1, 5, 23])
def test_detector_streaming_matches_one_shot(step):
    parser = FunctionCallParser(_tools(), "k2_horizon")
    normal, names, args = "", {}, {}
    for i in range(0, len(_XML), step):
        text, calls = parser.parse_stream_chunk(_XML[i : i + step])
        normal += text
        for call in calls:
            if call.name:
                names[call.tool_index] = call.name
            if call.parameters:
                args[call.tool_index] = args.get(call.tool_index, "") + call.parameters

    one = K2HorizonDetector().detect_and_parse(_XML, _tools())
    assert normal == one.normal_text
    assert [names[i] for i in sorted(names)] == [c.name for c in one.calls]
    for call in one.calls:
        assert json.loads(args[call.tool_index]) == json.loads(call.parameters)
