"""Typed agent events — discriminated union of frozen dataclasses."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

JSONValue = str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]
ToolArgs = dict[str, JSONValue]


class EventType(StrEnum):
    THOUGHT = "thought"
    TEXT_DELTA = "text_delta"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    COMPLETE = "complete"
    TURN_START = "turn_start"
    ERROR = "error"
    LLM_USAGE = "llm_usage"


class TerminateReason(StrEnum):
    GOAL = "GOAL"
    MAX_TURNS = "MAX_TURNS"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class ThoughtEvent:
    delta: str = ""
    type: EventType = field(default=EventType.THOUGHT, init=False)


@dataclass(frozen=True, slots=True)
class TextDeltaEvent:
    delta: str = ""
    type: EventType = field(default=EventType.TEXT_DELTA, init=False)


@dataclass(frozen=True, slots=True)
class ToolCallEvent:
    id: str = ""
    name: str = ""
    args: ToolArgs = field(default_factory=dict)
    type: EventType = field(default=EventType.TOOL_CALL, init=False)


@dataclass(frozen=True, slots=True)
class ToolResultEvent:
    tool_call_id: str = ""
    output: str = ""
    error: str = ""
    metadata: dict[str, float] = field(default_factory=dict)
    content_blocks: tuple[dict[str, Any], ...] = ()
    type: EventType = field(default=EventType.TOOL_RESULT, init=False)


@dataclass(frozen=True, slots=True)
class CompleteEvent:
    success: bool = False
    reason: TerminateReason = TerminateReason.ERROR
    message: str = ""
    type: EventType = field(default=EventType.COMPLETE, init=False)


@dataclass(frozen=True, slots=True)
class TurnStartEvent:
    turn: int = 0
    type: EventType = field(default=EventType.TURN_START, init=False)


@dataclass(frozen=True, slots=True)
class ErrorEvent:
    message: str = ""
    type: EventType = field(default=EventType.ERROR, init=False)


@dataclass(frozen=True, slots=True)
class LLMUsageEvent:
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    type: EventType = field(default=EventType.LLM_USAGE, init=False)


AgentEvent = (
    ThoughtEvent
    | TextDeltaEvent
    | ToolCallEvent
    | ToolResultEvent
    | CompleteEvent
    | TurnStartEvent
    | ErrorEvent
    | LLMUsageEvent
)


def event_to_dict(event: AgentEvent) -> dict[str, Any]:
    d = dataclasses.asdict(event)
    event_type = d.pop("type")
    return {"type": event_type, "data": d}
