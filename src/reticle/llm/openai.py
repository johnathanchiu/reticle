"""OpenAI API service with streaming tool support using the Responses API."""

import logging
import os
import uuid
from collections.abc import AsyncGenerator

import openai

from .base import (
    BaseLLMService,
    Message,
    StreamChunk,
    ThinkingLevel,
    ToolCall,
    ToolCallFunction,
)

logger = logging.getLogger(__name__)

REASONING_MODELS = ("gpt-5", "o1", "o3", "o4")

OPENAI_REASONING_EFFORT = {
    "low": "low",
    "med": "medium",
    "high": "high",
}


# ---- Schema normalization ----


def _normalize_json_schema(schema: dict) -> dict:
    """Normalize JSON schema for OpenAI compatibility.

    Converts Pydantic tuple schemas (prefixItems) to array schemas (items)
    since OpenAI doesn't support prefixItems.
    """
    if not isinstance(schema, dict):
        return schema

    result = {}
    for key, value in schema.items():
        if key == "prefixItems":
            # Convert tuple schema to array — use first item's type
            # (uniform-type tuples like tuple[float, float])
            if value and isinstance(value, list):
                result["items"] = _normalize_json_schema(value[0])
            continue
        elif key == "$defs":
            result[key] = {k: _normalize_json_schema(v) for k, v in value.items()}
        elif isinstance(value, dict):
            result[key] = _normalize_json_schema(value)
        elif isinstance(value, list):
            result[key] = [
                _normalize_json_schema(item) if isinstance(item, dict) else item for item in value
            ]
        else:
            result[key] = value

    return result


# ---- Content block conversion ----
#
# Our internal message format uses Anthropic-style content blocks:
#   {"type": "text", "text": "..."}
#   {"type": "image", "source": {"type": "base64", "media_type": "...", "data": "..."}}
#
# OpenAI Responses API uses:
#   {"type": "input_text", "text": "..."}   (user messages)
#   {"type": "output_text", "text": "..."}  (assistant messages)
#   {"type": "input_image", "image_url": "data:...;base64,..."}


def _convert_image_source(source: dict) -> dict | None:
    """Convert an internal image source to OpenAI input_image format."""
    if source.get("type") == "base64":
        return {
            "type": "input_image",
            "image_url": f"data:{source['media_type']};base64,{source['data']}",
        }
    if source.get("type") == "url":
        return {"type": "input_image", "image_url": source["url"]}
    return None


def _content_to_user_parts(content: str | list) -> list[dict]:
    """Convert user message content (string or multimodal blocks) to Responses API parts."""
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]

    if not isinstance(content, list):
        return [{"type": "input_text", "text": str(content)}]

    parts: list[dict] = []
    for block in content:
        if not isinstance(block, dict):
            parts.append({"type": "input_text", "text": str(block)})
        elif block.get("type") == "text":
            parts.append({"type": "input_text", "text": block["text"]})
        elif block.get("type") == "image":
            img = _convert_image_source(block.get("source", {}))
            if img:
                parts.append(img)
    return parts or [{"type": "input_text", "text": ""}]


def _split_tool_content(content: str | list) -> tuple[str, list[dict]]:
    """Split tool result content into a text string and any image parts.

    OpenAI's function_call_output only accepts a string, so we extract
    images separately to inject as a follow-up user message.
    """
    if isinstance(content, str):
        return content, []

    if not isinstance(content, list):
        return str(content), []

    text_parts: list[str] = []
    image_parts: list[dict] = []
    for block in content:
        if isinstance(block, str):
            text_parts.append(block)
        elif not isinstance(block, dict):
            continue
        elif block.get("type") == "text":
            text_parts.append(block["text"])
        elif block.get("type") == "image":
            img = _convert_image_source(block.get("source", {}))
            if img:
                image_parts.append(img)

    return "\n".join(text_parts) if text_parts else "OK", image_parts


def sanitize_tool_messages(messages: list[Message]) -> list[Message]:
    """Ensure tool call IDs are present and consistent.

    Shared by OpenAIService and QwenService.
    """
    sanitized: list[Message] = []
    pending_ids: list[str] = []

    for original_msg in messages or []:
        msg = Message(role=original_msg["role"], content=original_msg["content"])
        for key in (
            "tool_calls",
            "thought_summary",
            "thinking_signature",
            "reasoning_items",
            "tool_call_id",
            "name",
        ):
            if key in original_msg:
                msg[key] = original_msg[key]  # type: ignore[literal-required]

        role = msg["role"]

        if role == "assistant" and "tool_calls" in msg:
            tool_calls = msg["tool_calls"]
            normalized_tool_calls: list[ToolCall] = []
            for tc in tool_calls:
                tc_id = tc["id"] if tc["id"] else f"call_{uuid.uuid4().hex}"
                normalized_tc = ToolCall(
                    id=tc_id,
                    function=ToolCallFunction(
                        name=tc["function"]["name"],
                        arguments=tc["function"]["arguments"],
                    ),
                )
                if "type" in tc:
                    normalized_tc["type"] = tc["type"]
                if "thought_signature" in tc:
                    normalized_tc["thought_signature"] = tc["thought_signature"]
                if "extra_content" in tc:
                    normalized_tc["extra_content"] = tc["extra_content"]
                pending_ids.append(tc_id)
                normalized_tool_calls.append(normalized_tc)
            if normalized_tool_calls:
                msg["tool_calls"] = normalized_tool_calls

        elif role == "tool":
            tool_call_id = msg["tool_call_id"] if "tool_call_id" in msg else ""
            if tool_call_id:
                if pending_ids and pending_ids[0] == tool_call_id:
                    pending_ids.pop(0)
                elif tool_call_id in pending_ids:
                    pending_ids.remove(tool_call_id)
            elif pending_ids:
                msg["tool_call_id"] = pending_ids.pop(0)
            else:
                msg["tool_call_id"] = f"call_{uuid.uuid4().hex}"

        sanitized.append(msg)

    return sanitized


class OpenAIService(BaseLLMService):
    """Service for interacting with OpenAI API using the Responses API."""

    def __init__(self, model_id: str = "gpt-5.2", thinking_level: ThinkingLevel = "med"):
        super().__init__(model_id, thinking_level)
        self.client = openai.AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

    def _is_reasoning_model(self) -> bool:
        return any(self.model_id.startswith(prefix) for prefix in REASONING_MODELS)

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        """Convert OpenAI Chat Completions tool format to Responses API format."""
        responses_tools = []
        for tool in tools or []:
            func_def = tool.get("function") if isinstance(tool, dict) else None
            if not func_def:
                continue
            responses_tools.append(
                {
                    "type": "function",
                    "name": func_def.get("name", ""),
                    "description": func_def.get("description", ""),
                    "parameters": _normalize_json_schema(func_def.get("parameters", {})),
                    "strict": False,
                }
            )
        return responses_tools

    def _convert_messages_to_input(
        self, system_prompt: str, messages: list[Message]
    ) -> list[dict]:
        """Convert internal conversation to Responses API input format."""
        items: list[dict] = []

        for msg in messages:
            role = msg["role"]
            if role == "user":
                items.append(self._convert_user_msg(msg))
            elif role == "assistant":
                items.extend(self._convert_assistant_msg(msg))
            elif role == "tool":
                items.extend(self._convert_tool_msg(msg))

        return items

    @staticmethod
    def _convert_user_msg(msg: Message) -> dict:
        return {
            "type": "message",
            "role": "user",
            "content": _content_to_user_parts(msg["content"]),
        }

    @staticmethod
    def _convert_assistant_msg(msg: Message) -> list[dict]:
        items: list[dict] = []

        if "reasoning_items" in msg:
            items.extend(msg["reasoning_items"])

        if "tool_calls" in msg:
            for tool_call in msg["tool_calls"]:
                call_id = tool_call["id"] or f"call_{uuid.uuid4().hex}"
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call_id,
                        "name": tool_call["function"]["name"],
                        "arguments": tool_call["function"]["arguments"],
                    }
                )

        if msg["content"]:
            items.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": msg["content"]}],
                }
            )

        return items

    @staticmethod
    def _convert_tool_msg(msg: Message) -> list[dict]:
        """Convert a tool result message.

        OpenAI's function_call_output only accepts a text string, so any
        images in the tool result are injected as a follow-up user message
        placed *after* the output (preserving prompt cache prefixes).
        """
        items: list[dict] = []
        tool_call_id = msg["tool_call_id"] if "tool_call_id" in msg else f"call_{uuid.uuid4().hex}"

        text, image_parts = _split_tool_content(msg["content"])
        items.append(
            {
                "type": "function_call_output",
                "call_id": tool_call_id,
                "output": text,
            }
        )

        if image_parts:
            logger.warning(
                "OpenAI function_call_output does not support images. "
                "Injecting %d image(s) as a follow-up user message. "
                "The model will see them but they won't be associated "
                "with the tool call.",
                len(image_parts),
            )
            image_parts.append(
                {
                    "type": "input_text",
                    "text": "[Tool result images for the above function call]",
                }
            )
            items.append(
                {
                    "type": "message",
                    "role": "user",
                    "content": image_parts,
                }
            )

        return items

    async def generate_with_tools_streaming(
        self,
        system_prompt: str,
        messages: list[Message],
        tools: list[dict],
        response_format: dict | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream responses from OpenAI using the Responses API."""
        messages = sanitize_tool_messages(messages)
        responses_tools = self._convert_tools(tools)
        input_items = self._convert_messages_to_input(system_prompt, messages)

        request_params = {
            "model": self.model_id,
            "instructions": system_prompt,
            "input": input_items,
            "stream": True,
            "store": False,
        }

        if responses_tools:
            request_params["tools"] = responses_tools

        if response_format:
            # Responses API: text.format expects {type, name, schema, strict}
            # Convert from Chat Completions format if needed
            fmt = response_format
            if fmt.get("type") == "json_schema" and "json_schema" in fmt:
                js = fmt["json_schema"]
                fmt = {
                    "type": "json_schema",
                    "name": js.get("name", "response"),
                    "strict": js.get("strict", True),
                    "schema": js.get("schema", {}),
                }
            request_params["text"] = {"format": fmt}

        if self._is_reasoning_model():
            request_params["reasoning"] = {
                "effort": OPENAI_REASONING_EFFORT.get(self.thinking_level, "medium"),
                "summary": "auto",
            }
            request_params["include"] = ["reasoning.encrypted_content"]

        function_calls: dict[int, dict] = {}
        reasoning_items: list[dict] = []

        stream = await self.client.responses.create(**request_params)

        async for event in stream:
            event_type = event.type

            if event_type == "response.output_text.delta":
                yield StreamChunk(text_delta=event.delta)

            elif event_type == "response.reasoning_summary_text.delta":
                yield StreamChunk(thought_delta=event.delta)

            elif event_type == "response.output_item.added":
                item = event.item
                if getattr(item, "type", None) == "function_call":
                    output_index = event.output_index
                    function_calls[output_index] = {
                        "id": getattr(item, "call_id", None) or getattr(item, "id", None),
                        "name": getattr(item, "name", ""),
                        "arguments": "",
                    }

            elif event_type == "response.function_call_arguments.delta":
                output_index = event.output_index
                if output_index in function_calls:
                    function_calls[output_index]["arguments"] += event.delta

            elif event_type == "response.function_call_arguments.done":
                output_index = event.output_index
                if output_index in function_calls:
                    fc = function_calls[output_index]
                    call_id = fc.get("id") or f"call_{uuid.uuid4().hex}"
                    yield StreamChunk(
                        tool_calls_delta=[
                            {
                                "index": output_index,
                                "id": call_id,
                                "type": "function",
                                "thought_signature": None,
                                "extra_content": {},
                                "function": {
                                    "name": fc.get("name", ""),
                                    "arguments": fc.get("arguments", ""),
                                },
                            }
                        ]
                    )

            elif event_type == "response.output_item.done":
                item = event.item
                if getattr(item, "type", None) == "reasoning":
                    encrypted_content = getattr(item, "encrypted_content", None)
                    if encrypted_content:
                        summary_list = []
                        for s in getattr(item, "summary", []) or []:
                            summary_list.append(
                                {
                                    "type": getattr(s, "type", "summary_text"),
                                    "text": getattr(s, "text", ""),
                                }
                            )
                        reasoning_items.append(
                            {
                                "id": getattr(item, "id", None),
                                "type": "reasoning",
                                "summary": summary_list,
                                "encrypted_content": encrypted_content,
                            }
                        )

            elif event_type == "response.completed":
                if reasoning_items:
                    yield StreamChunk(reasoning_items=reasoning_items)
                response_obj = getattr(event, "response", None)
                if response_obj:
                    usage = getattr(response_obj, "usage", None)
                    if usage:
                        yield StreamChunk(
                            usage_raw={
                                "input_tokens": getattr(usage, "input_tokens", 0),
                                "output_tokens": getattr(usage, "output_tokens", 0),
                            }
                        )
                yield StreamChunk(finish_reason="stop")
