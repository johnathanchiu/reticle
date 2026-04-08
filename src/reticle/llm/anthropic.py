"""Anthropic API service with streaming tool support and extended thinking."""

import json
import os
from collections.abc import AsyncGenerator

import anthropic
from anthropic.types import (
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
)

from reticle.llm.base import BaseLLMService, Message, StreamChunk, ThinkingLevel

MAX_TOKENS = 64000
ANTHROPIC_THINKING_BUDGETS = {
    "low": 1024,
    "med": 10000,
    "high": 32000,
}


class AnthropicService(BaseLLMService):
    """Service for interacting with Anthropic Claude API."""

    def __init__(
        self,
        model_id: str = "claude-sonnet-4-20250514",
        thinking_level: ThinkingLevel = "med",
    ):
        super().__init__(model_id, thinking_level)
        self.client = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

    def _is_reasoning_model(self) -> bool:
        return any(x in self.model_id for x in ("claude-3", "claude-sonnet-4", "claude-opus-4"))

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        """Convert OpenAI-style tool specs to Anthropic format."""
        anthropic_tools = []
        for tool in tools or []:
            func_def = tool.get("function") if isinstance(tool, dict) else None
            if not func_def:
                continue
            anthropic_tools.append(
                {
                    "name": func_def.get("name", ""),
                    "description": func_def.get("description", ""),
                    "input_schema": func_def.get("parameters", {}),
                }
            )
        return anthropic_tools

    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        """Convert OpenAI-format messages to Anthropic format.

        Content can be a string or a list of content blocks (for multimodal).
        List content is passed through directly — Anthropic accepts both formats.
        """
        anthropic_messages: list[dict] = []

        for msg in messages:
            role = msg["role"]

            if role == "user":
                # Content may be str or list[dict] (multimodal blocks)
                anthropic_messages.append({"role": "user", "content": msg["content"]})

            elif role == "assistant":
                content_blocks = []

                if "thought_summary" in msg and msg["thought_summary"]:
                    thinking_block: dict = {
                        "type": "thinking",
                        "thinking": msg["thought_summary"],
                    }
                    if "thinking_signature" in msg and msg["thinking_signature"]:
                        thinking_block["signature"] = msg["thinking_signature"]
                    content_blocks.append(thinking_block)

                if msg["content"]:
                    content_blocks.append({"type": "text", "text": msg["content"]})

                tool_calls = msg["tool_calls"] if "tool_calls" in msg else []
                for tool_call in tool_calls:
                    func = tool_call.get("function", {})
                    try:
                        input_data = json.loads(func.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        input_data = {}
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tool_call.get("id", ""),
                            "name": func.get("name", ""),
                            "input": input_data,
                        }
                    )

                if content_blocks:
                    anthropic_messages.append({"role": "assistant", "content": content_blocks})

            elif role == "tool":
                # Content may be str (JSON) or list[dict] (rich content blocks)
                content = msg["content"]
                tool_result = {
                    "type": "tool_result",
                    "tool_use_id": msg["tool_call_id"] if "tool_call_id" in msg else "",
                    "content": content,
                }

                if (
                    anthropic_messages
                    and anthropic_messages[-1].get("role") == "user"
                    and isinstance(anthropic_messages[-1].get("content"), list)
                ):
                    anthropic_messages[-1]["content"].append(tool_result)
                else:
                    anthropic_messages.append({"role": "user", "content": [tool_result]})

        return anthropic_messages

    async def generate_with_tools_streaming(
        self,
        system_prompt: str,
        messages: list[Message],
        tools: list[dict],
        response_format: dict | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream responses from Anthropic with tool calling and extended thinking."""
        anthropic_tools = self._convert_tools(tools)
        anthropic_messages = self._convert_messages(messages)

        request_params = {
            "model": self.model_id,
            "system": system_prompt,
            "messages": anthropic_messages,
            "max_tokens": MAX_TOKENS,
        }

        if anthropic_tools:
            request_params["tools"] = anthropic_tools

        if response_format:
            # Convert from OpenAI-style response_format to Anthropic output_config
            fmt = response_format
            if fmt.get("type") == "json_schema" and "json_schema" in fmt:
                js = fmt["json_schema"]
                request_params["output_config"] = {
                    "format": {
                        "type": "json_schema",
                        "schema": js["schema"],
                    }
                }

        if self._is_reasoning_model():
            request_params["thinking"] = {
                "type": "enabled",
                "budget_tokens": ANTHROPIC_THINKING_BUDGETS.get(self.thinking_level, 10000),
            }

        current_block_type = None
        current_tool_id = None
        current_tool_name = None
        current_tool_input = ""
        tool_index = 0

        async with self.client.messages.stream(**request_params) as stream:
            async for event in stream:
                if isinstance(event, RawContentBlockStartEvent):
                    block = event.content_block
                    current_block_type = block.type
                    if block.type == "tool_use":
                        current_tool_id = block.id
                        current_tool_name = block.name
                        current_tool_input = ""

                elif isinstance(event, RawContentBlockDeltaEvent):
                    delta = event.delta
                    if current_block_type == "thinking":
                        thinking = getattr(delta, "thinking", None)
                        if thinking:
                            yield StreamChunk(thought_delta=thinking)
                    elif current_block_type == "text":
                        text = getattr(delta, "text", None)
                        if text:
                            yield StreamChunk(text_delta=text)
                    elif current_block_type == "tool_use":
                        partial = getattr(delta, "partial_json", None)
                        if partial:
                            current_tool_input += partial

                elif isinstance(event, RawContentBlockStopEvent):
                    if current_block_type == "tool_use" and current_tool_name:
                        yield StreamChunk(
                            tool_calls_delta=[
                                {
                                    "index": tool_index,
                                    "id": current_tool_id,
                                    "type": "function",
                                    "thought_signature": None,
                                    "extra_content": {},
                                    "function": {
                                        "name": current_tool_name,
                                        "arguments": current_tool_input,
                                    },
                                }
                            ]
                        )
                        tool_index += 1
                    current_block_type = None
                    current_tool_id = None
                    current_tool_name = None
                    current_tool_input = ""

                elif isinstance(event, RawMessageDeltaEvent):
                    stop_reason = event.delta.stop_reason
                    if stop_reason:
                        yield StreamChunk(finish_reason=stop_reason)

            final_message = await stream.get_final_message()
            if final_message:
                # Extract thinking signature
                if final_message.content:
                    for block in final_message.content:
                        if getattr(block, "type", None) == "thinking":
                            signature = getattr(block, "signature", None)
                            if signature:
                                yield StreamChunk(thought_signature=signature)
                            break
                # Extract token usage
                if final_message.usage:
                    yield StreamChunk(
                        usage_raw={
                            "input_tokens": final_message.usage.input_tokens,
                            "output_tokens": final_message.usage.output_tokens,
                        }
                    )
