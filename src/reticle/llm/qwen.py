"""Qwen API service via DashScope's OpenAI-compatible endpoint.

Uses the Chat Completions API (not Responses API) with Qwen-specific
thinking support via `enable_thinking`.
"""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import AsyncGenerator
from typing import Any, cast

import openai

from .base import (
    BaseLLMService,
    Message,
    StreamChunk,
    ThinkingLevel,
)
from .openai import _normalize_json_schema, _split_tool_content, sanitize_tool_messages

logger = logging.getLogger(__name__)

DASHSCOPE_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


class QwenService(BaseLLMService):
    """Service for Qwen models via DashScope's OpenAI-compatible API."""

    def __init__(
        self,
        model_id: str = "qwen3.6-plus",
        thinking_level: ThinkingLevel = "med",
    ):
        super().__init__(model_id, thinking_level)
        self.client = openai.AsyncOpenAI(
            api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
            base_url=DASHSCOPE_BASE_URL,
        )

    # ---- Message conversion (Chat Completions format) ----

    @staticmethod
    def _convert_messages(system_prompt: str, messages: list[Message]) -> list[dict]:
        """Convert internal messages to Chat Completions format."""
        chat_msgs: list[dict] = []
        if system_prompt:
            chat_msgs.append({"role": "system", "content": system_prompt})

        for msg in messages:
            role = msg["role"]
            if role == "user":
                content = cast("str | list[dict[str, Any]]", msg["content"])
                if isinstance(content, list):
                    parts = []
                    for block in content:
                        if not isinstance(block, dict):
                            parts.append({"type": "text", "text": str(block)})
                        elif block.get("type") == "text":
                            parts.append({"type": "text", "text": block["text"]})
                        elif block.get("type") == "image":
                            source = block.get("source", {})
                            if source.get("type") == "base64":
                                parts.append(
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{source['media_type']};base64,{source['data']}"
                                        },
                                    }
                                )
                    chat_msgs.append({"role": "user", "content": parts})
                else:
                    chat_msgs.append({"role": "user", "content": content})
            elif role == "assistant":
                m: dict = {"role": "assistant", "content": msg["content"] or None}
                if "tool_calls" in msg:
                    m["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["function"]["name"],
                                "arguments": tc["function"]["arguments"],
                            },
                        }
                        for tc in msg["tool_calls"]
                    ]
                chat_msgs.append(m)
            elif role == "tool":
                text, _images = _split_tool_content(msg["content"])
                chat_msgs.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.get("tool_call_id", ""),
                        "content": text,
                    }
                )

        return chat_msgs

    @staticmethod
    def _convert_tools(tools: list[dict]) -> list[dict]:
        """Convert tools to Chat Completions format."""
        chat_tools = []
        for tool in tools or []:
            func_def = tool.get("function") if isinstance(tool, dict) else None
            if not func_def:
                continue
            chat_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": func_def.get("name", ""),
                        "description": func_def.get("description", ""),
                        "parameters": _normalize_json_schema(func_def.get("parameters", {})),
                    },
                }
            )
        return chat_tools

    @staticmethod
    def _sanitize_messages(messages: list[Message]) -> list[Message]:
        return sanitize_tool_messages(messages)

    # ---- Streaming ----

    async def generate_with_tools_streaming(
        self,
        system_prompt: str,
        messages: list[Message],
        tools: list[dict],
        response_format: dict | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream via DashScope Chat Completions API with Qwen thinking support."""
        messages = self._sanitize_messages(messages)
        chat_messages = self._convert_messages(system_prompt, messages)
        chat_tools = self._convert_tools(tools)

        request_params: dict = {
            "model": self.model_id,
            "messages": chat_messages,
            "stream": True,
        }

        if chat_tools:
            request_params["tools"] = chat_tools

        # Enable Qwen thinking/reasoning
        request_params["extra_body"] = {"enable_thinking": True}

        function_calls: dict[int, dict] = {}

        stream = await self.client.chat.completions.create(**request_params)

        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
                continue

            finish_reason = chunk.choices[0].finish_reason if chunk.choices else None

            # Qwen reasoning_content (thinking)
            reasoning_content = getattr(delta, "reasoning_content", None)
            if reasoning_content:
                yield StreamChunk(thought_delta=reasoning_content)

            # Text content
            if delta.content:
                yield StreamChunk(text_delta=delta.content)

            # Tool calls
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in function_calls:
                        function_calls[idx] = {
                            "id": tc_delta.id or f"call_{uuid.uuid4().hex}",
                            "name": "",
                            "arguments": "",
                        }
                    if tc_delta.id:
                        function_calls[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            function_calls[idx]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            function_calls[idx]["arguments"] += tc_delta.function.arguments

            if finish_reason:
                # Emit completed tool calls
                for idx, fc in sorted(function_calls.items()):
                    yield StreamChunk(
                        tool_calls_delta=[
                            {
                                "index": idx,
                                "id": fc["id"],
                                "type": "function",
                                "thought_signature": None,
                                "extra_content": {},
                                "function": {
                                    "name": fc["name"],
                                    "arguments": fc["arguments"],
                                },
                            }
                        ]
                    )

                # Usage
                if hasattr(chunk, "usage") and chunk.usage:
                    yield StreamChunk(
                        usage_raw={
                            "input_tokens": getattr(chunk.usage, "prompt_tokens", 0),
                            "output_tokens": getattr(chunk.usage, "completion_tokens", 0),
                        }
                    )

                yield StreamChunk(finish_reason="stop")
