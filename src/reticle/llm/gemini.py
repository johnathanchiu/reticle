"""Gemini API service with streaming tool support."""

import base64
import json
import os
import uuid
from collections.abc import AsyncGenerator
from typing import Any, cast

from google import genai
from google.genai import types
from google.genai.types import Content, FunctionCall, FunctionDeclaration, Part

from reticle.llm.base import BaseLLMService, Message, StreamChunk, ThinkingLevel

# Thinking token budgets (works for both Gemini 2.5 and 3.x)
GEMINI_THINKING_BUDGETS: dict[str, int] = {
    "none": 0,
    "low": 4096,
    "med": 8192,
    "high": 24576,
}


class GeminiService(BaseLLMService):
    """Service for interacting with Gemini API."""

    def __init__(
        self,
        model_id: str = "gemini-3-pro-preview",
        thinking_level: ThinkingLevel = "med",
    ):
        clean_model_id = model_id.removeprefix("gemini/")
        super().__init__(clean_model_id, thinking_level)
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
        self._supports_thinking = self._check_thinking_support()

    def _check_thinking_support(self) -> bool:
        """Query the Gemini API to check if this model supports thinking."""
        try:
            model_info = self.client.models.get(model=self.model_id)
            return bool(model_info.thinking)
        except Exception:
            return False

    def _build_thinking_config(self) -> types.ThinkingConfig:
        """Build the right ThinkingConfig for this model's version.

        Uses thinking_budget for all models to allow precise control over
        thinking token usage. Both Gemini 2.5 and 3.x support this.
        """
        budget = GEMINI_THINKING_BUDGETS.get(self.thinking_level, 0)
        return types.ThinkingConfig(
            include_thoughts=budget > 0,
            thinking_budget=budget,
        )

    def _convert_tools(self, tools: list[dict]) -> list[FunctionDeclaration]:
        """Convert OpenAI-style tool specs into Gemini FunctionDeclarations."""
        declarations: list[FunctionDeclaration] = []
        for tool in tools or []:
            func_def = tool.get("function") if isinstance(tool, dict) else None
            if not func_def:
                continue
            parameters = func_def.get("parameters")
            if parameters and isinstance(parameters, dict):
                parameters = _dereference_schema(parameters)
                parameters = _strip_unsupported_keys(parameters)
            declarations.append(
                FunctionDeclaration(
                    name=func_def.get("name", ""),
                    description=func_def.get("description", ""),
                    parameters=cast(Any, parameters),
                )
            )
        return declarations

    @staticmethod
    def _convert_arguments(args: str | dict | list | None) -> dict[str, Any]:
        if args is None:
            return {}
        if isinstance(args, dict):
            return args
        if isinstance(args, list):
            return {"items": args}
        return json.loads(args)

    @staticmethod
    def _encode_signature(signature: bytes | None) -> str | None:
        if signature is None:
            return None
        return base64.b64encode(signature).decode("utf-8")

    @staticmethod
    def _decode_signature(signature: str | None) -> bytes | None:
        if signature is None:
            return None
        return base64.b64decode(signature)

    @staticmethod
    def _usage_metadata_to_dict(
        usage_metadata: types.GenerateContentResponseUsageMetadata | None,
    ) -> dict[str, Any] | None:
        if usage_metadata is None:
            return None
        raw = usage_metadata.model_dump(exclude_none=True)
        # Normalize to input_tokens / output_tokens for consistency
        return {
            "input_tokens": raw.get("prompt_token_count", 0),
            "output_tokens": raw.get("candidates_token_count", 0),
            **raw,
        }

    @staticmethod
    def _content_blocks_to_parts(content: list[dict]) -> list[Part]:
        """Convert Anthropic-style content blocks to Gemini Parts."""
        parts: list[Part] = []
        for block in content:
            block_type = block.get("type", "")
            if block_type == "text":
                parts.append(Part(text=block.get("text", "")))
            elif block_type == "image":
                source = block.get("source", {})
                if source.get("type") == "base64":
                    data = base64.b64decode(source["data"])
                    mime = source.get("media_type", "image/png")
                    parts.append(Part(inline_data=types.Blob(mime_type=mime, data=data)))
        return parts

    def _convert_message(self, message: Message) -> Content:
        """Translate OpenAI-formatted messages to Gemini Content objects.

        Content can be str or list[dict] (Anthropic-style content blocks).
        """
        role: str = message["role"]
        if role == "assistant":
            role = "model"

        parts: list[Part] = []
        content = message["content"]

        if role == "model" and "tool_calls" in message:
            if content:
                if isinstance(content, list):
                    parts.extend(self._content_blocks_to_parts(content))
                else:
                    parts.append(Part(text=content))
            for tool_call in message["tool_calls"]:
                function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
                signature = None
                if isinstance(tool_call, dict):
                    raw_signature = tool_call.get("thought_signature") or tool_call.get(
                        "extra_content", {}
                    ).get("google", {}).get("thought_signature")
                    signature = self._decode_signature(raw_signature)
                fn_call = FunctionCall(
                    name=function.get("name", ""),
                    args=cast(Any, self._convert_arguments(function.get("arguments"))),
                )
                parts.append(Part(function_call=fn_call, thought_signature=signature))

        elif message["role"] == "tool":
            # Tool results: content may be str (JSON) or list[dict] (rich blocks)
            if isinstance(content, list):
                # Extract text blocks for the response body, images as separate parts
                blocks = cast(list[dict[str, Any]], content)
                text_parts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
                response_body = {"result": "\n".join(text_parts)} if text_parts else {}
                # Add image parts directly
                for block in blocks:
                    if block.get("type") == "image":
                        source = block.get("source", {})
                        if source.get("type") == "base64":
                            data = base64.b64decode(source["data"])
                            mime = source.get("media_type", "image/png")
                            parts.append(Part(inline_data=types.Blob(mime_type=mime, data=data)))
            else:
                response_body = self._convert_arguments(content)

            tool_name = (
                message["name"]
                if "name" in message
                else (message["tool_call_id"] if "tool_call_id" in message else "")
            )
            parts.append(
                Part(
                    function_response=types.FunctionResponse(
                        name=tool_name,
                        response=response_body,
                    )
                )
            )
            role = "user"

        elif content:
            if isinstance(content, list):
                parts.extend(self._content_blocks_to_parts(content))
            else:
                parts.append(Part(text=content))

        if not parts:
            # Gemini requires at least one part; use empty text as a fallback
            # rather than crashing the loop.
            parts.append(Part(text=""))

        return Content(role=role, parts=parts)

    def _convert_messages(self, messages: list[Message]) -> list[Content]:
        return [self._convert_message(msg) for msg in messages]

    async def generate_with_tools_streaming(
        self,
        system_prompt: str,
        messages: list[Message],
        tools: list[dict],
        response_format: dict | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream responses from Gemini with tool calling and thought summaries."""
        tool_declarations = self._convert_tools(tools)
        contents = self._convert_messages(messages)

        thinking_config = None
        if self._supports_thinking:
            thinking_config = self._build_thinking_config()

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=[types.Tool(function_declarations=tool_declarations)]
            if tool_declarations
            else None,
            thinking_config=thinking_config,
            temperature=1.0,
        )

        stream = await self.client.aio.models.generate_content_stream(
            model=self.model_id,
            contents=cast(Any, contents),
            config=config,
        )

        async for chunk in stream:
            if not chunk.candidates:
                continue

            candidate = chunk.candidates[0]
            finish_reason = candidate.finish_reason
            text_delta = ""
            thought_delta = ""
            tool_calls_delta: list[dict] = []
            usage_raw = self._usage_metadata_to_dict(chunk.usage_metadata)

            parts = candidate.content.parts if candidate.content else []
            for part in parts or []:
                if part.text:
                    if part.thought:
                        thought_delta += part.text
                    else:
                        text_delta += part.text

                if part.function_call:
                    fc = part.function_call
                    args = fc.args
                    encoded_signature = self._encode_signature(part.thought_signature)
                    call_id = fc.id or f"call_{uuid.uuid4().hex}"
                    tool_calls_delta.append(
                        {
                            "index": len(tool_calls_delta),
                            "id": call_id,
                            "type": "function",
                            "thought_signature": encoded_signature,
                            "extra_content": {"google": {"thought_signature": encoded_signature}}
                            if encoded_signature
                            else {},
                            "function": {
                                "name": fc.name or "",
                                "arguments": (json.dumps(args) if isinstance(args, dict) else ""),
                            },
                        }
                    )

            if any([text_delta, thought_delta, tool_calls_delta, finish_reason, usage_raw]):
                yield StreamChunk(
                    text_delta=text_delta or None,
                    thought_delta=thought_delta or None,
                    tool_calls_delta=tool_calls_delta or None,
                    finish_reason=finish_reason,
                    usage_raw=usage_raw,
                )


def _dereference_schema(schema: dict) -> dict:
    """Inline all $ref references using $defs. Gemini doesn't support $ref."""
    defs = schema.get("$defs", {})
    if not defs:
        return schema

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                ref_path = node["$ref"]  # e.g. "#/$defs/WallSchema"
                ref_name = ref_path.rsplit("/", 1)[-1]
                if ref_name in defs:
                    return resolve(defs[ref_name])
                return node
            return {k: resolve(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    return resolve(schema)


def _strip_unsupported_keys(schema: Any) -> Any:
    """Remove keys Gemini rejects: additionalProperties, title, default."""
    _bad = {"additionalProperties", "title", "default"}
    if isinstance(schema, dict):
        return {k: _strip_unsupported_keys(v) for k, v in schema.items() if k not in _bad}
    if isinstance(schema, list):
        return [_strip_unsupported_keys(item) for item in schema]
    return schema
