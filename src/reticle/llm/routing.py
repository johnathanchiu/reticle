"""Model prefix -> provider routing."""

from __future__ import annotations

import os

from reticle.llm.base import BaseLLMService, ThinkingLevel

MODEL_PREFIXES: dict[str, str] = {
    "gpt": "openai",
    "o1": "openai",
    "o3": "openai",
    "o4": "openai",
    "gemini": "gemini",
}


def get_provider_for_model(model_id: str) -> str:
    if "/" in model_id:
        prefix = model_id.split("/")[0]
        if prefix in MODEL_PREFIXES:
            return MODEL_PREFIXES[prefix]

    for prefix, provider in MODEL_PREFIXES.items():
        if model_id.startswith(prefix):
            return provider

    raise ValueError(
        f"Cannot determine provider for model '{model_id}'. "
        f"Known prefixes: {list(MODEL_PREFIXES.keys())}"
    )


def get_llm_service(
    model_id: str | None = None,
    thinking_level: ThinkingLevel = "med",
) -> BaseLLMService:
    model_id = model_id or os.environ.get("RETICLE_DEFAULT_MODEL", "gpt-5.4")
    provider = get_provider_for_model(model_id)

    if provider == "openai":
        from reticle.llm.openai import OpenAIService

        return OpenAIService(model_id, thinking_level)
    elif provider == "gemini":
        from reticle.llm.gemini import GeminiService

        return GeminiService(model_id, thinking_level)

    raise ValueError(f"Unsupported provider: {provider}")
