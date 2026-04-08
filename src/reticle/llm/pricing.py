"""LLM pricing per million tokens — used for cost tracking."""

from __future__ import annotations

# (input_per_million, output_per_million) in USD
# Sources: provider pricing pages as of March 2026
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-5.2": (2.00, 8.00),
    "gpt-5": (2.00, 8.00),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "o4-mini": (1.10, 4.40),
    "o3": (2.00, 8.00),
    "o3-mini": (1.10, 4.40),
    "o1": (15.00, 60.00),
    "o1-mini": (1.10, 4.40),
    # Anthropic
    "claude-opus-4-20250514": (15.00, 75.00),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-haiku-4-20250514": (0.80, 4.00),
    # Gemini
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-2.0-flash": (0.10, 0.40),
}


def get_pricing(model_id: str) -> tuple[float, float]:
    """Return (input_cost_per_M, output_cost_per_M) for a model.

    Falls back to prefix matching, then returns (0, 0) if unknown.
    """
    if model_id in MODEL_PRICING:
        return MODEL_PRICING[model_id]
    # Prefix match: "gpt-4.1-mini-2025..." → "gpt-4.1-mini"
    for key in sorted(MODEL_PRICING, key=len, reverse=True):
        if model_id.startswith(key):
            return MODEL_PRICING[key]
    return (0.0, 0.0)


def compute_cost(
    model_id: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Compute cost in USD for a single LLM call."""
    inp_rate, out_rate = get_pricing(model_id)
    return (input_tokens * inp_rate + output_tokens * out_rate) / 1_000_000
