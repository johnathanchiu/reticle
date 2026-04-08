"""Quick-start: run a visual agent on an image.

Usage:
    uv run python main.py <image_path> [model]

Examples:
    uv run python main.py photo.png                    # uses gpt-5.4
    uv run python main.py photo.png gemini-2.5-flash   # uses Gemini
    uv run python main.py photo.png claude-sonnet-4-20250514

Set your API key via environment variables:
    export OPENAI_API_KEY=...
    export ANTHROPIC_API_KEY=...
    export GEMINI_API_KEY=...

Or set RETICLE_DEFAULT_MODEL to change the default model.
"""

import asyncio
import sys

from reticle.agent.events import CompleteEvent, TextDeltaEvent, ToolCallEvent, ToolResultEvent
from reticle.agent.loop import AgentLoop
from reticle.llm.base import Message
from reticle.llm.routing import get_llm_service
from reticle.tools.grid import render_grid_overlay
from reticle.tools.image import infer_media_type, load_image_base64
from reticle.tools.plot_points import PlotPointTool

SYSTEM_PROMPT = """\
You are a visual analysis agent. You can plot points on images to verify \
spatial coordinates and understand the layout of what you see.

Use the plot_points tool to mark key features, boundaries, and landmarks. \
The coordinate system is 0-1000 on both axes (origin at top-left). \
A grid overlay is shown with red lines for x-axis and blue for y-axis.

Analyze the image systematically:
1. Identify the main structural elements
2. Plot points on key features to calibrate your spatial understanding
3. Describe what you see with precise coordinate references
"""


async def main(image_path: str, model: str = "gpt-5.4") -> None:
    # Set up LLM + visual tools
    llm = get_llm_service(model, thinking_level="low")
    grid_b64 = render_grid_overlay(image_path)

    plot_tool = PlotPointTool()
    plot_tool.set_image_path(image_path)
    plot_tool.set_grid_b64(grid_b64)

    loop = AgentLoop(
        llm=llm,
        tools=[plot_tool],
        system_prompt=SYSTEM_PROMPT,
        max_turns=5,
        trace=True,
    )

    # Seed conversation with the image
    img_b64 = load_image_base64(image_path)
    media_type = infer_media_type(image_path)
    loop.conversation.append(
        Message(
            role="user",
            content=[  # type: ignore[typeddict-item]
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": img_b64},
                },
                {"type": "text", "text": "Analyze this image. Plot points on key features."},
            ],
        )
    )

    # Run the agent loop
    async for event in loop.run():
        if isinstance(event, TextDeltaEvent):
            print(event.delta, end="", flush=True)
        elif isinstance(event, ToolCallEvent):
            print(f"\n> [tool] {event.name}({event.args})")
        elif isinstance(event, ToolResultEvent):
            if event.error:
                print(f"\n> [error] {event.error}")
            else:
                print(f"\n> [result] {event.output[:200] if isinstance(event.output, str) else '(image + feedback)'}")
        elif isinstance(event, CompleteEvent):
            print(f"\n\n--- Done: {event.reason} ---")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python main.py <image_path> [model]")
        print("\nModels: gpt-5.4 (default), gemini-3.1-pro-preview, claude-opus-4-20250514, etc.")
        sys.exit(1)

    image = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "gpt-5.4"
    asyncio.run(main(image, model_name))
