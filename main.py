"""Quick-start demo: point to objects in an image.

Usage:
    uv run python main.py <image_path> "point to the <object>"
    uv run python main.py <image_path> "<any prompt>" --model gemini-3.1-pro-preview

Examples:
    uv run python main.py photo.png "point to the red car"
    uv run python main.py kitchen.jpg "point to every appliance"
    uv run python main.py floorplan.png "point to all the doors"
    uv run python main.py room.jpg "point to the corners of the table"

Set your API key via environment variables:
    export OPENAI_API_KEY=...
    export GEMINI_API_KEY=...

Or set RETICLE_DEFAULT_MODEL to change the default model.
"""

import argparse
import asyncio
import base64
import os
from io import BytesIO

from reticle.agent.events import (
    CompleteEvent,
    LLMUsageEvent,
    TextDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from reticle.agent.loop import AgentLoop
from reticle.llm.base import Message
from reticle.llm.routing import get_llm_service
from reticle.tools.grid import render_grid_overlay
from reticle.tools.image import infer_media_type, load_image_base64
from reticle.tools.plot_points import PlotPointTool

SYSTEM_PROMPT = """\
You are a visual pointing agent. When asked to point to something in an image, \
use the plot_points tool to place labeled markers precisely on the requested objects.

The coordinate system is 0-1000 on both axes (origin at top-left). \
A grid overlay is shown with red lines for x-axis and blue for y-axis.

## How to point

1. Study the image and the grid coordinates carefully.
2. Call plot_points with labeled markers on the requested object(s).
3. Check the feedback — if points are off-target, adjust and re-plot.
4. Once your points land correctly, briefly describe what you marked and where.

Be precise. Use the grid lines to estimate coordinates. Label each point clearly \
(e.g. "front-left wheel", "top-right corner", "door handle").
"""


async def run(image_path: str, prompt: str, model: str = "gpt-5.4") -> None:
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

    # Seed conversation with the image + user prompt
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
                {"type": "text", "text": prompt},
            ],
        )
    )

    print(f"\n  Image: {image_path}")
    print(f"  Model: {model}")
    print(f"  Prompt: {prompt}")
    print()

    # Run the agent loop
    last_plot_b64: str | None = None

    async for event in loop.run():
        if isinstance(event, TextDeltaEvent):
            print(event.delta, end="", flush=True)
        elif isinstance(event, ToolCallEvent):
            n_points = len(event.args.get("points", []))
            print(f"\n  >> plotting {n_points} point(s)...", flush=True)
        elif isinstance(event, ToolResultEvent):
            if event.error:
                print(f"\n  >> error: {event.error}")
            else:
                # Extract feedback text from content blocks
                for block in event.content_blocks:
                    if isinstance(block, dict) and block.get("type") == "text":
                        print(f"\n  >> feedback:\n{_indent(block['text'], 5)}")
                    if isinstance(block, dict) and block.get("type") == "image":
                        source = block.get("source", {})
                        if source.get("type") == "base64":
                            last_plot_b64 = source["data"]
        elif isinstance(event, LLMUsageEvent):
            print(f"\n  >> tokens: {event.input_tokens} in / {event.output_tokens} out")
        elif isinstance(event, CompleteEvent):
            print(f"\n\n  --- {event.reason} ---")

    # Save the final annotated image
    if last_plot_b64:
        out_path = _output_path(image_path)
        img_bytes = base64.b64decode(last_plot_b64)
        from PIL import Image

        img = Image.open(BytesIO(img_bytes))
        img.save(out_path)
        print(f"\n  Saved annotated image: {out_path}")


def _indent(text: str, n: int) -> str:
    pad = " " * n
    return "\n".join(f"{pad}{line}" for line in text.splitlines())


def _output_path(image_path: str) -> str:
    base, ext = os.path.splitext(image_path)
    return f"{base}_pointed{ext or '.png'}"


def main():
    parser = argparse.ArgumentParser(
        description="Point to objects in an image using an LLM agent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  uv run python main.py photo.png "point to the red car"
  uv run python main.py kitchen.jpg "point to every appliance"
  uv run python main.py room.jpg "where are the windows?" --model gemini-3.1-pro-preview
""",
    )
    parser.add_argument("image", help="Path to the image file")
    parser.add_argument(
        "prompt",
        nargs="?",
        default="point to the most interesting objects in this image",
        help="What to point to (default: interesting objects)",
    )
    parser.add_argument("--model", default="gpt-5.4", help="LLM model to use (default: gpt-5.4)")

    args = parser.parse_args()
    asyncio.run(run(args.image, args.prompt, args.model))


if __name__ == "__main__":
    main()
