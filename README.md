# reticle

Visual agent toolkit — grid overlays and spatial tools for LLM agents.

LLMs struggle with precise spatial reasoning in images. Reticle bridges this gap by giving agents a coordinate grid overlay and point-plotting tools with edge-detection feedback, enabling iterative spatial understanding through tool use.

## How it works

1. **Grid overlay** — renders a 1000x1000 normalized coordinate grid on any image (red=x, blue=y)
2. **Point plotting** — agents plot points and receive feedback: ON_EDGE, CLOSE_TO_EDGE, EMPTY_SPACE, with nearest-edge distances
3. **Agent loop** — multi-turn loop where the agent sees images, calls tools, gets visual feedback, and refines its understanding

## Quick start

```bash
# Install with your preferred LLM provider
pip install reticle[openai]    # or [gemini], [all]

# Set your API key
export OPENAI_API_KEY=sk-...

# Run on an image
python main.py photo.png
python main.py floorplan.png gemini-3.1-pro-preview
```

## Usage

```python
import asyncio
from reticle.agent.loop import AgentLoop
from reticle.llm.routing import get_llm_service
from reticle.tools.grid import render_grid_overlay
from reticle.tools.plot_points import PlotPointTool
from reticle.tools.image import load_image_base64, infer_media_type
from reticle.agent.events import TextDeltaEvent, ToolCallEvent, CompleteEvent

async def analyze(image_path: str):
    llm = get_llm_service("gpt-5.4", thinking_level="low")
    grid_b64 = render_grid_overlay(image_path)

    plot_tool = PlotPointTool()
    plot_tool.set_image_path(image_path)
    plot_tool.set_grid_b64(grid_b64)

    loop = AgentLoop(
        llm=llm,
        tools=[plot_tool],
        system_prompt="You are a visual analysis agent. Use plot_points to verify coordinates.",
        max_turns=5,
    )

    # Seed with image
    img_b64 = load_image_base64(image_path)
    media_type = infer_media_type(image_path)
    loop.conversation.append({
        "role": "user",
        "content": [
            {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": img_b64}},
            {"type": "text", "text": "Analyze this image."},
        ],
    })

    async for event in loop.run():
        if isinstance(event, TextDeltaEvent):
            print(event.delta, end="", flush=True)
        elif isinstance(event, CompleteEvent):
            print(f"\nDone: {event.reason}")

asyncio.run(analyze("photo.png"))
```

## Building custom tools

Extend `BaseDeclarativeTool` to create your own visual tools:

```python
from reticle.agent.tools.base import BaseDeclarativeTool, BaseToolInvocation, ToolResult

class MyTool(BaseDeclarativeTool):
    def __init__(self):
        schema = {
            "type": "function",
            "function": {
                "name": "my_tool",
                "description": "Does something visual",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
        super().__init__("my_tool", schema)

    async def build(self, params):
        return MyToolInvocation(params)

class MyToolInvocation(BaseToolInvocation):
    def get_description(self):
        return "Running my tool"

    async def execute(self):
        # Return rich content (images + text) for the agent to see
        return ToolResult(
            output="result",
            content_blocks=[
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}},
                {"type": "text", "text": "Feedback for the agent"},
            ],
        )
```

## Supported LLM providers

- **OpenAI** — GPT-4.1, GPT-5, o3, o4 (Responses API with reasoning)
- **Google** — Gemini 2.5/3.x (thinking budgets)

Set API keys via environment variables: `OPENAI_API_KEY`, `GEMINI_API_KEY`.

## Architecture

```
reticle/
├── agent/         # Core agent loop, events, tool framework
│   ├── loop.py    # AgentLoop — multi-turn agentic loop
│   ├── events.py  # Typed event system (streaming)
│   └── tools/     # BaseDeclarativeTool, ToolRegistry
├── llm/           # Multi-provider LLM abstraction
│   ├── base.py    # BaseLLMService interface
│   ├── routing.py # Prefix-based provider routing
│   └── ...        # OpenAI, Gemini
└── tools/         # Visual tools
    ├── grid.py    # Coordinate grid overlay
    ├── plot_points.py  # Point plotting + edge detection
    └── image.py   # Image utilities
```

## License

MIT
