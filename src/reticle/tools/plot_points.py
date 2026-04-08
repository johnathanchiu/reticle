"""PlotPointTool — plot points on images with edge detection feedback.

Lets LLM agents verify spatial coordinates by plotting points on a
grid-overlaid image and receiving feedback about edge proximity,
wall detection, and brightness analysis.
"""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

from reticle.agent.tools.base import (
    BaseDeclarativeTool,
    BaseToolInvocation,
    ToolResult,
    ToolSchema,
)
from reticle.tools.grid import _get_norm_scale, scale_factor


def _compute_edge_map(img_array: Any) -> Any:
    """Compute edge map using Sobel gradient magnitude. Returns float32 array 0-255."""
    import numpy as np

    gray = np.mean(img_array[:, :, :3].astype(np.float32), axis=2)

    h, w = gray.shape
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)

    gx[:, 1:-1] = -gray[:, :-2] + gray[:, 2:]
    gy[1:-1, :] = -gray[:-2, :] + gray[2:, :]

    magnitude = np.sqrt(gx**2 + gy**2)
    max_val = magnitude.max()
    if max_val > 0:
        magnitude = (magnitude / max_val) * 255.0
    return magnitude


def _find_nearest_edge(
    edge_map: Any,
    px_x: int,
    px_y: int,
    search_radius: int = 50,
    edge_threshold: float = 40.0,
) -> tuple[int, int, float] | None:
    """Find nearest edge pixel within search_radius. Returns (px_x, px_y, dist) or None."""
    import numpy as np

    h, w = edge_map.shape[:2]
    is_edge = edge_map > edge_threshold

    y_min = max(0, px_y - search_radius)
    y_max = min(h, px_y + search_radius + 1)
    x_min = max(0, px_x - search_radius)
    x_max = min(w, px_x + search_radius + 1)

    patch = is_edge[y_min:y_max, x_min:x_max]
    ys, xs = np.where(patch)
    if len(ys) == 0:
        return None

    ys = ys + y_min
    xs = xs + x_min

    dists = np.sqrt((xs - px_x) ** 2 + (ys - px_y) ** 2)
    best = np.argmin(dists)
    return int(xs[best]), int(ys[best]), float(dists[best])


_POINT_COLORS = [
    (255, 0, 255),  # magenta
    (0, 255, 0),  # green
    (255, 128, 0),  # orange
    (0, 200, 255),  # cyan
    (255, 255, 0),  # yellow
    (255, 0, 0),  # red
    (128, 0, 255),  # purple
    (0, 255, 128),  # spring green
]

_POINT_COLOR_NAMES = [
    "magenta",
    "green",
    "orange",
    "cyan",
    "yellow",
    "red",
    "purple",
    "spring green",
]


class _PlotPointsInvocation(BaseToolInvocation[dict, dict]):
    def __init__(self, params: dict, image_path: str, grid_b64: str) -> None:
        super().__init__(params)
        self.image_path = image_path
        self.grid_b64 = grid_b64

    def get_description(self) -> str:
        points = self.params.get("points", [])
        return f"Plotting {len(points)} points on the image"

    async def execute(self) -> ToolResult:
        import numpy as np
        from PIL import Image, ImageDraw

        points = self.params.get("points", [])
        if not points:
            return ToolResult(error="No points provided")

        scale_x, scale_y, img_w, img_h = _get_norm_scale(self.image_path)

        # Load base image — grid overlay if available, else original
        if self.grid_b64:
            grid_bytes = base64.b64decode(self.grid_b64)
            grid_img = Image.open(BytesIO(grid_bytes)).convert("RGBA")
        else:
            grid_img = Image.open(self.image_path).convert("RGBA")
        gw, gh = grid_img.size
        grid_sx = gw / img_w
        grid_sy = gh / img_h

        draw = ImageDraw.Draw(grid_img)

        # Load original for edge detection
        orig = Image.open(self.image_path).convert("RGB")
        orig_arr = np.array(orig)
        edge_map = _compute_edge_map(orig_arr)

        on_edge_threshold = 40.0
        near_edge_dist = 5.0  # units in 1000-space

        feedback_lines = []
        sf = scale_factor(gw, gh)
        r = max(4, int(6 * sf))
        cross_ext = max(3, int(4 * sf))
        outline_w = max(1, int(2 * sf))

        for i, pt in enumerate(points):
            x = pt.get("x", 0)
            y = pt.get("y", 0)
            label = pt.get("label", "")

            color = _POINT_COLORS[i % len(_POINT_COLORS)]
            color_name = _POINT_COLOR_NAMES[i % len(_POINT_COLORS)]

            px_x = int(round(x * scale_x))
            px_y = int(round(y * scale_y))

            gx = int(round(px_x * grid_sx))
            gy = int(round(px_y * grid_sy))

            draw.ellipse(
                [gx - r, gy - r, gx + r, gy + r],
                fill=color + (120,),
                outline=color + (200,),
                width=outline_w,
            )
            draw.line([(gx - r - cross_ext, gy), (gx + r + cross_ext, gy)], fill=color + (200,), width=1)
            draw.line([(gx, gy - r - cross_ext), (gx, gy + r + cross_ext)], fill=color + (200,), width=1)

            clamped_x = min(max(px_x, 0), img_w - 1)
            clamped_y = min(max(px_y, 0), img_h - 1)
            edge_strength = float(edge_map[clamped_y, clamped_x])
            pixel = orig_arr[clamped_y, clamped_x]
            brightness = int(pixel.mean())

            if edge_strength > on_edge_threshold:
                if brightness < 100:
                    feedback_lines.append(
                        f"[{color_name}] ({x}, {y}){f' {label}' if label else ''} — "
                        f"ON WALL EDGE (edge={edge_strength:.0f}, brightness={brightness})."
                    )
                else:
                    feedback_lines.append(
                        f"[{color_name}] ({x}, {y}){f' {label}' if label else ''} — "
                        f"ON EDGE (edge={edge_strength:.0f}, brightness={brightness})."
                    )
            else:
                nearest = _find_nearest_edge(edge_map, clamped_x, clamped_y)
                if nearest:
                    nx, ny, dist = nearest
                    near_norm_x = round(nx / scale_x, 1)
                    near_norm_y = round(ny / scale_y, 1)
                    dist_norm = round(dist / max(scale_x, scale_y), 1)

                    if dist_norm <= near_edge_dist:
                        feedback_lines.append(
                            f"[{color_name}] ({x}, {y}){f' {label}' if label else ''} — "
                            f"CLOSE to edge. Nearest edge at ({near_norm_x}, {near_norm_y}), "
                            f"{dist_norm} units away. Consider snapping."
                        )
                    else:
                        status = (
                            "EMPTY SPACE"
                            if brightness > 200
                            else f"OFF-EDGE (brightness={brightness})"
                        )
                        feedback_lines.append(
                            f"[{color_name}] ({x}, {y}){f' {label}' if label else ''} — "
                            f"WARNING: {status}, NOT on a wall/edge. "
                            f"Nearest edge at ({near_norm_x}, {near_norm_y}), "
                            f"{dist_norm} units away."
                        )
                else:
                    feedback_lines.append(
                        f"[{color_name}] ({x}, {y}){f' {label}' if label else ''} — "
                        f"WARNING: No edges found within 50 units."
                    )

        result_img = grid_img.convert("RGB")
        buf = BytesIO()
        result_img.save(buf, format="PNG")
        result_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        feedback_text = "\n".join(feedback_lines)

        content_blocks = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": result_b64,
                },
            },
            {"type": "text", "text": feedback_text},
        ]

        return ToolResult(
            output={"points": points, "feedback": feedback_text},
            content_blocks=content_blocks,
        )


class PlotPointTool(BaseDeclarativeTool):
    """Plot multiple points on a grid-overlaid image to verify coordinates.

    Each point gets a distinct color and feedback about edge proximity:
    ON_WALL_EDGE, ON_EDGE, CLOSE_TO_EDGE, OFF_EDGE, or EMPTY_SPACE.
    """

    def __init__(self) -> None:
        schema: ToolSchema = {
            "type": "function",
            "function": {
                "name": "plot_points",
                "description": (
                    "Plot points on the image to verify coordinates. "
                    "Each gets a color + feedback (on edge, empty, nearest edge). "
                    "Plot key features to calibrate spatial understanding."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "points": {
                            "type": "array",
                            "description": "List of points to plot",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "x": {
                                        "type": "number",
                                        "description": "X coordinate in 1000-normalized space",
                                    },
                                    "y": {
                                        "type": "number",
                                        "description": "Y coordinate in 1000-normalized space",
                                    },
                                    "label": {
                                        "type": "string",
                                        "description": "Label for this point",
                                    },
                                },
                                "required": ["x", "y"],
                            },
                        },
                    },
                    "required": ["points"],
                },
            },
        }
        super().__init__("plot_points", schema)
        self._image_path: str = ""
        self._grid_b64: str = ""

    def set_image_path(self, path: str) -> None:
        self._image_path = path

    def set_grid_b64(self, grid_b64: str) -> None:
        self._grid_b64 = grid_b64

    async def build(self, params: dict[str, Any]) -> _PlotPointsInvocation:
        return _PlotPointsInvocation(params, self._image_path, self._grid_b64)
