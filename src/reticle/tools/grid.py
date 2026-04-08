"""Grid overlay rendering for spatial coordinate reference.

Renders a coordinate grid on images in 1000-normalized space so LLM agents
can reason about spatial positions. Red vertical lines = x-axis, blue
horizontal lines = y-axis. Labeled every 200 units, minor lines every 100.
"""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any


def _get_norm_scale(image_path: str) -> tuple[float, float, int, int]:
    """Return (scale_x, scale_y, img_w, img_h) for denormalizing 1000-coords to pixels.

    The 0-1000 space maps independently per axis:
      x: 0-1000 -> 0-width
      y: 0-1000 -> 0-height
    """
    from PIL import Image

    img = Image.open(image_path)
    w, h = img.size
    return w / 1000.0, h / 1000.0, w, h


def denorm_polygon(
    poly: list[list[float]],
    scale: float | tuple[float, float],
) -> list[tuple[int, int]]:
    """Denormalize a 1000-space polygon to pixel coordinates."""
    if isinstance(scale, tuple):
        sx, sy = scale
    else:
        sx = sy = scale
    return [(int(round(x * sx)), int(round(y * sy))) for x, y in poly]


def scale_factor(w: int, h: int) -> float:
    """Compute a drawing scale factor based on image dimensions.

    Returns a multiplier relative to a 768px baseline so that all visual
    elements (fonts, dots, line widths) scale proportionally with image size.
    """
    return max(min(w, h), 400) / 768.0


def _get_font(size: int) -> Any:
    """Get a font at the given pixel size, with platform fallback."""
    from PIL import ImageFont

    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except (OSError, AttributeError):
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except OSError:
            return ImageFont.load_default()


def _get_grid_font(w: int, h: int) -> Any:
    """Get a bold font for grid coordinate labels, scaled to image size."""
    size = max(10, int(16 * scale_factor(w, h)))
    return _get_font(size)


def draw_label(
    draw: Any,
    xy: tuple[int, int],
    text: str,
    fill: tuple[int, ...],
    font: Any,
    bg: tuple[int, ...] = (0, 0, 0, 210),
    pad: int = 2,
) -> None:
    """Draw text on a solid background rectangle for max contrast."""
    x, y = xy
    bbox = font.getbbox(text) if hasattr(font, "getbbox") else (0, 0, len(text) * 10, 16)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.rectangle([x - pad, y - pad, x + tw + pad, y + th + pad], fill=bg)
    draw.text((x, y), text, fill=fill, font=font)


def render_grid_overlay(image_path: str, max_dim: int = 768) -> str:
    """Render a coordinate grid directly on the image in 1000-normalized space.

    Red vertical lines = x-axis, blue horizontal lines = y-axis.
    Thick + labeled every 200 units, thin lines every 100.
    No padding — grid is drawn on the image so pixel positions match
    the model's internal coordinate system.
    Returns base64-encoded PNG.
    """
    from PIL import Image, ImageDraw

    img = Image.open(image_path).convert("RGBA")
    orig_w, orig_h = img.size
    w, h = orig_w, orig_h
    if max(w, h) > max_dim:
        ratio = max_dim / max(w, h)
        w, h = int(w * ratio), int(h * ratio)
        img = img.resize((w, h), Image.LANCZOS)  # type: ignore[attr-defined]

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _get_grid_font(w, h)

    grid_step = 100
    x_color_major = (255, 50, 50, 160)
    x_color_minor = (255, 50, 50, 50)
    x_label_color = (255, 120, 120, 255)
    y_color_major = (80, 150, 255, 160)
    y_color_minor = (80, 150, 255, 50)
    y_label_color = (120, 180, 255, 255)

    for norm_x in range(grid_step, 1000, grid_step):
        px_x = (norm_x / 1000.0) * w
        if norm_x % 200 == 0:
            draw.line([(px_x, 0), (px_x, h)], fill=x_color_major, width=2)
            label = str(norm_x)
            draw_label(draw, (int(px_x) + 3, 4), label, x_label_color, font)
            draw_label(draw, (int(px_x) + 3, h - 22), label, x_label_color, font)
        else:
            draw.line([(px_x, 0), (px_x, h)], fill=x_color_minor, width=1)

    for norm_y in range(grid_step, 1000, grid_step):
        px_y = (norm_y / 1000.0) * h
        if norm_y % 200 == 0:
            draw.line([(0, px_y), (w, px_y)], fill=y_color_major, width=2)
            label = str(norm_y)
            draw_label(draw, (4, int(px_y) + 3), label, y_label_color, font)
            draw_label(draw, (w - 36, int(px_y) + 3), label, y_label_color, font)
        else:
            draw.line([(0, px_y), (w, px_y)], fill=y_color_minor, width=1)

    composite = Image.alpha_composite(img, overlay).convert("RGB")
    buf = BytesIO()
    composite.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
