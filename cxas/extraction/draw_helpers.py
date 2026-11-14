import numpy as np
from PIL import Image, ImageDraw


def as_pil(image):
    """PIL draws on images only; the segmentor passes the (normalised) model input as an array."""
    if isinstance(image, Image.Image):
        return image
    a = np.asarray(image, dtype=np.float32).squeeze()
    a = (a - a.min()) / max(float(a.max() - a.min()), 1e-6) * 255
    return Image.fromarray(a.astype(np.uint8)).convert("RGB")


def draw_point(image, point, color, size):
    image = as_pil(image)
    draw = ImageDraw.Draw(image)
    draw.ellipse(
        [
            (point[0] - size / 2, point[1] - size / 2),
            (point[0] + size / 2, point[1] + size / 2),
        ],
        fill=color,
    )
    return image


def draw_line(image, p1, p2, color, width):
    image = as_pil(image)
    draw = ImageDraw.Draw(image)
    draw.line((p1, p2), fill=color, width=width)
    return image
