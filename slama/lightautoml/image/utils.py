"""Image utils."""

from PIL import Image


def pil_loader(path: str) -> Image:
    """Load image from pathes.

    Args:
        path: Image path.

    Returns:
        Loaded PIL Image in rgb.

    """
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")
