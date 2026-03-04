"""
Pixel art edge processor: detect outer edge, then take N pixels inward (band).
Two separate operations:
- Black outline: in band, near-black pixels -> pitch black.
- Remove transparent: in band, pixels with alpha <= threshold -> fully transparent.
"""
import numpy as np
from PIL import Image

# Alpha >= this is "content" for edge detection (internal).
_CONTENT_ALPHA = 128


def _get_content_mask(alpha: np.ndarray) -> np.ndarray:
    """True where image has visible content (non-transparent)."""
    return alpha >= _CONTENT_ALPHA


def _get_edge_pixels(content: np.ndarray) -> np.ndarray:
    """Pixels that are in content and have at least one non-content 4-neighbor (outer boundary)."""
    h, w = content.shape
    edge = np.zeros_like(content, dtype=bool)
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny = np.clip(np.arange(h).reshape(-1, 1) + dy, 0, h - 1)
        nx = np.clip(np.arange(w).reshape(1, -1) + dx, 0, w - 1)
        neighbor = content[ny, nx]
        edge = edge | (content & ~neighbor)
    return edge


def _band_mask_from_edges(content: np.ndarray, edge: np.ndarray, num_pixels: int) -> np.ndarray:
    """
    Band = pixels within num_pixels steps INWARD from the edge (only through content).
    Implemented by BFS: start from edge, add content pixels adjacent to current band, repeat.
    """
    if num_pixels <= 0:
        return np.zeros_like(content, dtype=bool)
    h, w = content.shape
    # Current band (frontier = last added layer)
    band = np.array(edge, dtype=bool, copy=True)
    frontier = np.array(edge, dtype=bool, copy=True)
    for _ in range(num_pixels - 1):
        # Pixels that are 4-adjacent to frontier: (i,j) has neighbor in frontier
        f_up = np.vstack([np.zeros((1, w), dtype=bool), frontier[:-1, :]])
        f_dn = np.vstack([frontier[1:, :], np.zeros((1, w), dtype=bool)])
        f_lt = np.hstack([np.zeros((h, 1), dtype=bool), frontier[:, :-1]])
        f_rt = np.hstack([frontier[:, 1:], np.zeros((h, 1), dtype=bool)])
        adjacent_to_frontier = f_up | f_dn | f_lt | f_rt
        # Next layer: in content, not yet in band, adjacent to current frontier
        next_layer = (content & ~band & adjacent_to_frontier)
        band = band | next_layer
        frontier = next_layer
        if not np.any(frontier):
            break
    return band


def _get_edge_band(image: Image.Image, num_pixels: int):
    """Load RGBA, compute content, edge, and band (N pixels inward from edge). Returns (rgba, band)."""
    img = image.convert("RGBA")
    rgba = np.asarray(img, dtype=np.uint8)
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("Image must be RGBA")
    a = rgba[:, :, 3]
    content = _get_content_mask(a)
    # Edge: outer boundary of content (pixels that touch non-content)
    if not np.any(content):
        band = np.zeros_like(content, dtype=bool)
        return rgba, band
    if np.all(content):
        # Fully opaque: edge = image rectangle border
        h, w = content.shape
        edge = np.zeros_like(content, dtype=bool)
        edge[0, :] = True
        edge[-1, :] = True
        edge[:, 0] = True
        edge[:, -1] = True
    else:
        edge = _get_edge_pixels(content)
    band = _band_mask_from_edges(content, edge, num_pixels)
    return rgba, band


def process_black_outline(
    image: Image.Image,
    num_pixels: int,
    black_threshold: int,
) -> Image.Image:
    """
    From the outer edge, go inward `num_pixels`. In that band only:
    pixels with max(R,G,B) <= black_threshold become pitch black (alpha unchanged).
    Pixel-perfect, no resampling.
    """
    rgba, band = _get_edge_band(image, num_pixels)
    out = np.array(rgba, dtype=np.uint8, copy=True)
    r, g, b = out[:, :, 0], out[:, :, 1], out[:, :, 2]
    max_rgb = np.maximum(np.maximum(r, g), b)
    near_black = (max_rgb <= black_threshold) & band
    out[near_black, 0] = 0
    out[near_black, 1] = 0
    out[near_black, 2] = 0
    return Image.fromarray(out, "RGBA")


def process_remove_transparent(
    image: Image.Image,
    num_pixels: int,
    transparency_threshold: int,
) -> Image.Image:
    """
    From the outer edge, go inward `num_pixels`. In that band only:
    pixels with alpha <= transparency_threshold are made fully transparent (removed).
    Pixel-perfect, no resampling.
    """
    rgba, band = _get_edge_band(image, num_pixels)
    out = np.array(rgba, dtype=np.uint8, copy=True)
    a = out[:, :, 3]
    too_transparent = (a <= transparency_threshold) & band
    out[too_transparent, 0] = 0
    out[too_transparent, 1] = 0
    out[too_transparent, 2] = 0
    out[too_transparent, 3] = 0
    return Image.fromarray(out, "RGBA")


def process_remove_transparent_all(
    image: Image.Image,
    transparency_threshold: int,
) -> Image.Image:
    """
    Check ALL pixels in the image (no edge band).
    Pixels with alpha <= transparency_threshold are made fully transparent (removed).
    Pixel-perfect, no resampling.
    """
    img = image.convert("RGBA")
    rgba = np.asarray(img, dtype=np.uint8)
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("Image must be RGBA")
    out = np.array(rgba, dtype=np.uint8, copy=True)
    a = out[:, :, 3]
    too_transparent = a <= transparency_threshold
    out[too_transparent, 0] = 0
    out[too_transparent, 1] = 0
    out[too_transparent, 2] = 0
    out[too_transparent, 3] = 0
    return Image.fromarray(out, "RGBA")
