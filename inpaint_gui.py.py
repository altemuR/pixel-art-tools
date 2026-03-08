"""
Batch Content-Aware Inpainting GUI
====================================
Content-aware fill using OpenCV (built-in, fast) or LaMa AI (optional, high quality).
Also supports automatic transparent-hole detection and filling.

Requirements — CORE (always needed):
    pip install Pillow opencv-python tkinterdnd2 numpy scipy

Requirements — AI backend (optional, much better quality):
    pip install simple-lama-inpainting

Usage:
    python inpaint_gui.py
"""

import os
import re
import threading
from pathlib import Path
from tkinter import *  # pyright: ignore[reportWildcardImportFromLibrary]
from tkinter import ttk, filedialog, messagebox

try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    DND_AVAILABLE = True
except ImportError:
    TkinterDnD = None  # type: ignore[assignment]
    DND_FILES = None   # type: ignore[assignment]
    DND_AVAILABLE = False

import numpy as np
from PIL import Image, ImageTk

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore[assignment]
    CV2_AVAILABLE = False


# ── Palette ───────────────────────────────────────────────────────────────────
BG       = "#0d0d11"
PANEL    = "#16161f"
SURFACE  = "#1e1e2e"
SURFACE2 = "#252535"
BORDER   = "#2a2a40"
ACCENT   = "#6e57f8"
ACCENT2  = "#a855f7"
TEXT     = "#e2e0ff"
SUBTEXT  = "#6b6893"
SUCCESS  = "#34d399"
ERROR    = "#f87171"
WARN     = "#fbbf24"

FONT_HEAD  = ("Georgia", 20, "bold")
FONT_LABEL = ("Courier New", 9, "bold")
FONT_BODY  = ("Courier New", 10)
FONT_SMALL = ("Courier New", 9)
FONT_BTN   = ("Courier New", 10, "bold")
FONT_RUN   = ("Courier New", 12, "bold")


# ══════════════════════════════════════════════════════════════════════════════
#  HOLE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def _get_alpha(pil_img: Image.Image) -> "np.ndarray | None":
    """
    Return the alpha channel as a uint8 numpy array, or None if the image
    has no transparency at all.  Handles RGBA, LA, PA, P+transparency.
    """
    mode = pil_img.mode
    if mode == "RGBA":
        return np.array(pil_img)[:, :, 3]
    if mode == "LA":
        return np.array(pil_img)[:, :, 1]
    if mode in ("P", "PA"):
        # Convert palette image to RGBA so we get a real alpha channel
        rgba = np.array(pil_img.convert("RGBA"))
        return rgba[:, :, 3]
    # RGB / L / etc. → no alpha
    return None


def detect_interior_holes(pil_img: Image.Image) -> "np.ndarray | None":
    """
    Detect transparent pixels that are *interior* holes (not background).

    Strategy:
      1. Extract the alpha channel (handles RGBA, LA, P with transparency).
      2. Build a binary map: 255 = transparent, 0 = opaque.
      3. Flood-fill with scipy from the four image borders to mark every
         transparent pixel reachable from outside → those are background.
      4. Any transparent pixel NOT reached = interior hole.

    Returns uint8 ndarray (255 = hole, 0 = keep), or None if no holes.
    """
    alpha = _get_alpha(pil_img)
    if alpha is None:
        return None

    h, w = alpha.shape
    # 255 where transparent (alpha < 128), 0 where opaque
    trans = (alpha < 128).astype(np.uint8)

    if trans.sum() == 0:
        return None  # fully opaque

    # --- flood fill exterior using scipy (avoids OpenCV ff_mask sharing bug) ---
    from scipy.ndimage import label as scipy_label

    # Label connected components of transparent pixels
    structure = np.ones((3, 3), dtype=np.uint8)  # 8-connectivity
    labeled, n_labels = scipy_label(trans, structure=structure)

    if n_labels == 0:
        return None

    # A component is "exterior" if it touches any image border
    border_labels = set()
    border_labels.update(labeled[0,   :].tolist())   # top
    border_labels.update(labeled[h-1, :].tolist())   # bottom
    border_labels.update(labeled[:,   0].tolist())   # left
    border_labels.update(labeled[:, w-1].tolist())   # right
    border_labels.discard(0)  # 0 = opaque pixels, not a component

    # Interior holes = transparent components that never touch the border
    interior = np.zeros((h, w), dtype=np.uint8)
    for lbl in range(1, n_labels + 1):
        if lbl not in border_labels:
            interior[labeled == lbl] = 255

    if interior.sum() == 0:
        return None

    return interior


def build_hole_mask(pil_img: Image.Image, dilation: int = 0) -> "Image.Image | None":
    """
    Return a PIL grayscale mask (white = fill, black = keep) for interior
    transparent holes, or None if no holes are found.
    """
    hole_arr = detect_interior_holes(pil_img)
    if hole_arr is None:
        return None

    if dilation > 0 and CV2_AVAILABLE:
        assert cv2 is not None
        kernel   = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilation * 2 + 1, dilation * 2 + 1))
        hole_arr = cv2.dilate(hole_arr, kernel)

    return Image.fromarray(hole_arr)


# ══════════════════════════════════════════════════════════════════════════════
#  PIXEL-ART SAFE NEAREST-NEIGHBOUR HOLE FILL
# ══════════════════════════════════════════════════════════════════════════════

def fill_holes_nearest_neighbour(pil_img: Image.Image,
                                 hole_mask_arr: np.ndarray,
                                 exclude_dark: bool = False,
                                 dark_threshold: int = 30
                                 ) -> "tuple[Image.Image, int]":
    """
    Fill hole pixels by copying the RGBA value of the nearest *eligible* opaque pixel.

    eligible = opaque  AND  (luminance >= dark_threshold  OR  exclude_dark is False)

    If a hole pixel has no eligible neighbour (e.g. it is completely surrounded
    by black outlines) the absolute nearest opaque pixel is used as a fallback
    so no hole is ever left unfilled.

    Returns (result_image_RGBA, filled_pixel_count).
    """
    from scipy.ndimage import distance_transform_edt

    rgba = np.array(pil_img.convert("RGBA"), dtype=np.uint8)

    hole_pixels = hole_mask_arr > 127          # (h, w) bool
    opaque      = rgba[:, :, 3] >= 128         # (h, w) bool

    if not opaque.any():
        return pil_img.convert("RGBA"), 0

    ys, xs = np.where(hole_pixels)
    if len(ys) == 0:
        return pil_img.convert("RGBA"), 0

    # ── Build the "eligible" sampling mask ────────────────────────────────────
    if exclude_dark and dark_threshold > 0:
        r = rgba[:, :, 0].astype(np.float32)
        g = rgba[:, :, 1].astype(np.float32)
        b = rgba[:, :, 2].astype(np.float32)
        lum = 0.299 * r + 0.587 * g + 0.114 * b   # ITU-R BT.601 luminance
        bright_enough = lum >= dark_threshold
        eligible = opaque & bright_enough
    else:
        eligible = opaque

    result = rgba.copy()

    if eligible.any():
        # Primary pass: nearest eligible neighbour
        _, (ny_elig, nx_elig) = distance_transform_edt(~eligible, return_indices=True)
        ny_primary = ny_elig[ys, xs]
        nx_primary = nx_elig[ys, xs]

        if exclude_dark and dark_threshold > 0:
            # Fallback pass: nearest opaque neighbour (for holes with no bright neighbour)
            _, (ny_any, nx_any) = distance_transform_edt(~opaque, return_indices=True)

            # A hole pixel needs fallback when its nearest eligible pixel is itself
            # non-eligible (distance_transform still returns *something*, but it may
            # point to a dark pixel if eligible is False everywhere nearby).
            # Check: is the colour at the primary source actually eligible?
            primary_lum = (0.299 * rgba[ny_primary, nx_primary, 0].astype(np.float32)
                         + 0.587 * rgba[ny_primary, nx_primary, 1].astype(np.float32)
                         + 0.114 * rgba[ny_primary, nx_primary, 2].astype(np.float32))
            needs_fallback = primary_lum < dark_threshold

            ny_chosen = np.where(needs_fallback, ny_any[ys, xs], ny_primary)
            nx_chosen = np.where(needs_fallback, nx_any[ys, xs], nx_primary)

            n_fallback = int(needs_fallback.sum())
            if n_fallback:
                pass  # reported in the caller log
        else:
            ny_chosen = ny_primary
            nx_chosen = nx_primary
            n_fallback = 0
    else:
        # No eligible pixels at all → fall back to plain nearest opaque
        _, (ny_any, nx_any) = distance_transform_edt(~opaque, return_indices=True)
        ny_chosen  = ny_any[ys, xs]
        nx_chosen  = nx_any[ys, xs]
        n_fallback = len(ys)

    result[ys, xs]    = rgba[ny_chosen, nx_chosen]
    result[ys, xs, 3] = 255   # make filled pixels fully opaque

    return Image.fromarray(result, "RGBA"), int(len(ys))


# ══════════════════════════════════════════════════════════════════════════════
#  INPAINTING BACKENDS  (mask-based, for when a user mask is supplied)
# ══════════════════════════════════════════════════════════════════════════════

def inpaint_opencv(pil_img: Image.Image, pil_mask: Image.Image,
                   method: str, radius: int) -> Image.Image:
    """
    Content-aware inpainting using OpenCV.
    method : 'telea' or 'ns'
    mask   : white = fill region, black = keep region
    """
    if not CV2_AVAILABLE:
        raise RuntimeError(
            "opencv-python is not installed.\nRun: pip install opencv-python")
    assert cv2 is not None

    has_alpha = pil_img.mode == "RGBA"
    rgba_arr  = np.array(pil_img.convert("RGBA"))
    alpha_ch  = rgba_arr[:, :, 3]
    src_bgr   = cv2.cvtColor(rgba_arr[:, :, :3], cv2.COLOR_RGB2BGR)

    mask_np = np.array(pil_mask.convert("L"))
    if mask_np.shape[:2] != src_bgr.shape[:2]:
        mask_np = cv2.resize(mask_np,
                             (src_bgr.shape[1], src_bgr.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
    _, mask_bin = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)

    # Pre-fill transparent pixels with nearest opaque neighbour so OpenCV
    # has valid RGB data to sample at the boundary.
    if has_alpha:
        trans = alpha_ch < 128
        if trans.any():
            from scipy.ndimage import distance_transform_edt
            opaque = ~trans
            _, (ny, nx) = distance_transform_edt(trans, return_indices=True)
            # Vectorised nearest-neighbour copy
            ys, xs = np.where(trans)
            src_bgr_filled = src_bgr.copy()
            src_bgr_filled[ys, xs] = src_bgr[ny[ys, xs], nx[ys, xs]]
            src_bgr = src_bgr_filled

    flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    result_bgr = cv2.inpaint(src_bgr, mask_bin, inpaintRadius=radius, flags=flag)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    result_pil = Image.fromarray(result_rgb)

    if has_alpha:
        result_pil = result_pil.convert("RGBA")
        new_alpha = alpha_ch.copy()
        hole_px   = mask_bin > 127
        new_alpha[hole_px] = 255
        result_pil.putalpha(Image.fromarray(new_alpha))

    return result_pil


def inpaint_lama(pil_img: Image.Image, pil_mask: Image.Image) -> Image.Image:
    """
    Content-aware inpainting using LaMa (large-mask model).
    Requires:  pip install simple-lama-inpainting
    """
    try:
        from simple_lama_inpainting import SimpleLama
    except ImportError:
        raise RuntimeError(
            "simple-lama-inpainting not installed.\n"
            "Run:  pip install simple-lama-inpainting\n"
            "Model weights (~200 MB) download automatically on first use.")

    lama = SimpleLama()
    rgb_img   = pil_img.convert("RGB")
    gray_mask = pil_mask.convert("L")
    if gray_mask.size != rgb_img.size:
        gray_mask = gray_mask.resize(rgb_img.size, Image.Resampling.NEAREST)

    result = lama(rgb_img, gray_mask)

    if pil_img.mode == "RGBA":
        result = result.convert("RGBA")
        orig_alpha = np.array(pil_img.split()[3])
        new_alpha  = orig_alpha.copy()
        hole_px    = np.array(gray_mask) > 127
        if hole_px.shape == new_alpha.shape:
            new_alpha[hole_px] = 255
        result.putalpha(Image.fromarray(new_alpha))

    return result



# ══════════════════════════════════════════════════════════════════════════════
#  OUTER EDGE TRIMMING
# ══════════════════════════════════════════════════════════════════════════════

def trim_outer_edge(pil_img: Image.Image,
                    min_length_pct: float = 5.0,
                    straightness_threshold: float = 0.92,
                    trim_px: int = 1,
                    ) -> "tuple[Image.Image, int, int]":
    """
    Remove the outermost layer of opaque pixels that form long, jagged edges.

    These are the pixels that sit right at the boundary between opaque content
    and transparency — the outer edge ring.  Setting them transparent effectively
    trims a 1-pixel fringe of jaggies without touching any interior pixels.

    Only opaque pixels that directly border the transparent background are ever
    modified.  Interior pixels (surrounded by other opaque pixels on all sides)
    are guaranteed to be untouched.

    Parameters
    ----------
    min_length_pct        : minimum contour arc-length as % of image diagonal.
                            Short contours (dots, tiny artifacts) are skipped.
    straightness_threshold: arc/chord score cutoff.  Contours already close to
                            a straight line (score >= threshold) are left alone.
                            Lower = only trim very jagged edges.
                            Higher = also trim mildly jagged edges.
    trim_px               : how many outer-edge pixel layers to remove (1–3).

    Returns
    -------
    (result_image_RGBA, contours_trimmed, pixels_removed)
    """
    if not CV2_AVAILABLE:
        raise RuntimeError("opencv-python is required for edge trimming.\n"
                           "Run: pip install opencv-python")
    assert cv2 is not None

    rgba  = np.array(pil_img.convert("RGBA"), dtype=np.uint8)
    h, w  = rgba.shape[:2]
    alpha = rgba[:, :, 3]

    # ── 1. Outer-edge mask: opaque pixels that touch ≥1 transparent px ───────
    opaque_mask = (alpha >= 128).astype(np.uint8) * 255
    trans_mask  = (alpha  < 128).astype(np.uint8) * 255
    kernel_3    = np.ones((3, 3), np.uint8)
    trans_dilat = cv2.dilate(trans_mask, kernel_3, iterations=1)
    edge_mask   = cv2.bitwise_and(opaque_mask, trans_dilat)   # 0/255

    if edge_mask.max() == 0:
        return pil_img.convert("RGBA"), 0, 0

    # ── 2. Find contours on the edge mask (every pixel) ──────────────────────
    contours, _ = cv2.findContours(edge_mask, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        return pil_img.convert("RGBA"), 0, 0

    # ── 3. Filter: keep long + sufficiently jagged contours ──────────────────
    diagonal    = (h ** 2 + w ** 2) ** 0.5
    min_arc_len = diagonal * (min_length_pct / 100.0)

    kept = []
    for cnt in contours:
        arc = cv2.arcLength(cnt, closed=False)
        if arc < min_arc_len:
            continue
        pts = cnt[:, 0, :]
        if len(pts) < 4:
            continue
        start = pts[0].astype(float)
        end   = pts[-1].astype(float)
        chord = max(float(np.linalg.norm(end - start)), 1.0)
        ratio = arc / chord                        # 1.0 = straight, >1 = jagged
        straightness_score = 1.0 / max(ratio, 1.0)
        if straightness_score >= straightness_threshold:
            continue   # already straight — leave it alone
        kept.append(cnt)

    if not kept:
        return pil_img.convert("RGBA"), 0, 0

    # ── 4. Build a trim mask covering only the kept contour pixels ────────────
    trim_mask = np.zeros((h, w), dtype=np.uint8)
    for cnt in kept:
        cv2.drawContours(trim_mask, [cnt], -1, 255, thickness=trim_px)

    # Safety: only erase pixels that are genuinely on the outer edge.
    # AND with the full edge_mask so we can never touch interior pixels.
    trim_mask = cv2.bitwise_and(trim_mask, edge_mask)

    # ── 5. Set those pixels to fully transparent ──────────────────────────────
    result = rgba.copy()
    ey, ex = np.where(trim_mask > 0)
    result[ey, ex, 3] = 0          # alpha → 0 (transparent)
    # Clear RGB too so no fringe shows up under semi-transparent compositing
    result[ey, ex, :3] = 0

    return Image.fromarray(result, "RGBA"), len(kept), int(len(ey))




# ══════════════════════════════════════════════════════════════════════════════
#  OUTLINE REDRAW  (straighten the outer edge in-place)
# ══════════════════════════════════════════════════════════════════════════════

def redraw_outline(pil_img: Image.Image,
                   epsilon_pct: float = 2.0,
                   color_mode: str = "auto",
                   custom_color: "tuple[int,int,int] | None" = None,
                   thickness: int = 1,
                   min_length_pct: float = 2.0,
                   ) -> "tuple[Image.Image, int, int]":
    """
    Straighten the existing outer-edge outline in-place.

    The outline stays exactly where it is — on the outermost opaque pixels
    that touch the transparent background.  No pixels move inward.

    Algorithm per contour
    ---------------------
    1. Detect outer-edge pixels (opaque touching transparent).
    2. Simplify the contour with Douglas-Peucker → a clean polyline of
       key vertices that approximates the same path with fewer corners.
    3. For every original edge pixel, find the nearest point on the
       simplified polyline.  Recolour that pixel with either the dominant
       colour sampled from the contour itself (auto) or a custom colour.
       The pixel's *position* never changes — only its colour is updated.
    4. Additionally draw the simplified polyline segments directly onto
       the edge mask region (clipped to existing opaque pixels), filling
       any small gaps left after trimming.

    Result: the outline is in the same place but the colour is applied in
    a geometrically smooth pattern rather than a jagged staircase.

    Parameters
    ----------
    epsilon_pct    : Douglas-Peucker tolerance as % of contour arc-length.
    color_mode     : "auto" → dominant colour of the contour's own pixels
                     "custom" → custom_color (R,G,B)
    custom_color   : (R,G,B) tuple for custom mode.
    thickness      : how many px wide to paint along the simplified path.
    min_length_pct : skip contours shorter than this % of image diagonal.

    Returns
    -------
    (result_image_RGBA, contours_processed, pixels_painted)
    """
    if not CV2_AVAILABLE:
        raise RuntimeError("opencv-python is required for outline redraw.\n"
                           "Run: pip install opencv-python")
    assert cv2 is not None

    rgba  = np.array(pil_img.convert("RGBA"), dtype=np.uint8)
    h, w  = rgba.shape[:2]
    alpha = rgba[:, :, 3]

    # ── 1. Outer-edge mask (same definition as trim step) ────────────────────
    opaque_mask = (alpha >= 128).astype(np.uint8) * 255
    trans_mask  = (alpha  < 128).astype(np.uint8) * 255
    kernel_3    = np.ones((3, 3), np.uint8)
    trans_dilat = cv2.dilate(trans_mask, kernel_3, iterations=1)
    edge_mask   = cv2.bitwise_and(opaque_mask, trans_dilat)   # 0/255

    if edge_mask.max() == 0:
        return pil_img.convert("RGBA"), 0, 0

    contours, _ = cv2.findContours(edge_mask, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        return pil_img.convert("RGBA"), 0, 0

    diagonal    = (h ** 2 + w ** 2) ** 0.5
    min_arc_len = diagonal * (min_length_pct / 100.0)

    result       = rgba.copy()
    total_pixels = 0
    drawn        = 0

    for cnt in contours:
        arc = cv2.arcLength(cnt, closed=False)
        if arc < min_arc_len:
            continue

        pts = cnt[:, 0, :].astype(np.float32)   # (N, 2) float x,y

        # ── 2. Sample contour's own dominant colour ───────────────────────────
        if color_mode == "custom" and custom_color is not None:
            draw_color = np.array([*custom_color, 255], dtype=np.uint8)
        else:
            from collections import Counter
            cys = pts[:, 1].clip(0, h - 1).astype(int)
            cxs = pts[:, 0].clip(0, w - 1).astype(int)
            own = [tuple(rgba[cy, cx, :3].tolist()) for cy, cx in zip(cys, cxs)]
            most_common = Counter(own).most_common(1)[0][0]
            draw_color  = np.array([*most_common, 255], dtype=np.uint8)

        # ── 3. Douglas-Peucker simplification ────────────────────────────────
        epsilon    = max(1.0, arc * (epsilon_pct / 100.0))
        simplified = cv2.approxPolyDP(cnt, epsilon=epsilon, closed=False)
        pts_s      = simplified[:, 0, :].astype(np.float32)   # (M, 2)

        if len(pts_s) < 2:
            continue

        # ── 4. Build the simplified polyline as a list of segments ────────────
        # segments: list of (A, B) where A,B are (x,y) float points
        segments = [(pts_s[i], pts_s[i + 1]) for i in range(len(pts_s) - 1)]

        # ── 5. For each original edge pixel, find nearest point on polyline ───
        # and recolour it.  The pixel stays in place — we only update its RGB.
        edge_ys, edge_xs = np.where(edge_mask > 0)
        if len(edge_ys) == 0:
            continue

        # Vectorised nearest-point-on-segment calculation
        # For each edge pixel P and each segment AB, project P onto AB.
        P = np.stack([edge_xs, edge_ys], axis=1).astype(np.float32)  # (K,2)

        min_dist_sq = np.full(len(P), np.inf, dtype=np.float32)

        for A, B in segments:
            AB   = B - A                          # (2,)
            AB_sq = float(np.dot(AB, AB))
            if AB_sq < 1e-6:
                # Degenerate segment — use point distance to A
                d_sq = np.sum((P - A) ** 2, axis=1)
            else:
                # t = dot(AP, AB) / |AB|^2, clamped to [0,1]
                AP = P - A                        # (K,2)
                t  = np.clip(AP @ AB / AB_sq, 0.0, 1.0)  # (K,)
                proj = A + np.outer(t, AB)        # (K,2) nearest point on seg
                d_sq = np.sum((P - proj) ** 2, axis=1)

            min_dist_sq = np.minimum(min_dist_sq, d_sq)

        # Keep only pixels whose nearest point on the simplified path is
        # within `thickness` pixels — this preserves the edge position.
        near = min_dist_sq <= (thickness + 0.5) ** 2
        ys_v = edge_ys[near]
        xs_v = edge_xs[near]

        result[ys_v, xs_v] = draw_color
        total_pixels += int(len(ys_v))
        drawn += 1

    return Image.fromarray(result, "RGBA"), drawn, total_pixels



# ══════════════════════════════════════════════════════════════════════════════
#  WIDGETS
# ══════════════════════════════════════════════════════════════════════════════

def styled_btn(parent, text, command, primary=False, small=False):
    bg  = ACCENT  if primary else SURFACE2
    abg = ACCENT2 if primary else BORDER
    fn  = FONT_BTN if not small else FONT_SMALL
    return Button(parent, text=text, command=command,
                  bg=bg, fg=TEXT, relief="flat",
                  activebackground=abg, activeforeground=TEXT,
                  font=fn, padx=10, pady=4, cursor="hand2", bd=0)


class SectionLabel(Frame):
    def __init__(self, master, text, **kw):
        super().__init__(master, bg=BG, **kw)
        Label(self, text=text, font=FONT_LABEL, bg=BG, fg=SUBTEXT).pack(side=LEFT)
        Frame(self, bg=BORDER, height=1).pack(side=LEFT, fill=X, expand=True, padx=(8, 0))


class TooltipLabel(Label):
    def __init__(self, master, tip, **kw):
        super().__init__(master, text=" ? ", font=FONT_SMALL,
                         bg=SURFACE2, fg=SUBTEXT, cursor="question_arrow", **kw)
        self._tip = tip
        self.bind("<Enter>", self._show)
        self.bind("<Leave>", self._hide)
        self._tw = None

    def _show(self, e):
        x = self.winfo_rootx() + 24
        y = self.winfo_rooty() + 24
        self._tw = Toplevel(self)
        self._tw.wm_overrideredirect(True)
        self._tw.wm_geometry(f"+{x}+{y}")
        Label(self._tw, text=self._tip, font=FONT_SMALL,
              bg=SURFACE2, fg=TEXT, relief="flat",
              padx=8, pady=6, wraplength=260, justify=LEFT).pack()

    def _hide(self, e):
        if self._tw:
            self._tw.destroy()
            self._tw = None


class DropZone(Frame):
    def __init__(self, master, label, on_files=None, on_file=None,
                 multi=True, accept=None, **kw):
        super().__init__(master, bg=SURFACE, bd=0,
                         highlightthickness=1, highlightbackground=BORDER, **kw)
        self.on_files = on_files
        self.on_file  = on_file
        self.multi    = multi
        self.accept   = accept or [("Image files",
                                    "*.png *.jpg *.jpeg *.bmp *.tiff *.webp")]
        self._paths   = []

        inner = Frame(self, bg=SURFACE)
        inner.place(relx=0.5, rely=0.5, anchor="center")

        self.icon      = Label(inner, text="⊕", font=("Courier New", 22), bg=SURFACE, fg=ACCENT)
        self.title_lbl = Label(inner, text=label, font=FONT_BTN, bg=SURFACE, fg=TEXT)
        self.hint      = Label(inner, text="drag & drop  or  click to browse",
                               font=FONT_SMALL, bg=SURFACE, fg=SUBTEXT)
        self.status    = Label(inner, text="", font=FONT_SMALL, bg=SURFACE, fg=SUCCESS,
                               wraplength=200, justify=CENTER)
        for w in [self.icon, self.title_lbl, self.hint, self.status]:
            w.pack()

        all_widgets = [self, inner, self.icon, self.title_lbl, self.hint, self.status]
        for w in all_widgets:
            w.bind("<Button-1>", self._browse)
            w.bind("<Enter>", lambda e: self._border(ACCENT))
            w.bind("<Leave>", lambda e: self._border(BORDER))

        if DND_AVAILABLE:
            self.drop_target_register(DND_FILES)  # type: ignore[attr-defined]
            self.dnd_bind("<<Drop>>",      self._drop)  # type: ignore[attr-defined]
            self.dnd_bind("<<DragEnter>>", lambda e: self._border(ACCENT))  # type: ignore[attr-defined]
            self.dnd_bind("<<DragLeave>>", lambda e: self._border(BORDER))  # type: ignore[attr-defined]

    def _border(self, c):
        self.configure(highlightbackground=c)

    def _drop(self, e):
        self._border(BORDER)
        paths = [p.strip("{}") for p in
                 re.findall(r'\{[^}]+\}|[^\s]+', e.data.strip())]
        self._apply(paths)

    def _browse(self, e=None):
        if self.multi:
            paths = filedialog.askopenfilenames(
                title="Select Images",
                filetypes=self.accept + [("All files", "*.*")])
            if paths:
                self._apply(list(paths))
        else:
            path = filedialog.askopenfilename(
                title="Select File",
                filetypes=self.accept + [("All files", "*.*")])
            if path:
                self._apply([path])

    def _apply(self, paths):
        valid = [p for p in paths if os.path.isfile(p)]
        if not valid:
            return
        if self.multi:
            self._paths = valid
            self.status.config(text=f"{len(valid)} file(s) loaded", fg=SUCCESS)
            if self.on_files:
                self.on_files(valid)
        else:
            self._paths = [valid[0]]
            self.status.config(text=Path(valid[0]).name, fg=SUCCESS)
            if self.on_file:
                self.on_file(valid[0])

    def clear(self):
        self._paths = []
        self.status.config(text="")

    def get_paths(self):
        return self._paths


class ThumbnailStrip(Frame):
    def __init__(self, master, **kw):
        super().__init__(master, bg=PANEL, **kw)
        self._refs = []
        canvas = Canvas(self, bg=PANEL, bd=0, highlightthickness=0, height=76)
        sb = ttk.Scrollbar(self, orient=HORIZONTAL, command=canvas.xview)
        canvas.configure(xscrollcommand=sb.set)
        sb.pack(side=BOTTOM, fill=X)
        canvas.pack(fill=BOTH, expand=True)
        self.inner = Frame(canvas, bg=PANEL)
        canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>",
                        lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    def set_images(self, paths):
        for w in self.inner.winfo_children():
            w.destroy()
        self._refs = []
        cap = 50
        for p in paths[:cap]:
            try:
                img = Image.open(p).convert("RGBA")
                img.thumbnail((66, 66))
                tk_img = ImageTk.PhotoImage(img)
                Label(self.inner, image=tk_img, bg=PANEL,
                      highlightbackground=BORDER,
                      highlightthickness=1, relief="flat").pack(
                          side=LEFT, padx=2, pady=5)
                self._refs.append(tk_img)
            except Exception:
                pass
        if len(paths) > cap:
            Label(self.inner, text=f"+{len(paths)-cap}\nmore",
                  font=FONT_SMALL, bg=PANEL, fg=SUBTEXT,
                  justify=CENTER).pack(side=LEFT, padx=8)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

class InpaintApp(TkinterDnD.Tk if DND_AVAILABLE else Tk):  # type: ignore[misc]

    def __init__(self):
        super().__init__()
        self.title("Batch Content-Aware Inpainter")
        self.configure(bg=BG)
        self.geometry("920x900")
        self.minsize(800, 750)

        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("TProgressbar", troughcolor=SURFACE, background=ACCENT,
                    bordercolor=BORDER, lightcolor=ACCENT, darkcolor=ACCENT2)

        self._image_paths   = []
        self._mask_path     = None
        self._output_dir    = StringVar(value="")
        self._backend       = StringVar(value="lama")
        self._radius        = IntVar(value=5)
        self._mask_dilation = IntVar(value=0)
        self._fill_holes      = BooleanVar(value=False)
        self._exclude_dark    = BooleanVar(value=False)
        self._dark_threshold  = IntVar(value=30)   # luminance 0-255
        # trim outer edge
        self._smooth_outlines          = BooleanVar(value=False)
        self._outline_min_len          = DoubleVar(value=5.0)
        self._outline_straightness_pct = IntVar(value=90)
        self._outline_trim_px          = IntVar(value=1)
        # redraw outline
        self._redraw_outline           = BooleanVar(value=False)
        self._redraw_epsilon_x10       = IntVar(value=20)   # ×10, so 20 = 2.0%
        self._redraw_color_mode        = StringVar(value="auto")
        self._redraw_custom_rgb        = (0, 0, 0)
        self._redraw_thickness         = IntVar(value=1)
        self._redraw_min_len           = DoubleVar(value=2.0)
        self._cancel_flag              = threading.Event()

        self._build_ui()
        self._on_engine_change()
        self.after(200, self._check_deps)

    # ── Dependency check ──────────────────────────────────────────────────────
    def _check_deps(self):
        if not CV2_AVAILABLE:
            self._log("⚠  opencv-python not found → pip install opencv-python", "warn")
        else:
            self._log("✓  OpenCV ready  (Telea & Navier-Stokes + hole detection)", "good")

        try:
            import scipy  # noqa
            self._log("✓  scipy ready  (hole detection + nearest-neighbour fill)", "good")
        except ImportError:
            self._log("⚠  scipy not found → pip install scipy  (required for Fill Holes)", "warn")

        if not DND_AVAILABLE:
            self._log("ℹ  Drag-and-drop disabled → pip install tkinterdnd2", "warn")

        try:
            import simple_lama_inpainting  # noqa
            self._log("✓  LaMa AI backend available", "good")
        except ImportError:
            self._log("ℹ  LaMa AI not installed → pip install simple-lama-inpainting")

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Header
        hdr = Frame(self, bg=BG)
        hdr.pack(fill=X, padx=28, pady=(20, 0))
        Label(hdr, text="CONTENT-AWARE", font=("Courier New", 9, "bold"),
              bg=BG, fg=ACCENT).pack(anchor=W)
        Label(hdr, text="Batch Inpainter", font=FONT_HEAD, bg=BG, fg=TEXT).pack(anchor=W)
        Label(hdr,
              text="mask-driven fill  ·  OpenCV Telea / Navier-Stokes  ·  LaMa AI  ·  hole fill",
              font=FONT_SMALL, bg=BG, fg=SUBTEXT).pack(anchor=W)
        Frame(self, bg=BORDER, height=1).pack(fill=X, padx=28, pady=(10, 0))

        # Two-column body
        body = Frame(self, bg=BG)
        body.pack(fill=BOTH, expand=True, padx=28, pady=14)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2, minsize=300)
        body.rowconfigure(0, weight=1)

        left_outer = Frame(body, bg=BG)
        right = Frame(body, bg=BG)
        left_outer.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        right.grid(row=0, column=1, sticky="nsew")

        # Scrollable left column
        left_canvas = Canvas(left_outer, bg=BG, bd=0, highlightthickness=0)
        left_vsb    = ttk.Scrollbar(left_outer, orient=VERTICAL,
                                    command=left_canvas.yview)
        left_canvas.configure(yscrollcommand=left_vsb.set)
        left_vsb.pack(side=RIGHT, fill=Y)
        left_canvas.pack(side=LEFT, fill=BOTH, expand=True)
        left = Frame(left_canvas, bg=BG)
        left_canvas.create_window((0, 0), window=left, anchor="nw")

        def _on_left_configure(e):
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))
            left_canvas.itemconfigure(1, width=left_canvas.winfo_width())
        left.bind("<Configure>", _on_left_configure)

        # Mouse-wheel scroll on left panel
        def _on_mousewheel(e):
            left_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        left_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # ── LEFT column ───────────────────────────────────────────────────────
        SectionLabel(left, "SOURCE IMAGES").pack(fill=X, pady=(0, 4))
        self.img_drop = DropZone(left, "Drop images here",
                                 multi=True, on_files=self._on_images, height=120)
        self.img_drop.pack(fill=X)

        img_btn_row = Frame(left, bg=BG)
        img_btn_row.pack(fill=X, pady=(4, 0))
        styled_btn(img_btn_row, "Add more…",
                   self.img_drop._browse, small=True).pack(side=LEFT)
        styled_btn(img_btn_row, "Clear",
                   self._clear_images, small=True).pack(side=LEFT, padx=(4, 0))
        self.img_count_lbl = Label(img_btn_row, text="", font=FONT_SMALL,
                                   bg=BG, fg=SUBTEXT)
        self.img_count_lbl.pack(side=RIGHT)

        self.thumb = ThumbnailStrip(left, height=86)
        self.thumb.pack(fill=X, pady=(6, 12))

        SectionLabel(left, "MASK IMAGE  (optional)").pack(fill=X, pady=(0, 4))
        Label(left, text="White = fill region     Black = keep region",
              font=FONT_SMALL, bg=BG, fg=SUBTEXT).pack(anchor=W, pady=(0, 2))
        Label(left, text="Leave empty to use Fill Holes mode only.",
              font=FONT_SMALL, bg=BG, fg=SUBTEXT).pack(anchor=W, pady=(0, 4))
        self.mask_drop = DropZone(left, "Drop mask here",
                                  multi=False, on_file=self._on_mask, height=90)
        self.mask_drop.pack(fill=X)

        # Mask clear button
        mask_btn_row = Frame(left, bg=BG)
        mask_btn_row.pack(fill=X, pady=(4, 0))
        styled_btn(mask_btn_row, "Clear mask",
                   self._clear_mask, small=True).pack(side=LEFT)

        mask_preview_row = Frame(left, bg=BG)
        mask_preview_row.pack(fill=X, pady=(6, 0))
        self.mask_thumb_lbl = Label(mask_preview_row, bg=BG)
        self.mask_thumb_lbl.pack(side=LEFT)
        self.mask_info_lbl  = Label(mask_preview_row, text="", font=FONT_SMALL,
                                    bg=BG, fg=SUBTEXT, justify=LEFT)
        self.mask_info_lbl.pack(side=LEFT, padx=10)

        # ── Fill Holes panel ─────────────────────────────────────────────────
        SectionLabel(left, "HOLE FILLING").pack(fill=X, pady=(14, 4))

        hole_card = Frame(left, bg=SURFACE2,
                          highlightthickness=1, highlightbackground=BORDER)
        hole_card.pack(fill=X)

        hole_top = Frame(hole_card, bg=SURFACE2)
        hole_top.pack(fill=X, padx=10, pady=(8, 4))

        self.fill_holes_chk = Checkbutton(
            hole_top,
            text="Fill Holes  —  auto-detect & fill transparent interior regions",
            variable=self._fill_holes,
            command=self._on_fill_holes_change,
            bg=SURFACE2, fg=TEXT, selectcolor=SURFACE,
            activebackground=SURFACE2, activeforeground=ACCENT,
            font=FONT_BODY, anchor=W)
        self.fill_holes_chk.pack(side=LEFT, fill=X, expand=True)
        TooltipLabel(
            hole_top,
            "Scans each image for transparent 'holes' that are completely\n"
            "surrounded by opaque pixels (i.e. interior gaps, not the\n"
            "image background) and fills them using content-aware sampling\n"
            "of the surrounding pixels.\n\n"
            "Works on RGBA/PNG images with an alpha channel.\n"
            "Can be combined with a mask, or used alone without a mask.\n\n"
            "Tip: Use PNG sources — JPEG has no transparency."
        ).pack(side=RIGHT, padx=6)

        self.hole_desc = Label(
            hole_card,
            text="Finds pixels enclosed inside a shape (alpha = 0 inside opaque edges)\n"
                 "and reconstructs them by sampling the surrounding texture.",
            font=FONT_SMALL, bg=SURFACE2, fg=SUBTEXT,
            justify=LEFT, wraplength=340)
        self.hole_desc.pack(anchor=W, padx=14, pady=(0, 6))

        # Exclude-dark row
        dark_row = Frame(hole_card, bg=SURFACE2)
        dark_row.pack(fill=X, padx=10, pady=(0, 4))

        self.excl_dark_chk = Checkbutton(
            dark_row,
            text="Exclude dark neighbors",
            variable=self._exclude_dark,
            command=self._on_exclude_dark_change,
            bg=SURFACE2, fg=TEXT, selectcolor=SURFACE,
            activebackground=SURFACE2, activeforeground=ACCENT,
            font=FONT_BODY, anchor=W)
        self.excl_dark_chk.pack(side=LEFT)
        TooltipLabel(
            dark_row,
            "When sampling the nearest opaque pixel to fill a hole,\n"
            "skip any pixel whose luminance is below the threshold.\n\n"
            "Luminance = 0.299·R + 0.587·G + 0.114·B\n\n"
            "Useful for pixel art where black outlines surround the holes\n"
            "and you want to sample the interior fill colour instead.\n\n"
            "If no bright-enough neighbour exists the absolute nearest\n"
            "opaque pixel is used as a fallback so holes are always filled."
        ).pack(side=LEFT, padx=4)

        # Threshold slider — shown/hidden via _on_exclude_dark_change
        self.dark_thresh_frame = Frame(hole_card, bg=SURFACE2)
        thresh_lbl_row = Frame(self.dark_thresh_frame, bg=SURFACE2)
        thresh_lbl_row.pack(fill=X, padx=14)
        Label(thresh_lbl_row, text="Min luminance threshold",
              font=FONT_LABEL, bg=SURFACE2, fg=SUBTEXT).pack(side=LEFT)
        Label(thresh_lbl_row, text="(0 = no exclusion, 255 = exclude all)",
              font=FONT_SMALL, bg=SURFACE2, fg=SUBTEXT).pack(side=LEFT, padx=(6, 0))
        Scale(self.dark_thresh_frame, from_=0, to=254, orient=HORIZONTAL,
              variable=self._dark_threshold,
              bg=SURFACE2, fg=TEXT, troughcolor=SURFACE,
              highlightthickness=0, activebackground=ACCENT,
              sliderrelief="flat", font=FONT_SMALL,
              showvalue=True).pack(fill=X, padx=14, pady=(0, 6))

        # ── Trim Outer Edge panel ─────────────────────────────────────────────
        SectionLabel(left, "TRIM OUTER EDGE").pack(fill=X, pady=(14, 4))

        outline_card = Frame(left, bg=SURFACE2,
                             highlightthickness=1, highlightbackground=BORDER)
        outline_card.pack(fill=X)

        # Enable checkbox
        outline_top = Frame(outline_card, bg=SURFACE2)
        outline_top.pack(fill=X, padx=10, pady=(8, 4))
        Checkbutton(
            outline_top,
            text="Trim Outer Edge  —  remove jagged outer-edge pixels",
            variable=self._smooth_outlines,
            command=self._on_smooth_outlines_change,
            bg=SURFACE2, fg=TEXT, selectcolor=SURFACE,
            activebackground=SURFACE2, activeforeground=ACCENT,
            font=FONT_BODY, anchor=W).pack(side=LEFT, fill=X, expand=True)
        TooltipLabel(
            outline_top,
            "Finds opaque pixels that sit right on the outer boundary\n"
            "(touching transparent background) and removes them by\n"
            "setting their alpha to 0.\n\n"
            "Only the outermost pixel ring is ever affected — interior\n"
            "pixels surrounded by other opaque pixels are guaranteed\n"
            "to be untouched.\n\n"
            "Short or already-straight edges are skipped. Use the\n"
            "sliders below to control which edges are trimmed."
        ).pack(side=RIGHT, padx=6)

        # Sub-controls frame (shown/hidden)
        self.outline_controls = Frame(outline_card, bg=SURFACE2)

        # Min length
        len_row = Frame(self.outline_controls, bg=SURFACE2)
        len_row.pack(fill=X, padx=10, pady=(4, 0))
        lh = Frame(len_row, bg=SURFACE2)
        lh.pack(fill=X)
        Label(lh, text="MIN EDGE LENGTH", font=FONT_LABEL,
              bg=SURFACE2, fg=SUBTEXT).pack(side=LEFT)
        TooltipLabel(lh,
            "Minimum length of an edge segment to be trimmed,\n"
            "as % of the image diagonal.\n\n"
            "Tiny edges (dots, small artifacts) are below this\n"
            "and are left untouched.\n"
            "Increase to ignore more small details."
        ).pack(side=LEFT, padx=4)
        Scale(len_row, from_=1, to=30, resolution=1, orient=HORIZONTAL,
              variable=self._outline_min_len,
              bg=SURFACE2, fg=TEXT, troughcolor=SURFACE,
              highlightthickness=0, activebackground=ACCENT,
              sliderrelief="flat", font=FONT_SMALL,
              showvalue=True).pack(fill=X)

        # Straightness filter
        str_row = Frame(self.outline_controls, bg=SURFACE2)
        str_row.pack(fill=X, padx=10, pady=(4, 0))
        sh = Frame(str_row, bg=SURFACE2)
        sh.pack(fill=X)
        Label(sh, text="JAGGEDNESS THRESHOLD", font=FONT_LABEL,
              bg=SURFACE2, fg=SUBTEXT).pack(side=LEFT)
        TooltipLabel(sh,
            "Controls how jagged an edge must be before it gets trimmed.\n\n"
            "Measured as: arc_length / endpoint_distance\n"
            "  1.0 = perfectly straight line → skipped\n"
            "  >1  = jagged → trimmed\n\n"
            "Higher value = trim more edges (including mild jags).\n"
            "Lower value  = only trim very jagged edges.\n\n"
            "Recommended: 85–95 for pixel art."
        ).pack(side=LEFT, padx=4)
        Scale(str_row, from_=50, to=99, resolution=1, orient=HORIZONTAL,
              variable=self._outline_straightness_pct,
              bg=SURFACE2, fg=TEXT, troughcolor=SURFACE,
              highlightthickness=0, activebackground=ACCENT,
              sliderrelief="flat", font=FONT_SMALL,
              showvalue=True).pack(fill=X)
        Label(str_row, text="(×100 — e.g. 90 = skip edges with score ≥ 0.90)",
              font=FONT_SMALL, bg=SURFACE2, fg=SUBTEXT).pack(anchor=W)

        # Trim depth
        trim_row = Frame(self.outline_controls, bg=SURFACE2)
        trim_row.pack(fill=X, padx=10, pady=(6, 8))
        Label(trim_row, text="Trim depth", font=FONT_SMALL,
              bg=SURFACE2, fg=SUBTEXT).pack(side=LEFT)
        Spinbox(trim_row, from_=1, to=3, textvariable=self._outline_trim_px,
                width=3, bg=SURFACE, fg=TEXT,
                buttonbackground=SURFACE2, relief="flat",
                font=FONT_BODY, insertbackground=TEXT).pack(side=LEFT, padx=6)
        Label(trim_row, text="px  (outer layers to remove)",
              font=FONT_SMALL, bg=SURFACE2, fg=SUBTEXT).pack(side=LEFT)

        # ── Redraw Outline panel ──────────────────────────────────────────────
        SectionLabel(left, "REDRAW OUTLINE").pack(fill=X, pady=(14, 4))

        redraw_card = Frame(left, bg=SURFACE2,
                            highlightthickness=1, highlightbackground=BORDER)
        redraw_card.pack(fill=X)

        redraw_top = Frame(redraw_card, bg=SURFACE2)
        redraw_top.pack(fill=X, padx=10, pady=(8, 4))
        Checkbutton(
            redraw_top,
            text="Redraw Outline  —  draw a clean geometric outline",
            variable=self._redraw_outline,
            command=self._on_redraw_outline_change,
            bg=SURFACE2, fg=TEXT, selectcolor=SURFACE,
            activebackground=SURFACE2, activeforeground=ACCENT,
            font=FONT_BODY, anchor=W).pack(side=LEFT, fill=X, expand=True)
        TooltipLabel(
            redraw_top,
            "After trimming, draws a new clean outline along the current\n"
            "outer edge using Douglas-Peucker simplification.\n\n"
            "The outline colour is sampled from the edge pixels themselves\n"
            "so no new colour is introduced — or you can pick a fixed colour.\n\n"
            "Lines are drawn only onto already-opaque pixels, so the\n"
            "transparent background is never touched.\n\n"
            "Best used AFTER Trim Outer Edge."
        ).pack(side=RIGHT, padx=6)

        # Sub-controls (shown/hidden)
        self.redraw_controls = Frame(redraw_card, bg=SURFACE2)

        # Simplification strength
        eps_row = Frame(self.redraw_controls, bg=SURFACE2)
        eps_row.pack(fill=X, padx=10, pady=(4, 0))
        eps_hdr = Frame(eps_row, bg=SURFACE2)
        eps_hdr.pack(fill=X)
        Label(eps_hdr, text="SIMPLIFICATION STRENGTH", font=FONT_LABEL,
              bg=SURFACE2, fg=SUBTEXT).pack(side=LEFT)
        TooltipLabel(eps_hdr,
            "Douglas-Peucker tolerance as % of each contour's arc-length.\n\n"
            "Low (5)  = subtle — preserves most corners, minor cleanup.\n"
            "High (50) = aggressive — reduces to few long straight segments.\n\n"
            "Value shown ×10, so 20 = 2.0% tolerance.\n"
            "Recommended: 15–30 for pixel art."
        ).pack(side=LEFT, padx=4)
        Scale(eps_row, from_=5, to=100, resolution=1, orient=HORIZONTAL,
              variable=self._redraw_epsilon_x10,
              bg=SURFACE2, fg=TEXT, troughcolor=SURFACE,
              highlightthickness=0, activebackground=ACCENT,
              sliderrelief="flat", font=FONT_SMALL,
              showvalue=True).pack(fill=X)
        Label(eps_row, text="(×10 — e.g. 20 = 2.0% tolerance)",
              font=FONT_SMALL, bg=SURFACE2, fg=SUBTEXT).pack(anchor=W)

        # Min length
        rlen_row = Frame(self.redraw_controls, bg=SURFACE2)
        rlen_row.pack(fill=X, padx=10, pady=(4, 0))
        rlh = Frame(rlen_row, bg=SURFACE2)
        rlh.pack(fill=X)
        Label(rlh, text="MIN SEGMENT LENGTH", font=FONT_LABEL,
              bg=SURFACE2, fg=SUBTEXT).pack(side=LEFT)
        TooltipLabel(rlh,
            "Skip contours shorter than this % of image diagonal.\n"
            "Keeps tiny dots and specks from being redrawn."
        ).pack(side=LEFT, padx=4)
        Scale(rlen_row, from_=1, to=20, resolution=1, orient=HORIZONTAL,
              variable=self._redraw_min_len,
              bg=SURFACE2, fg=TEXT, troughcolor=SURFACE,
              highlightthickness=0, activebackground=ACCENT,
              sliderrelief="flat", font=FONT_SMALL,
              showvalue=True).pack(fill=X)

        # Thickness
        rthick_row = Frame(self.redraw_controls, bg=SURFACE2)
        rthick_row.pack(fill=X, padx=10, pady=(6, 0))
        Label(rthick_row, text="Line thickness", font=FONT_SMALL,
              bg=SURFACE2, fg=SUBTEXT).pack(side=LEFT)
        Spinbox(rthick_row, from_=1, to=3, textvariable=self._redraw_thickness,
                width=3, bg=SURFACE, fg=TEXT,
                buttonbackground=SURFACE2, relief="flat",
                font=FONT_BODY, insertbackground=TEXT).pack(side=LEFT, padx=6)
        Label(rthick_row, text="px  (1 recommended for pixel art)",
              font=FONT_SMALL, bg=SURFACE2, fg=SUBTEXT).pack(side=LEFT)

        # Color mode
        rcol_hdr = Frame(self.redraw_controls, bg=SURFACE2)
        rcol_hdr.pack(fill=X, padx=10, pady=(8, 2))
        Label(rcol_hdr, text="COLOUR SOURCE", font=FONT_LABEL,
              bg=SURFACE2, fg=SUBTEXT).pack(side=LEFT)
        TooltipLabel(rcol_hdr,
            "Auto: reads the dominant colour of the edge pixels themselves.\n"
            "No new colour is introduced — uses what's already there.\n\n"
            "Custom: override with a colour you pick.\n"
            "Useful when edges have mixed or noisy colours."
        ).pack(side=LEFT, padx=4)

        rcol_row = Frame(self.redraw_controls, bg=SURFACE2)
        rcol_row.pack(fill=X, padx=10, pady=(0, 4))
        Radiobutton(rcol_row, text="Auto  (sample edge colour)",
                    variable=self._redraw_color_mode, value="auto",
                    command=self._on_redraw_color_mode_change,
                    bg=SURFACE2, fg=TEXT, selectcolor=SURFACE,
                    activebackground=SURFACE2, activeforeground=ACCENT,
                    font=FONT_BODY).pack(side=LEFT)
        Radiobutton(rcol_row, text="Custom",
                    variable=self._redraw_color_mode, value="custom",
                    command=self._on_redraw_color_mode_change,
                    bg=SURFACE2, fg=TEXT, selectcolor=SURFACE,
                    activebackground=SURFACE2, activeforeground=ACCENT,
                    font=FONT_BODY).pack(side=LEFT, padx=(12, 0))

        # Custom color picker (shown when "custom")
        self.redraw_color_row = Frame(self.redraw_controls, bg=SURFACE2)
        Label(self.redraw_color_row, text="Color:", font=FONT_SMALL,
              bg=SURFACE2, fg=SUBTEXT).pack(side=LEFT, padx=(10, 4))
        self.redraw_swatch = Label(self.redraw_color_row, width=4, bg="#000000",
                                   relief="flat", highlightthickness=1,
                                   highlightbackground=BORDER)
        self.redraw_swatch.pack(side=LEFT)
        self.redraw_hex_lbl = Label(self.redraw_color_row, text="#000000",
                                    font=FONT_SMALL, bg=SURFACE2, fg=SUBTEXT)
        self.redraw_hex_lbl.pack(side=LEFT, padx=6)
        styled_btn(self.redraw_color_row, "Pick…",
                   self._pick_redraw_color, small=True).pack(side=LEFT, padx=(4, 10))
        Frame(self.redraw_controls, bg=SURFACE2, height=6).pack()

        # ── RIGHT column ──────────────────────────────────────────────────────
        SectionLabel(right, "INPAINTING ENGINE").pack(fill=X, pady=(0, 6))

        engines = [
            ("telea", "Telea  (fast, sharp edges)",
             "Alexandru Telea's fast marching method.\n"
             "Best for: text removal, thin objects, small regions.\n"
             "Speed: very fast  |  Quality: good"),
            ("ns",    "Navier-Stokes  (smooth, fluid)",
             "Fluid-dynamics propagation model.\n"
             "Best for: large smooth areas, organic textures.\n"
             "Speed: fast  |  Quality: good"),
            ("lama",  "LaMa AI  ✦  (best quality)",
             "Deep-learning large-mask inpainting model.\n"
             "Best for: complex backgrounds, large regions.\n"
             "Requires: pip install simple-lama-inpainting\n"
             "Speed: slower  |  Quality: excellent"),
        ]
        for val, label, tip in engines:
            row = Frame(right, bg=SURFACE2,
                        highlightthickness=1, highlightbackground=BORDER)
            row.pack(fill=X, pady=2)
            Radiobutton(row, text=label, variable=self._backend, value=val,
                        command=self._on_engine_change,
                        bg=SURFACE2, fg=TEXT, selectcolor=SURFACE,
                        activebackground=SURFACE2, activeforeground=ACCENT,
                        font=FONT_BODY, indicatoron=True,
                        padx=10, pady=7).pack(side=LEFT, fill=X, expand=True)
            TooltipLabel(row, tip).pack(side=RIGHT, padx=6)

        # Radius (CV2 only)
        self.radius_frame = Frame(right, bg=BG)
        self.radius_frame.pack(fill=X, pady=(10, 0))
        r_hdr = Frame(self.radius_frame, bg=BG)
        r_hdr.pack(fill=X)
        Label(r_hdr, text="INPAINT RADIUS", font=FONT_LABEL,
              bg=BG, fg=SUBTEXT).pack(side=LEFT)
        TooltipLabel(r_hdr,
                     "How many pixels outside the mask boundary the algorithm\n"
                     "samples to reconstruct texture.\n"
                     "Larger = smoother but may blur detail.").pack(side=LEFT, padx=4)
        Scale(self.radius_frame, from_=1, to=30, orient=HORIZONTAL,
              variable=self._radius, bg=BG, fg=TEXT, troughcolor=SURFACE,
              highlightthickness=0, activebackground=ACCENT,
              sliderrelief="flat", font=FONT_SMALL,
              showvalue=True).pack(fill=X)

        # Mask dilation
        SectionLabel(right, "MASK REFINEMENT").pack(fill=X, pady=(12, 4))
        dil_row = Frame(right, bg=BG)
        dil_row.pack(fill=X)
        Label(dil_row, text="Dilate mask by", font=FONT_SMALL,
              bg=BG, fg=SUBTEXT).pack(side=LEFT)
        Spinbox(dil_row, from_=0, to=50, textvariable=self._mask_dilation,
                width=4, bg=SURFACE2, fg=TEXT,
                buttonbackground=SURFACE, relief="flat",
                font=FONT_BODY, insertbackground=TEXT).pack(side=LEFT, padx=6)
        Label(dil_row, text="px  (reduces halo edges)",
              font=FONT_SMALL, bg=BG, fg=SUBTEXT).pack(side=LEFT)

        # Output folder
        SectionLabel(right, "OUTPUT FOLDER").pack(fill=X, pady=(12, 4))
        out_row = Frame(right, bg=BG)
        out_row.pack(fill=X)
        Entry(out_row, textvariable=self._output_dir,
              bg=SURFACE2, fg=TEXT, insertbackground=TEXT,
              relief="flat", font=FONT_SMALL,
              highlightthickness=1,
              highlightbackground=BORDER).pack(side=LEFT, fill=X, expand=True)
        styled_btn(out_row, "…", self._browse_output,
                   small=True).pack(side=LEFT, padx=(4, 0))

        # Run / Cancel
        btn_row = Frame(right, bg=BG)
        btn_row.pack(fill=X, pady=(16, 4))
        self.run_btn = Button(btn_row, text="▶  PROCESS  IMAGE(S)",
                              command=self._run,
                              bg=ACCENT, fg="#ffffff", relief="flat",
                              disabledforeground="#ffffff",
                              activebackground=ACCENT2, font=FONT_RUN,
                              padx=14, pady=10, cursor="hand2")
        self.run_btn.pack(side=LEFT, fill=X, expand=True)
        self.cancel_btn = Button(btn_row, text="✕", command=self._cancel,
                                 bg=SURFACE2, fg=ERROR, relief="flat",
                                 activebackground=ERROR, activeforeground=TEXT,
                                 font=FONT_BTN, padx=10, pady=10,
                                 cursor="hand2", state=DISABLED)
        self.cancel_btn.pack(side=LEFT, padx=(4, 0))

        # Progress + status
        self.progress = ttk.Progressbar(right, mode="determinate")
        self.progress.pack(fill=X, pady=(0, 4))
        self.status_lbl = Label(right, text="Ready.", font=FONT_SMALL,
                                bg=BG, fg=SUBTEXT, anchor=W, wraplength=320)
        self.status_lbl.pack(fill=X)

        # Log
        SectionLabel(right, "LOG").pack(fill=X, pady=(10, 4))
        log_wrap = Frame(right, bg=SURFACE,
                         highlightthickness=1, highlightbackground=BORDER)
        log_wrap.pack(fill=BOTH, expand=True)
        self.log_box = Text(log_wrap, bg=SURFACE, fg=TEXT, relief="flat",
                            font=FONT_SMALL, state=DISABLED, wrap=WORD,
                            insertbackground=TEXT)
        vsb = Scrollbar(log_wrap, command=self.log_box.yview, bg=SURFACE)
        self.log_box.configure(yscrollcommand=vsb.set)
        vsb.pack(side=RIGHT, fill=Y)
        self.log_box.pack(fill=BOTH, expand=True, padx=6, pady=4)
        self.log_box.tag_configure("err",  foreground=ERROR)
        self.log_box.tag_configure("warn", foreground=WARN)
        self.log_box.tag_configure("good", foreground=SUCCESS)
        self.log_box.tag_configure("ok",   foreground=TEXT)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def _on_images(self, paths):
        self._image_paths = paths
        self.thumb.set_images(paths)
        self.img_count_lbl.configure(text=f"{len(paths)} image(s)")
        self._log(f"Loaded {len(paths)} source image(s).")
        auto_out = str(Path(paths[0]).parent / "Output")
        self._output_dir.set(auto_out)

    def _clear_images(self):
        self._image_paths = []
        self.img_drop.clear()
        self.img_count_lbl.configure(text="")
        for w in self.thumb.inner.winfo_children():
            w.destroy()

    def _on_mask(self, path):
        self._mask_path = path
        try:
            img = Image.open(path).convert("L")
            w, h = img.size
            self.mask_info_lbl.configure(text=f"{Path(path).name}\n{w} × {h} px")
            img.thumbnail((88, 88))
            tk_img = ImageTk.PhotoImage(img)
            self.mask_thumb_lbl.configure(image=tk_img)
            self.mask_thumb_lbl.image = tk_img  # type: ignore[attr-defined]
        except Exception as e:
            self.mask_info_lbl.configure(text=str(e))
        self._log(f"Mask loaded: {Path(path).name}")

    def _clear_mask(self):
        self._mask_path = None
        self.mask_drop.clear()
        self.mask_thumb_lbl.configure(image="")
        self.mask_thumb_lbl.image = None  # type: ignore[attr-defined]
        self.mask_info_lbl.configure(text="")
        self._log("Mask cleared.")

    def _on_engine_change(self):
        if self._backend.get() == "lama":
            self.radius_frame.pack_forget()
        else:
            self.radius_frame.pack(fill=X, pady=(10, 0))

    def _on_fill_holes_change(self):
        if self._fill_holes.get():
            self._log("Fill Holes enabled — interior transparent regions will be auto-filled.")
        else:
            self._log("Fill Holes disabled.")

    def _on_exclude_dark_change(self):
        if self._exclude_dark.get():
            self.dark_thresh_frame.pack(fill=X, pady=(2, 6))
            self._log(f"Exclude dark: ON  (threshold={self._dark_threshold.get()})")
        else:
            self.dark_thresh_frame.pack_forget()
            self._log("Exclude dark: OFF")

    def _on_smooth_outlines_change(self):
        if self._smooth_outlines.get():
            self.outline_controls.pack(fill=X)
            self._log("Trim Outer Edge enabled.")
        else:
            self.outline_controls.pack_forget()
            self._log("Trim Outer Edge disabled.")

    def _on_redraw_outline_change(self):
        if self._redraw_outline.get():
            self.redraw_controls.pack(fill=X)
            self._log("Redraw Outline enabled.")
        else:
            self.redraw_controls.pack_forget()
            self._log("Redraw Outline disabled.")

    def _on_redraw_color_mode_change(self):
        if self._redraw_color_mode.get() == "custom":
            self.redraw_color_row.pack(fill=X, pady=(0, 4))
        else:
            self.redraw_color_row.pack_forget()

    def _pick_redraw_color(self):
        from tkinter import colorchooser
        init = "#{:02x}{:02x}{:02x}".format(*self._redraw_custom_rgb)
        res  = colorchooser.askcolor(color=init, title="Choose outline colour")
        if res and res[0]:
            r, g, b = (int(c) for c in res[0])
            self._redraw_custom_rgb = (r, g, b)
            hex_str = "#{:02x}{:02x}{:02x}".format(r, g, b)
            self.redraw_swatch.configure(bg=hex_str)
            self.redraw_hex_lbl.configure(text=hex_str)
            self._log(f"Redraw colour set to {hex_str}  RGB({r},{g},{b})")

    def _browse_output(self):
        d = filedialog.askdirectory(title="Select output folder")
        if d:
            self._output_dir.set(d)

    # ── Run ───────────────────────────────────────────────────────────────────
    def _run(self):
        if not self._image_paths:
            messagebox.showwarning("No images", "Please load source images first.")
            return

        has_mask           = bool(self._mask_path)
        fill_holes_on      = self._fill_holes.get()
        smooth_outlines_on = self._smooth_outlines.get()
        redraw_outline_on  = self._redraw_outline.get()

        if not has_mask and not fill_holes_on and not smooth_outlines_on and not redraw_outline_on:
            messagebox.showwarning(
                "Nothing to do",
                "Please load a mask or enable at least one processing step.")
            return

        backend = self._backend.get()
        if not CV2_AVAILABLE and backend != "lama":
            messagebox.showerror("Missing dependency",
                                 "opencv-python is required for this engine.\n"
                                 "Run:  pip install opencv-python")
            return
        if not CV2_AVAILABLE and fill_holes_on:
            messagebox.showerror("Missing dependency",
                                 "opencv-python is required for hole detection.\n"
                                 "Run:  pip install opencv-python")
            return

        self._cancel_flag.clear()
        self.run_btn.configure(state=DISABLED, text="Processing…")
        self.cancel_btn.configure(state=NORMAL)
        self.progress["value"] = 0
        self.progress["maximum"] = len(self._image_paths)
        threading.Thread(target=self._process_thread, daemon=True).start()

    def _cancel(self):
        self._cancel_flag.set()
        self._log("Cancelling after current image…", "warn")
        self.cancel_btn.configure(state=DISABLED)

    # ── Processing thread ─────────────────────────────────────────────────────
    def _process_thread(self):
        out_dir   = Path(self._output_dir.get())
        out_dir.mkdir(parents=True, exist_ok=True)
        backend            = self._backend.get()
        radius             = self._radius.get()
        dilation           = self._mask_dilation.get()
        has_mask           = bool(self._mask_path)
        fill_holes_on      = self._fill_holes.get()
        smooth_outlines_on   = self._smooth_outlines.get()
        outline_min_len      = float(self._outline_min_len.get())
        outline_straightness = self._outline_straightness_pct.get() / 100.0
        outline_trim_px      = self._outline_trim_px.get()
        redraw_outline_on    = self._redraw_outline.get()
        redraw_epsilon_pct   = self._redraw_epsilon_x10.get() / 10.0
        redraw_color_mode    = self._redraw_color_mode.get()
        redraw_custom_rgb    = self._redraw_custom_rgb
        redraw_thickness     = self._redraw_thickness.get()
        redraw_min_len       = float(self._redraw_min_len.get())

        # Load and optionally dilate the user-supplied mask once
        mask_pil: "Image.Image | None" = None
        if has_mask:
            assert self._mask_path is not None
            try:
                mask_pil = Image.open(self._mask_path).convert("L")
                if dilation > 0 and CV2_AVAILABLE:
                    assert cv2 is not None
                    mask_np  = np.array(mask_pil)
                    kernel   = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (dilation * 2 + 1, dilation * 2 + 1))
                    mask_np  = cv2.dilate(mask_np, kernel)
                    mask_pil = Image.fromarray(mask_np)
                    self._log(f"Mask dilated by {dilation}px")
            except Exception as e:
                self._log(f"Failed to load mask: {e}", "err")
                self.after(0, self._done)
                return

        success = fail = 0
        for i, p in enumerate(self._image_paths):
            if self._cancel_flag.is_set():
                self._log("Cancelled by user.", "warn")
                break
            try:
                src = Image.open(p)

                self.after(0, self.status_lbl.configure,
                           {"text": f"[{i+1}/{len(self._image_paths)}] {Path(p).name}…",
                            "fg": ACCENT})

                result = src.copy()

                # ── Step 1: apply user mask inpainting (if mask loaded) ──
                if mask_pil is not None:
                    m = mask_pil
                    if m.size != src.size:
                        m = m.resize(src.size, Image.Resampling.NEAREST)
                    if backend in ("telea", "ns"):
                        result = inpaint_opencv(result, m, method=backend, radius=radius)
                    else:
                        result = inpaint_lama(result, m)
                    self._log(f"  Mask inpainting done: {Path(p).name}")

                # ── Step 2: fill interior transparent holes ──────────────
                if fill_holes_on:
                    # Diagnose image mode first
                    self._log(f"  mode:{result.mode} size:{result.size[0]}x{result.size[1]}")
                    alpha_dbg = _get_alpha(result)
                    if alpha_dbg is None:
                        self._log("  No alpha channel — hole fill skipped (need RGBA/PNG)", "warn")
                    else:
                        total_trans = int((alpha_dbg < 128).sum())
                        self._log(f"  {total_trans} transparent px detected in alpha channel")
                        hole_mask_arr = detect_interior_holes(result)
                        if hole_mask_arr is None:
                            self._log("  No interior holes — all transparent px touch the border")
                        else:
                            hole_count = int((hole_mask_arr > 127).sum())
                            self._log(f"  {hole_count} interior hole px -> nearest-neighbour fill…")
                            result, filled = fill_holes_nearest_neighbour(
                                result, hole_mask_arr,
                                exclude_dark=self._exclude_dark.get(),
                                dark_threshold=self._dark_threshold.get())
                            excl_note = (f"  (dark exclusion ON, threshold={self._dark_threshold.get()})"
                                         if self._exclude_dark.get() else "")
                            self._log(f"  Filled {filled} px (existing palette colours only){excl_note}", "good")

                # ── Step 3: trim outer edge ──────────────────────────────
                if smooth_outlines_on:
                    self._log(f"  Trim outer edge (min={outline_min_len:.1f}% diag, "
                              f"jaggedness<{outline_straightness:.2f}, "
                              f"depth={outline_trim_px}px)…")
                    result, n_edges, n_px = trim_outer_edge(
                        result,
                        min_length_pct=outline_min_len,
                        straightness_threshold=outline_straightness,
                        trim_px=outline_trim_px)
                    if n_edges == 0:
                        self._log("  No qualifying jagged edges found — try raising the jaggedness threshold or lowering min length", "warn")
                    else:
                        self._log(f"  Trimmed {n_edges} edge segment(s), {n_px} px removed", "good")

                # ── Step 4: redraw clean outline ─────────────────────────
                if redraw_outline_on:
                    self._log(f"  Redraw outline (epsilon={redraw_epsilon_pct:.1f}%, "
                              f"min={redraw_min_len:.1f}% diag, "
                              f"color={redraw_color_mode}, thick={redraw_thickness}px)…")
                    result, n_drawn, n_px2 = redraw_outline(
                        result,
                        epsilon_pct=redraw_epsilon_pct,
                        color_mode=redraw_color_mode,
                        custom_color=redraw_custom_rgb if redraw_color_mode == "custom" else None,
                        thickness=redraw_thickness,
                        min_length_pct=redraw_min_len)
                    if n_drawn == 0:
                        self._log("  No outline segments found to redraw — try lowering min segment length", "warn")
                    else:
                        self._log(f"  Redrawn {n_drawn} segment(s), {n_px2} px painted", "good")

                out_name = Path(p).stem + "_processed.png"
                result.save(out_dir / out_name)
                self._log(f"✓  {Path(p).name}", "good")
                success += 1
            except Exception as e:
                self._log(f"✗  {Path(p).name}: {e}", "err")
                fail += 1

            self.after(0, self._tick_progress, i + 1)

        summary_tag = "good" if fail == 0 else "warn"
        self._log(
            f"\n{'✓' if fail == 0 else '!'} Finished — "
            f"{success} succeeded" + (f", {fail} failed" if fail else "") +
            f"\n  Output → {out_dir}",
            summary_tag)
        self.after(0, self._done, SUCCESS if fail == 0 else WARN)

    def _tick_progress(self, val):
        self.progress["value"] = val

    def _done(self, colour=SUCCESS):
        self.run_btn.configure(state=NORMAL, text="▶  PROCESS  IMAGE(S)")
        self.cancel_btn.configure(state=DISABLED)
        self.status_lbl.configure(text="Done!", fg=colour)

    # ── Logging ───────────────────────────────────────────────────────────────
    def _log(self, msg, tag="ok"):
        def _write():
            self.log_box.configure(state=NORMAL)
            self.log_box.insert(END, msg + "\n", tag)
            self.log_box.see(END)
            self.log_box.configure(state=DISABLED)
        self.after(0, _write)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if not DND_AVAILABLE:
        print("Tip: pip install tkinterdnd2  →  enables drag-and-drop")
    app = InpaintApp()
    app.mainloop()
    # this is a test comment