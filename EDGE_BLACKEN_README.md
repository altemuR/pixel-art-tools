# Pixel Art Edge Blacken

GUI tool that processes pixel art by tracing edges from **outside to inside** for a set number of pixels, then snapping **near-black** pixels in that band to **pitch black** (cleans up dark anti-aliasing on outlines).

## Run

```bash
pip install -r requirements.txt
python edge_blacken_gui.py
```

## Usage

1. **Add images** – Select one or more images (PNG, JPG, etc.).
2. **Edge depth (pixels)** – How many pixels inward from the outer edge to process (e.g. 2–5 for typical outlines).
3. **Black threshold (0–255)** – Pixels with `max(R,G,B) <=` this value in the edge band become `(0,0,0)`. Lower = only very dark pixels; higher = more pixels forced to black.
4. **Remove transparent pixels** – When enabled, pixels in the edge band with `alpha <=` the threshold are made fully transparent (removed).
5. **Output folder** – Where to save processed images (suffix `_edge_blacken`).
6. Click **Process and save**.

## Behavior

- **With transparency:** The “edge” is the boundary of non-transparent content; the band is that boundary plus N pixels inward.
- **Fully opaque images:** The outer border of the image is treated as the edge; the band is the first N pixels inward from each side.
- Only pixels **inside the edge band** are considered. Near-black pixels become pure black; optionally, semi-transparent pixels are removed (alpha → 0).
- All processing is **pixel-perfect** (no resampling or interpolation).
