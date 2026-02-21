# Image Cleaner — Game Asset Tool

A desktop GUI for cleaning and preparing sprite/texture images for in-game use.  
Wraps ImageMagick (`magick`) with a clean PyQt6 interface.

---

## Requirements

| Dependency | Install |
|---|---|
| Python 3.9+ | https://python.org |
| ImageMagick 7+ | https://imagemagick.org/script/download.php |
| PyQt6 | `pip install PyQt6` |

> **Windows tip:** When installing ImageMagick, tick **"Add to system PATH"** in the installer.

---

## Quick start

```bash
pip install PyQt6
python image_cleaner.py
```

---

## Features

### Background Removal
Flood-fills white pixels from all four corners of the image and makes them transparent.
- **Fuzz %** — tolerance for "near-white" colours (0 = exact white, 10 = light greys too)
- **Erode (px)** — shrinks the mask edge inward by N pixels to remove white fringing

### Resize
- **Scale %** — resize relative to original size (e.g. 50 = half size)
- **Filter** — `Point` (nearest-neighbour, pixel art), `Lanczos` (best for photos), `Box`, `Mitchell`, `Sinc`

### Color Quantization
Reduces the number of unique colours in the output (useful for indexed-colour sprites).
- **Colors** — maximum palette size (2–256)
- **Dithering** — blend colours at edges; disable for crisp pixel art

### Output
- **Format** — `PNG32` (full transparency), `PNG`, `TGA`, `BMP`
- **Suffix** — appended to the filename before the extension (e.g. `_clean` → `hero_clean.png`)
- **Overwrite originals** — saves back to the same filename if checked

### Batch Processing
Add as many images as you like via **Add Images** or drag-and-drop onto the file list.  
All images are processed with the same settings in a background thread — the UI stays responsive.

### Before / After Preview
Click any file in the list to see the original on the left.  
After processing, click the same file again to see the cleaned result on the right.

---

## Equivalent magick commands

The app builds commands equivalent to:

```
# Background removal
magick input.png -alpha set -fuzz 5% -fill none \
  -draw "color 0,0 floodfill" -draw "color %[fx:w-1],0 floodfill" \
  -draw "color 0,%[fx:h-1] floodfill" -draw "color %[fx:w-1],%[fx:h-1] floodfill" \
  -alpha set -morphology erode square:1 PNG32:output.png

# Resize
magick input.png -filter Point -resize 50% PNG32:output.png

# Quantize
magick input.png +dither -colors 128 PNG32:output.png
```

All three steps can be chained together in a single command.
