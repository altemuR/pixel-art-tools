"""Quick test: create a small RGBA image, run processor, check that output differs."""
import numpy as np
from PIL import Image
from edge_blacken_processor import _get_edge_band, process_black_outline, process_remove_transparent

# 20x20: transparent with 10x10 white square in center (alpha=255)
h, w = 20, 20
rgba = np.zeros((h, w, 4), dtype=np.uint8)
rgba[5:15, 5:15, :3] = 255
rgba[5:15, 5:15, 3] = 255
img = Image.fromarray(rgba, "RGBA")

# Band should be the 10x10 square's border + 2 pixels inward
_, band = _get_edge_band(img, num_pixels=3)
band_count = np.sum(band)
print(f"Band pixel count (expect > 0): {band_count}")

# Black outline with threshold 255 = blacken ALL pixels in band
out = process_black_outline(img, num_pixels=3, black_threshold=255)
arr = np.array(out)
# Some pixels in the band should now be black (0,0,0)
black_in_band = np.sum((arr[:, :, :3] == 0).all(axis=2) & band)
print(f"Black pixels in band (expect > 0): {black_in_band}")

# Remove transparent: set threshold 254 so we keep only alpha=255; remove rest in band
out2 = process_remove_transparent(img, num_pixels=3, transparency_threshold=254)
arr2 = np.array(out2)
# Center should be unchanged; edge band might have been trimmed if any alpha < 255
print("Test done. If band_count and black_in_band are > 0, edge/band logic works.")
