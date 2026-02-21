#!/usr/bin/env python3
"""
Image Collage Tool
- Combine multiple images into a collage (with metadata saved)
- Separate collage back into original images (supports resized collage input)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import os
from PIL import Image, ImageDraw, ImageFont
import math

# ── Collage logic ──────────────────────────────────────────────────────────────

def build_collage(image_paths, padding=10, bg_color=(255, 255, 255), keep_alpha=False):
    """Arrange images in a grid, return (collage_image, metadata_list)."""
    images = [Image.open(p).convert("RGBA") for p in image_paths]
    n = len(images)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    max_w = max(im.width for im in images)
    max_h = max(im.height for im in images)

    canvas_w = cols * max_w + (cols + 1) * padding
    canvas_h = rows * max_h + (rows + 1) * padding

    canvas = Image.new("RGBA", (canvas_w, canvas_h), bg_color + (255,))

    metadata = []
    for idx, (im, path) in enumerate(zip(images, image_paths)):
        row = idx // cols
        col = idx % cols
        x = padding + col * (max_w + padding)
        y = padding + row * (max_h + padding)
        canvas.paste(im, (x, y), mask=im)
        metadata.append({
            "index": idx,
            "source": os.path.abspath(path),
            "filename": os.path.basename(path),
            "x": x,
            "y": y,
            "width": im.width,
            "height": im.height,
        })

    # Drop alpha to RGB unless explicitly requested — saves ~25% file size
    if not keep_alpha:
        canvas = canvas.convert("RGB")

    return canvas, metadata


def separate_collage(collage_path, metadata, output_dir, resize_factor=1.0):
    """
    Extract sub-images from a (possibly resized) collage.
    resize_factor = actual_collage_size / original_collage_size
    Always saves output as PNG (lossless).
    """
    collage = Image.open(collage_path).convert("RGBA")
    os.makedirs(output_dir, exist_ok=True)
    saved = []
    for item in metadata:
        x = round(item["x"] * resize_factor)
        y = round(item["y"] * resize_factor)
        w = round(item["width"] * resize_factor)
        h = round(item["height"] * resize_factor)
        crop = collage.crop((x, y, x + w, y + h))
        # Restore original resolution using nearest-neighbour (pixel art)
        if resize_factor != 1.0:
            crop = crop.resize((item["width"], item["height"]), Image.NEAREST)
        # Always output PNG regardless of original extension
        base = os.path.splitext(item["filename"])[0]
        out_name = f"{idx_pad(item['index'])}_{base}.png"
        out_path = os.path.join(output_dir, out_name)
        crop.save(out_path, format="PNG", compress_level=6)  # lossless, default compression
        saved.append(out_path)
    return saved


def idx_pad(n):
    return str(n).zfill(3)


# ── GUI ────────────────────────────────────────────────────────────────────────

class CollageTool(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Collage Tool")
        self.resizable(False, False)
        self.configure(bg="#f0f0f0")

        self._image_paths = []       # paths chosen for collage
        self._metadata = []          # set after saving collage
        self._collage_orig_size = None  # (w, h) of original collage

        self._build_ui()

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=10, pady=10)

        self._tab_create = ttk.Frame(nb)
        self._tab_separate = ttk.Frame(nb)
        nb.add(self._tab_create,   text="  Create Collage  ")
        nb.add(self._tab_separate, text="  Separate Collage  ")

        self._build_create_tab(self._tab_create)
        self._build_separate_tab(self._tab_separate)

    # ── Create tab ─────────────────────────────────────────────────────────────

    def _build_create_tab(self, parent):
        pad = {"padx": 10, "pady": 6}

        # Image list
        lf = ttk.LabelFrame(parent, text="Images to combine")
        lf.pack(fill="both", expand=True, **pad)

        self._listbox = tk.Listbox(lf, selectmode="extended", height=10,
                                   font=("Courier", 10))
        self._listbox.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        sb = ttk.Scrollbar(lf, orient="vertical", command=self._listbox.yview)
        sb.pack(side="right", fill="y")
        self._listbox.config(yscrollcommand=sb.set)

        btn_frame = ttk.Frame(parent)
        btn_frame.pack(**pad)
        ttk.Button(btn_frame, text="Add Images…",  command=self._add_images).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Remove Selected", command=self._remove_selected).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Clear All",    command=self._clear_images).pack(side="left", padx=4)

        # Padding + alpha options
        opt_frame = ttk.Frame(parent)
        opt_frame.pack(**pad)
        ttk.Label(opt_frame, text="Padding (px):").pack(side="left")
        self._padding_var = tk.IntVar(value=10)
        ttk.Spinbox(opt_frame, from_=0, to=100, textvariable=self._padding_var,
                    width=6).pack(side="left", padx=6)

        self._keep_alpha_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_frame, text="Keep alpha channel (RGBA — only if images use transparency)",
                        variable=self._keep_alpha_var).pack(side="left", padx=12)

        # Action
        ttk.Button(parent, text="💾  Save Collage…", command=self._save_collage,
                   style="Accent.TButton").pack(pady=8)

        self._create_status = ttk.Label(parent, text="", foreground="green")
        self._create_status.pack()

    def _add_images(self):
        paths = filedialog.askopenfilenames(
            title="Select images",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"), ("All", "*.*")]
        )
        for p in paths:
            if p not in self._image_paths:
                self._image_paths.append(p)
                self._listbox.insert("end", os.path.basename(p))

    def _remove_selected(self):
        for i in reversed(self._listbox.curselection()):
            self._listbox.delete(i)
            del self._image_paths[i]

    def _clear_images(self):
        self._listbox.delete(0, "end")
        self._image_paths.clear()

    def _save_collage(self):
        if len(self._image_paths) < 2:
            messagebox.showwarning("No images", "Add at least 2 images.")
            return
        out_path = filedialog.asksaveasfilename(
            title="Save collage as (PNG — lossless)",
            defaultextension=".png",
            filetypes=[("PNG (lossless)", "*.png")]
        )
        if not out_path:
            return
        # Ensure .png extension regardless of what the user typed
        if not out_path.lower().endswith(".png"):
            out_path += ".png"
        try:
            collage, metadata = build_collage(
                self._image_paths,
                padding=self._padding_var.get(),
                keep_alpha=self._keep_alpha_var.get()
            )
            # Save losslessly, no compression (compress_level=0 = fastest, still lossless)
            collage.save(out_path, format="PNG", compress_level=6)
            self._collage_orig_size = collage.size
            self._metadata = metadata

            # Compute sizes
            collage_mb = os.path.getsize(out_path) / 1_048_576
            src_mb = sum(os.path.getsize(p) for p in self._image_paths) / 1_048_576

            # Save metadata sidecar
            meta_path = out_path + ".meta.json"
            with open(meta_path, "w") as f:
                json.dump({
                    "collage_width": collage.width,
                    "collage_height": collage.height,
                    "images": metadata
                }, f, indent=2)

            self._create_status.config(
                text=(
                    f"✓ {os.path.basename(out_path)} saved  "
                    f"({collage_mb:.2f} MB collage | {src_mb:.2f} MB source total)\n"
                    f"Metadata: {os.path.basename(meta_path)}"
                )
            )
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ── Separate tab ───────────────────────────────────────────────────────────

    def _build_separate_tab(self, parent):
        pad = {"padx": 10, "pady": 6}

        # Collage file
        cf = ttk.LabelFrame(parent, text="Collage file")
        cf.pack(fill="x", **pad)
        self._collage_path_var = tk.StringVar()
        ttk.Entry(cf, textvariable=self._collage_path_var, width=50).pack(side="left", padx=5, pady=5)
        ttk.Button(cf, text="Browse…", command=self._browse_collage).pack(side="left")

        # Metadata file
        mf = ttk.LabelFrame(parent, text="Metadata file (.meta.json)")
        mf.pack(fill="x", **pad)
        self._meta_path_var = tk.StringVar()
        ttk.Entry(mf, textvariable=self._meta_path_var, width=50).pack(side="left", padx=5, pady=5)
        ttk.Button(mf, text="Browse…", command=self._browse_meta).pack(side="left")

        # Resize section
        rf = ttk.LabelFrame(parent, text="Collage was resized?")
        rf.pack(fill="x", **pad)

        self._resized_var = tk.BooleanVar(value=False)
        cb = ttk.Checkbutton(rf, text="The collage file has been resized",
                             variable=self._resized_var, command=self._toggle_resize)
        cb.pack(anchor="w", padx=8, pady=4)

        self._resize_frame = ttk.Frame(rf)
        self._resize_frame.pack(fill="x", padx=8, pady=(0, 6))

        ttk.Label(self._resize_frame, text="Scale factor:").grid(row=0, column=0, sticky="w")
        self._scale_var = tk.DoubleVar(value=1.0)
        self._scale_slider = ttk.Scale(self._resize_frame, from_=0.05, to=4.0,
                                       orient="horizontal", variable=self._scale_var,
                                       command=self._slider_moved, length=300)
        self._scale_slider.grid(row=0, column=1, padx=8)
        self._scale_entry = ttk.Entry(self._resize_frame, width=8)
        self._scale_entry.insert(0, "1.0")
        self._scale_entry.grid(row=0, column=2)
        self._scale_entry.bind("<Return>", self._entry_changed)
        self._scale_entry.bind("<FocusOut>", self._entry_changed)

        ttk.Label(self._resize_frame,
                  text="(e.g. 0.5 = collage shrunk to 50%  |  2.0 = doubled)",
                  foreground="gray").grid(row=1, column=0, columnspan=3, sticky="w", pady=2)

        # Helpful size fields
        sz_frame = ttk.Frame(rf)
        sz_frame.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Label(sz_frame, text="Or enter current collage size (px):").pack(side="left")
        self._curr_w_var = tk.StringVar()
        self._curr_h_var = tk.StringVar()
        ttk.Entry(sz_frame, textvariable=self._curr_w_var, width=7).pack(side="left", padx=4)
        ttk.Label(sz_frame, text="×").pack(side="left")
        ttk.Entry(sz_frame, textvariable=self._curr_h_var, width=7).pack(side="left", padx=4)
        ttk.Button(sz_frame, text="Compute scale", command=self._compute_scale).pack(side="left", padx=6)

        self._toggle_resize()  # start disabled

        # Output dir
        od = ttk.LabelFrame(parent, text="Output directory")
        od.pack(fill="x", **pad)
        self._outdir_var = tk.StringVar()
        ttk.Entry(od, textvariable=self._outdir_var, width=50).pack(side="left", padx=5, pady=5)
        ttk.Button(od, text="Browse…", command=self._browse_outdir).pack(side="left")

        # Action
        ttk.Button(parent, text="✂  Separate Images", command=self._separate).pack(pady=8)
        self._sep_status = ttk.Label(parent, text="", foreground="green")
        self._sep_status.pack()

    def _toggle_resize(self):
        state = "normal" if self._resized_var.get() else "disabled"
        for child in self._resize_frame.winfo_children():
            try:
                child.config(state=state)
            except Exception:
                pass

    def _slider_moved(self, val):
        v = round(float(val), 3)
        self._scale_entry.delete(0, "end")
        self._scale_entry.insert(0, str(v))

    def _entry_changed(self, event=None):
        try:
            v = float(self._scale_entry.get())
            v = max(0.05, min(4.0, v))
            self._scale_var.set(v)
        except ValueError:
            pass

    def _compute_scale(self):
        """Compute scale from the metadata's original size vs entered current size."""
        try:
            meta = self._load_meta()
        except Exception as e:
            messagebox.showerror("Error", f"Load metadata first:\n{e}")
            return
        try:
            cw = float(self._curr_w_var.get())
            ch = float(self._curr_h_var.get())
        except ValueError:
            messagebox.showwarning("Input error", "Enter valid width and height.")
            return
        orig_w = meta["collage_width"]
        orig_h = meta["collage_height"]
        scale_w = cw / orig_w
        scale_h = ch / orig_h
        if abs(scale_w - scale_h) > 0.01:
            messagebox.showwarning("Aspect mismatch",
                f"Width scale={scale_w:.3f} ≠ Height scale={scale_h:.3f}.\n"
                "The collage may have been non-uniformly scaled. Using width scale.")
        scale = round(scale_w, 4)
        self._scale_var.set(max(0.05, min(4.0, scale)))
        self._scale_entry.delete(0, "end")
        self._scale_entry.insert(0, str(scale))
        self._resized_var.set(True)
        self._toggle_resize()

    def _browse_collage(self):
        p = filedialog.askopenfilename(
            title="Select collage",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"), ("All", "*.*")]
        )
        if p:
            self._collage_path_var.set(p)
            # Auto-locate sidecar
            guess = p + ".meta.json"
            if os.path.exists(guess):
                self._meta_path_var.set(guess)

    def _browse_meta(self):
        p = filedialog.askopenfilename(
            title="Select metadata",
            filetypes=[("JSON", "*.json"), ("All", "*.*")]
        )
        if p:
            self._meta_path_var.set(p)

    def _browse_outdir(self):
        d = filedialog.askdirectory(title="Select output directory")
        if d:
            self._outdir_var.set(d)

    def _load_meta(self):
        with open(self._meta_path_var.get()) as f:
            return json.load(f)

    def _separate(self):
        collage_path = self._collage_path_var.get()
        meta_path    = self._meta_path_var.get()
        out_dir      = self._outdir_var.get()

        if not collage_path or not os.path.exists(collage_path):
            messagebox.showwarning("Missing file", "Select a valid collage file.")
            return
        if not meta_path or not os.path.exists(meta_path):
            messagebox.showwarning("Missing file", "Select a valid metadata JSON file.")
            return
        if not out_dir:
            messagebox.showwarning("Missing directory", "Select an output directory.")
            return

        try:
            meta = self._load_meta()
            scale = self._scale_var.get() if self._resized_var.get() else 1.0
            saved = separate_collage(collage_path, meta["images"], out_dir, scale)
            total_mb = sum(os.path.getsize(p) for p in saved) / 1_048_576
            self._sep_status.config(
                text=f"✓ Extracted {len(saved)} image(s) → {out_dir}  ({total_mb:.2f} MB total)"
            )
        except Exception as e:
            messagebox.showerror("Error", str(e))


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        from PIL import Image
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow", "--quiet"])
        from PIL import Image

    app = CollageTool()
    app.mainloop()