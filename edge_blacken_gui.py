"""
Pixel Art Edge Blacken – GUI.
Select multiple images, set edge depth and black threshold, process and save.
"""
import os
import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from PIL import Image

from edge_blacken_processor import (
    _get_edge_band,
    process_black_outline,
    process_remove_transparent,
    process_remove_transparent_all,
)

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DND_AVAILABLE = True
except ImportError:
    DND_FILES = None  # type: ignore[assignment]
    TkinterDnD = None  # type: ignore[assignment]
    DND_AVAILABLE = False


def run_gui():
    root = (TkinterDnD.Tk() if DND_AVAILABLE else tk.Tk())
    root.title("Pixel Art Edge Blacken")
    root.minsize(420, 380)
    root.resizable(True, True)

    # State
    image_paths = []
    output_dir = tk.StringVar(value="")

    # --- Widgets
    main = ttk.Frame(root, padding=12)
    main.pack(fill=tk.BOTH, expand=True)

    ttk.Label(main, text="Images (drag & drop or add below)", font=("", 10, "bold")).pack(anchor=tk.W)
    list_frame = tk.Frame(main)  # tk.Frame for drag-and-drop support
    list_frame.pack(fill=tk.BOTH, expand=True)
    listbox = tk.Listbox(list_frame, height=6, selectmode=tk.EXTENDED)
    scroll = ttk.Scrollbar(list_frame)
    listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scroll.pack(side=tk.RIGHT, fill=tk.Y)
    listbox.config(yscrollcommand=scroll.set)
    scroll.config(command=listbox.yview)

    # Drag and drop: make list_frame a drop target
    if DND_AVAILABLE:
        list_frame.drop_target_register(DND_FILES)
        list_frame.dnd_bind("<<Drop>>", lambda e: _on_drop(e, listbox, image_paths, output_dir))

    btn_frame = ttk.Frame(main)
    btn_frame.pack(fill=tk.X, pady=(4, 0))
    ttk.Button(btn_frame, text="Add images…", command=lambda: add_images(listbox, image_paths, output_dir)).pack(side=tk.LEFT, padx=(0, 6))
    ttk.Button(btn_frame, text="Clear list", command=lambda: clear_list(listbox, image_paths)).pack(side=tk.LEFT)

    ttk.Separator(main, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=12)

    opts = ttk.LabelFrame(main, text="Options", padding=8)
    opts.pack(fill=tk.X)

    # Shared: pixels inward from edge
    row1 = ttk.Frame(opts)
    row1.pack(fill=tk.X)
    ttk.Label(row1, text="Edge depth (pixels):").pack(side=tk.LEFT, padx=(0, 6))
    pixels_var = tk.IntVar(value=3)
    pixels_spin = tk.Spinbox(row1, from_=1, to=50, width=6, textvariable=pixels_var)
    pixels_spin.pack(side=tk.LEFT)
    ttk.Label(row1, text="  (inward from outer edge)").pack(side=tk.LEFT, padx=(4, 0))

    # Mode: one of three
    mode_var = tk.StringVar(value="black")
    row_mode = ttk.Frame(opts)
    row_mode.pack(fill=tk.X, pady=(8, 0))
    ttk.Radiobutton(row_mode, text="Black outline — snap near-black to black in edge band", variable=mode_var, value="black").pack(anchor=tk.W)
    row_mode2 = ttk.Frame(opts)
    row_mode2.pack(fill=tk.X)
    ttk.Radiobutton(row_mode2, text="Remove transparent (edge) — semi-transparent pixels in edge band only", variable=mode_var, value="transparent").pack(anchor=tk.W)
    row_mode3 = ttk.Frame(opts)
    row_mode3.pack(fill=tk.X)
    ttk.Radiobutton(row_mode3, text="Remove transparent (all) — check ALL pixels, remove if alpha ≤ threshold", variable=mode_var, value="transparent_all").pack(anchor=tk.W)

    # Black threshold (only for black outline mode)
    row_black = ttk.Frame(opts)
    row_black.pack(fill=tk.X, pady=(6, 0))
    ttk.Label(row_black, text="Black threshold (0–255):").pack(side=tk.LEFT, padx=(0, 6))
    threshold_var = tk.IntVar(value=128)
    threshold_scale = ttk.Scale(row_black, from_=0, to=255, variable=threshold_var, orient=tk.HORIZONTAL, length=180)
    threshold_scale.pack(side=tk.LEFT, padx=(0, 8))
    th_label = ttk.Label(row_black, text="128", width=4)
    th_label.pack(side=tk.LEFT)

    def on_threshold(*_):
        try:
            v = int(threshold_var.get())
            th_label.config(text=str(min(255, max(0, v))))
        except (ValueError, tk.TclError):
            pass
    threshold_var.trace_add("write", on_threshold)

    # Transparency threshold (only for remove-transparent mode)
    row_transp = ttk.Frame(opts)
    row_transp.pack(fill=tk.X, pady=(6, 0))
    ttk.Label(row_transp, text="Remove if alpha ≤ (0–255):").pack(side=tk.LEFT, padx=(0, 6))
    transparency_var = tk.IntVar(value=128)
    transparency_scale = ttk.Scale(row_transp, from_=0, to=255, variable=transparency_var, orient=tk.HORIZONTAL, length=180)
    transparency_scale.pack(side=tk.LEFT, padx=(0, 8))
    transp_label = ttk.Label(row_transp, text="128", width=4)
    transp_label.pack(side=tk.LEFT)

    def on_transparency(*_):
        try:
            v = int(transparency_var.get())
            transp_label.config(text=str(min(255, max(0, v))))
        except (ValueError, tk.TclError):
            pass
    transparency_var.trace_add("write", on_transparency)

    def toggle_thresholds(*_):
        m = mode_var.get()
        if m == "black":
            row1.pack(fill=tk.X)
            row_black.pack(fill=tk.X, pady=(6, 0))
            row_transp.pack_forget()
        elif m == "transparent":
            row1.pack(fill=tk.X)
            row_black.pack_forget()
            row_transp.pack(fill=tk.X, pady=(6, 0))
        else:  # transparent_all
            row1.pack_forget()  # edge depth not used
            row_black.pack_forget()
            row_transp.pack(fill=tk.X, pady=(6, 0))
    mode_var.trace_add("write", toggle_thresholds)
    toggle_thresholds()  # initial state

    ttk.Separator(main, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=12)

    out_frame = ttk.Frame(main)
    out_frame.pack(fill=tk.X)
    ttk.Label(out_frame, text="Output folder:").pack(anchor=tk.W)
    out_entry = ttk.Entry(out_frame, textvariable=output_dir, width=50)
    out_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
    ttk.Button(out_frame, text="Browse…", command=lambda: choose_output(output_dir)).pack(side=tk.LEFT)

    save_band_preview_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(
        main, text="Save band preview (first image only; white = pixels that will be processed)",
        variable=save_band_preview_var,
    ).pack(anchor=tk.W, pady=(8, 0))

    def run():
        if not image_paths:
            messagebox.showwarning("No images", "Add at least one image.")
            return
        out = output_dir.get().strip()
        if not out:
            messagebox.showwarning("Output folder", "Choose an output folder.")
            return
        mode = mode_var.get()
        npix = 1
        if mode != "transparent_all":
            try:
                raw = pixels_spin.get().strip() or str(pixels_var.get())
                npix = int(raw)
                if npix < 1 or npix > 100:
                    raise ValueError("Edge depth must be between 1 and 100")
            except (ValueError, tk.TclError, AttributeError):
                messagebox.showwarning("Options", "Edge depth must be a number between 1 and 100.")
                return
        if mode == "black":
            try:
                th = int(round(float(threshold_var.get())))
                th = max(0, min(255, th))
            except (ValueError, tk.TclError, TypeError):
                th = 128
            process_all(image_paths, out, npix, mode="black", black_threshold=th)
        elif mode == "transparent":
            try:
                transp_th = int(round(float(transparency_var.get())))
                transp_th = max(0, min(255, transp_th))
            except (ValueError, tk.TclError, TypeError):
                transp_th = 128
            process_all(image_paths, out, npix, mode="transparent", transparency_threshold=transp_th)
        else:
            try:
                transp_th = int(round(float(transparency_var.get())))
                transp_th = max(0, min(255, transp_th))
            except (ValueError, tk.TclError, TypeError):
                transp_th = 128
            process_all(image_paths, out, npix, mode="transparent_all", transparency_threshold=transp_th)
        if save_band_preview_var.get() and image_paths and mode != "transparent_all":
            _save_band_preview(image_paths[0], out, npix)
        messagebox.showinfo("Done", f"Processed {len(image_paths)} image(s).\nSaved to:\n{out}")

    ttk.Button(main, text="Process and save", command=run).pack(pady=12)

    root.mainloop()


def _parse_dropped_paths(data: str) -> list[str]:
    """Parse event.data from DND_FILES drop (format: {path1} {path2} or {path})."""
    paths = [p.strip("{}").strip() for p in re.findall(r"\{[^}]+\}|[^\s]+", data.strip())]
    return [p for p in paths if p and os.path.isfile(p)]


def _on_drop(event, listbox: tk.Listbox, paths: list, output_dir: tk.StringVar):
    dropped = _parse_dropped_paths(event.data)
    _add_paths_to_list(listbox, paths, output_dir, dropped)


def add_images(listbox: tk.Listbox, paths: list, output_dir: tk.StringVar | None = None):
    files = filedialog.askopenfilenames(
        title="Select images",
        filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif"), ("All", "*.*")],
    )
    _add_paths_to_list(listbox, paths, output_dir or tk.StringVar(), list(files))


def _add_paths_to_list(
    listbox: tk.Listbox, paths: list, output_dir: tk.StringVar, new_paths: list[str]
):
    was_empty = len(paths) == 0
    first_added = None
    for p in new_paths:
        if p not in paths:
            paths.append(p)
            listbox.insert(tk.END, os.path.basename(p))
            if first_added is None:
                first_added = p
    # Default output folder: Output subfolder of first image's directory
    if was_empty and first_added and output_dir.get().strip() == "":
        parent = os.path.dirname(os.path.abspath(first_added))
        output_dir.set(os.path.join(parent, "Output"))


def clear_list(listbox: tk.Listbox, paths: list):
    paths.clear()
    listbox.delete(0, tk.END)


def choose_output(var: tk.StringVar):
    path = filedialog.askdirectory(title="Output folder")
    if path:
        var.set(path)


def _save_band_preview(first_image_path: str, output_dir: str, num_pixels: int):
    """Save band mask as PNG (white = in band) for the first image."""
    try:
        img = Image.open(first_image_path).convert("RGBA")
        _, band = _get_edge_band(img, num_pixels)
        band_uint8 = (band.astype(np.uint8)) * 255
        band_img = Image.fromarray(band_uint8, "L")
        base = os.path.splitext(os.path.basename(first_image_path))[0]
        preview_path = os.path.join(output_dir, f"{base}_band_preview.png")
        band_img.save(preview_path)
    except Exception:
        pass  # Don't block main process


def process_all(
    paths: list,
    output_dir: str,
    num_pixels: int,
    mode: str,
    black_threshold: int = 64,
    transparency_threshold: int = 128,
):
    os.makedirs(output_dir, exist_ok=True)
    for path in paths:
        try:
            img = Image.open(path).convert("RGBA")
        except Exception as e:
            messagebox.showerror("Open error", f"Could not open {path}\n{e}")
            continue
        if mode == "black":
            out_img = process_black_outline(img, num_pixels, black_threshold)
        elif mode == "transparent":
            out_img = process_remove_transparent(img, num_pixels, transparency_threshold)
        else:
            out_img = process_remove_transparent_all(img, transparency_threshold)
        base, ext = os.path.splitext(os.path.basename(path))
        suffix = "_edge_blacken" if mode == "black" else "_edge_clean" if mode == "transparent" else "_alpha_clean"
        out_path = os.path.join(output_dir, f"{base}{suffix}{ext}")
        if ext.lower() not in (".png", ".bmp", ".tiff", ".tif"):
            out_path = os.path.join(output_dir, f"{base}{suffix}.png")
        out_img.save(out_path)


if __name__ == "__main__":
    run_gui()
