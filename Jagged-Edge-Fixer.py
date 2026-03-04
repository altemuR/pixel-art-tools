#!/usr/bin/env python3
"""
Pixel Art Artifact Cleaner
Fixes jagged edges and outline artifacts from AI-generated pixel art
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.font as tkfont
from PIL import Image, ImageTk, ImageFilter
import numpy as np
from pathlib import Path
import threading
import os
import shutil

# ── Palette ──────────────────────────────────────────────────────────────────
BG       = "#0d0f14"
PANEL    = "#13161e"
SURFACE  = "#1a1e2a"
BORDER   = "#252a38"
ACCENT   = "#4f8ef7"
ACCENT2  = "#7c5cbf"
SUCCESS  = "#3ecf8e"
WARNING  = "#f5a623"
TEXT     = "#e8ecf4"
SUBTEXT  = "#7a8299"
HOVER    = "#1f2535"
DROP_BG  = "#111522"

# ─────────────────────────────────────────────────────────────────────────────
# CORE CLEANING ALGORITHMS
# ─────────────────────────────────────────────────────────────────────────────

def get_alpha_channel(img: Image.Image) -> np.ndarray:
    if img.mode == "RGBA":
        return np.array(img)[:, :, 3]
    arr = np.array(img.convert("RGBA"))
    return arr[:, :, 3]

def clean_pixel_art(
    img: Image.Image,
    alpha_threshold: int = 128,
    edge_smooth_passes: int = 2,
    remove_isolated: bool = True,
    isolated_radius: int = 1,
    anti_alias_strength: float = 0.5,
    fill_holes: bool = True,
    despeckle: bool = True,
) -> Image.Image:
    """Main cleaning pipeline."""
    img = img.convert("RGBA")
    arr = np.array(img, dtype=np.float32)

    alpha = arr[:, :, 3]

    # 1. Hard threshold – binarize the alpha
    mask = (alpha >= alpha_threshold).astype(np.uint8)

    # 2. Remove isolated / floating pixels
    if remove_isolated:
        mask = remove_isolated_pixels(mask, radius=isolated_radius)

    # 3. Fill small holes inside the mask
    if fill_holes:
        mask = fill_small_holes(mask)

    # 4. Edge smoothing passes
    smooth_alpha = mask.astype(np.float32) * 255.0
    for _ in range(edge_smooth_passes):
        smooth_alpha = smooth_edge_pass(smooth_alpha, mask)

    # 5. Optional gentle anti-alias on the boundary
    if anti_alias_strength > 0:
        smooth_alpha = apply_boundary_antialiasing(
            smooth_alpha, mask, strength=anti_alias_strength
        )

    # 6. Despeckle colour noise near transparent edges
    if despeckle:
        arr = despeckle_edges(arr, mask)

    arr[:, :, 3] = np.clip(smooth_alpha, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), "RGBA")


def remove_isolated_pixels(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Remove pixels whose neighborhood is mostly empty, and fill tiny gaps."""
    from scipy.ndimage import uniform_filter
    kernel_size = radius * 2 + 1
    neighbor_sum = uniform_filter(mask.astype(np.float32), size=kernel_size) * (kernel_size ** 2)
    # A fully-isolated single pixel has neighbor_sum == 1 (only itself)
    isolated = (mask == 1) & (neighbor_sum <= max(1, (kernel_size**2) * 0.15))
    result = mask.copy()
    result[isolated] = 0
    return result


def fill_small_holes(mask: np.ndarray) -> np.ndarray:
    """Fill tiny transparent holes fully surrounded by opaque pixels."""
    from scipy.ndimage import uniform_filter
    inv = (mask == 0).astype(np.float32)
    neighbor_sum = uniform_filter(inv, size=3) * 9
    # A hole pixel where all 8 neighbours are opaque → fill it
    holes = (mask == 0) & (neighbor_sum <= 1)
    result = mask.copy()
    result[holes] = 1
    return result


def smooth_edge_pass(alpha: np.ndarray, hard_mask: np.ndarray) -> np.ndarray:
    """One pass of edge-aware smoothing."""
    from scipy.ndimage import uniform_filter
    smoothed = uniform_filter(alpha, size=3)
    # Only update edge pixels – preserve fully-opaque interior
    interior = hard_mask & (uniform_filter(hard_mask.astype(np.float32), size=3) > 0.99)
    result = alpha.copy()
    result[~interior.astype(bool)] = smoothed[~interior.astype(bool)]
    return result


def apply_boundary_antialiasing(
    alpha: np.ndarray, mask: np.ndarray, strength: float = 0.5
) -> np.ndarray:
    """Feather the very boundary of the mask slightly."""
    from scipy.ndimage import uniform_filter, binary_dilation, binary_erosion
    dilated  = binary_dilation(mask, iterations=1).astype(np.float32)
    eroded   = binary_erosion(mask,  iterations=1).astype(np.float32)
    boundary = (dilated - eroded).astype(bool)

    smooth = uniform_filter(alpha, size=5)
    result = alpha.copy()
    blend  = strength
    result[boundary] = (1 - blend) * alpha[boundary] + blend * smooth[boundary]
    return result


def despeckle_edges(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fix colour bleed near transparent edges (semi-transparent fringe colours)."""
    from scipy.ndimage import binary_dilation
    # Find the 1-pixel ring just outside the mask
    dilated = binary_dilation(mask, iterations=2)
    fringe  = dilated & ~mask.astype(bool)
    result  = arr.copy()
    # For fringe pixels, blend their colour toward the nearest interior median
    if np.any(fringe):
        interior_rgb = arr[mask.astype(bool)][:, :3]
        if len(interior_rgb) > 0:
            median_rgb = np.median(interior_rgb, axis=0)
            result[fringe, :3] = (
                0.5 * result[fringe, :3] + 0.5 * median_rgb
            )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────────────────────────────────

class ImageCard(tk.Frame):
    def __init__(self, parent, path: str, on_remove, **kwargs):
        super().__init__(parent, bg=SURFACE, **kwargs)
        self.path      = path
        self.on_remove = on_remove
        self.status    = "pending"   # pending | processing | done | error
        self._build()

    def _build(self):
        self.configure(relief="flat", bd=0, highlightthickness=1,
                       highlightbackground=BORDER, highlightcolor=ACCENT)

        # Thumbnail
        try:
            img = Image.open(self.path).convert("RGBA")
            img.thumbnail((56, 56), Image.NEAREST)
            self._thumb = ImageTk.PhotoImage(img)
        except Exception:
            self._thumb = None

        left = tk.Frame(self, bg=SURFACE)
        left.pack(side="left", padx=10, pady=8)

        if self._thumb:
            tk.Label(left, image=self._thumb, bg=SURFACE).pack()

        mid = tk.Frame(self, bg=SURFACE)
        mid.pack(side="left", fill="x", expand=True, padx=6)

        name = Path(self.path).name
        tk.Label(mid, text=name, fg=TEXT, bg=SURFACE,
                 font=("Courier New", 9, "bold"),
                 anchor="w", wraplength=220).pack(anchor="w")

        try:
            img2 = Image.open(self.path)
            size_kb = os.path.getsize(self.path) // 1024
            info = f"{img2.width}×{img2.height} px  •  {size_kb} KB"
        except Exception:
            info = self.path

        tk.Label(mid, text=info, fg=SUBTEXT, bg=SURFACE,
                 font=("Courier New", 8)).pack(anchor="w")

        self.status_var = tk.StringVar(value="● PENDING")
        self.status_lbl = tk.Label(mid, textvariable=self.status_var,
                                   fg=SUBTEXT, bg=SURFACE,
                                   font=("Courier New", 8, "bold"))
        self.status_lbl.pack(anchor="w")

        # Remove button
        rm = tk.Label(self, text="✕", fg=SUBTEXT, bg=SURFACE,
                      font=("Courier New", 11), cursor="hand2", padx=10)
        rm.pack(side="right")
        rm.bind("<Button-1>", lambda e: self.on_remove(self))
        rm.bind("<Enter>",    lambda e: rm.configure(fg=WARNING))
        rm.bind("<Leave>",    lambda e: rm.configure(fg=SUBTEXT))

    def set_status(self, status: str):
        self.status = status
        colours = {
            "pending":    (SUBTEXT,  "● PENDING"),
            "processing": (WARNING,  "◈ PROCESSING…"),
            "done":       (SUCCESS,  "✓ DONE"),
            "error":      ("#e05c5c","✗ ERROR"),
        }
        col, txt = colours.get(status, (SUBTEXT, status))
        self.status_var.set(txt)
        self.status_lbl.configure(fg=col)
        self.update_idletasks()


class DropZone(tk.Frame):
    def __init__(self, parent, on_drop_files, **kwargs):
        super().__init__(parent, bg=DROP_BG, **kwargs)
        self.on_drop_files = on_drop_files
        self._active = False
        self._build()
        self._setup_drag_drop()

    def _build(self):
        self.configure(relief="flat", bd=0, highlightthickness=2,
                       highlightbackground=BORDER)
        self.inner = tk.Frame(self, bg=DROP_BG)
        self.inner.place(relx=0.5, rely=0.5, anchor="center")

        self.icon_lbl = tk.Label(self.inner, text="⬇", fg=ACCENT,
                                 bg=DROP_BG, font=("Courier New", 32))
        self.icon_lbl.pack()

        self.text_lbl = tk.Label(self.inner,
                                 text="Drag & drop images here",
                                 fg=TEXT, bg=DROP_BG,
                                 font=("Courier New", 13, "bold"))
        self.text_lbl.pack(pady=(4, 2))

        sub = tk.Label(self.inner,
                       text="or click to browse  •  PNG / WEBP / GIF / BMP",
                       fg=SUBTEXT, bg=DROP_BG,
                       font=("Courier New", 9))
        sub.pack()

        self.bind("<Button-1>", self._browse)
        self.inner.bind("<Button-1>", self._browse)
        for w in (self.icon_lbl, self.text_lbl, sub):
            w.bind("<Button-1>", self._browse)

    def _setup_drag_drop(self):
        # tkinterdnd2 if available, otherwise poll clipboard / manual browse
        try:
            self.drop_target_register("DND_Files")  # type: ignore
            self.dnd_bind("<<Drop>>", self._on_dnd_drop)            # type: ignore
            self.dnd_bind("<<DragEnter>>", self._on_dnd_enter)      # type: ignore
            self.dnd_bind("<<DragLeave>>", self._on_dnd_leave)      # type: ignore
        except Exception:
            pass

    def _on_dnd_drop(self, event):
        self._set_active(False)
        files = self._parse_dnd_data(event.data)
        self.on_drop_files(files)

    def _on_dnd_enter(self, event):
        self._set_active(True)

    def _on_dnd_leave(self, event):
        self._set_active(False)

    def _parse_dnd_data(self, data: str):
        import re
        # handles {path with spaces} or plain paths
        return [p.strip("{}") for p in re.findall(r'\{[^}]+\}|\S+', data)]

    def _set_active(self, active: bool):
        self._active = active
        col = ACCENT if active else BORDER
        self.configure(highlightbackground=col)
        self.inner.configure(bg=DROP_BG)

    def _browse(self, event=None):
        files = filedialog.askopenfilenames(
            title="Select pixel art images",
            filetypes=[("Images", "*.png *.webp *.gif *.bmp *.jpg *.jpeg"),
                       ("All files", "*.*")]
        )
        if files:
            self.on_drop_files(list(files))


class App(tk.Tk):
    def __init__(self):
        # Try tkinterdnd2 for native drag-and-drop
        try:
            import tkinterdnd2
            tkinterdnd2.Tk.__init__(self)
            self._has_dnd = True
        except ImportError:
            super().__init__()
            self._has_dnd = False

        self.title("Pixel Art Artifact Cleaner")
        self.configure(bg=BG)
        self.geometry("920x820")
        self.minsize(720, 720)

        self._cards: list[ImageCard] = []
        self._output_dir: str = ""

        self._build_ui()
        self._center()

    def _center(self):
        self.update_idletasks()
        x = (self.winfo_screenwidth()  - self.winfo_width())  // 2
        y = (self.winfo_screenheight() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Title bar
        title_bar = tk.Frame(self, bg=PANEL, height=52)
        title_bar.pack(fill="x")
        title_bar.pack_propagate(False)

        tk.Label(title_bar, text="✦  PIXEL ART ARTIFACT CLEANER",
                 fg=TEXT, bg=PANEL,
                 font=("Courier New", 13, "bold")).pack(side="left", padx=20)

        tk.Label(title_bar, text="by Claude",
                 fg=SUBTEXT, bg=PANEL,
                 font=("Courier New", 9)).pack(side="right", padx=20)

        sep = tk.Frame(self, bg=BORDER, height=1)
        sep.pack(fill="x")

        # Main horizontal layout
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=16, pady=12)

        # Left column: drop zone + image list
        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="both", expand=True)

        # Drop zone
        self.drop_zone = DropZone(left, self._add_files,
                                  height=130)
        self.drop_zone.pack(fill="x", pady=(0, 10))

        # Queue header
        queue_hdr = tk.Frame(left, bg=BG)
        queue_hdr.pack(fill="x", pady=(0, 4))
        tk.Label(queue_hdr, text="IMAGE QUEUE", fg=SUBTEXT, bg=BG,
                 font=("Courier New", 8, "bold")).pack(side="left")
        self.count_lbl = tk.Label(queue_hdr, text="0 images",
                                  fg=ACCENT, bg=BG,
                                  font=("Courier New", 8, "bold"))
        self.count_lbl.pack(side="right")

        # Scrollable card list
        list_frame = tk.Frame(left, bg=BG)
        list_frame.pack(fill="both", expand=True)

        canvas = tk.Canvas(list_frame, bg=BG, highlightthickness=0)
        scroll = ttk.Scrollbar(list_frame, orient="vertical",
                               command=canvas.yview)
        self.card_frame = tk.Frame(canvas, bg=BG)

        self.card_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.card_frame, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)

        scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        canvas.bind("<MouseWheel>",
                    lambda e: canvas.yview_scroll(-1*(e.delta//120), "units"))

        # Right column: controls
        right = tk.Frame(body, bg=PANEL, width=240)
        right.pack(side="right", fill="y", padx=(14, 0))
        right.pack_propagate(False)
        self._build_controls(right)

    def _section(self, parent, label: str) -> tk.Frame:
        tk.Label(parent, text=label, fg=ACCENT, bg=PANEL,
                 font=("Courier New", 8, "bold")).pack(anchor="w",
                                                       padx=14, pady=(10, 2))
        sep = tk.Frame(parent, bg=BORDER, height=1)
        sep.pack(fill="x", padx=14, pady=(0, 8))
        f = tk.Frame(parent, bg=PANEL)
        f.pack(fill="x", padx=14)
        return f

    def _slider_row(self, parent, label, var, from_, to, resolution=1,
                    fmt="{:.0f}", tooltip=""):
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill="x", pady=1)
        tk.Label(row, text=label, fg=TEXT, bg=PANEL,
                 font=("Courier New", 8), width=22, anchor="w").pack(side="left")
        val_lbl = tk.Label(row, textvariable=tk.StringVar(), fg=ACCENT,
                           bg=PANEL, font=("Courier New", 8, "bold"), width=5)
        val_lbl.pack(side="right")

        def _update(*_):
            val_lbl.configure(text=fmt.format(var.get()))

        var.trace_add("write", _update)

        s = ttk.Scale(parent, from_=from_, to=to, variable=var,
                      orient="horizontal")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TScale", background=PANEL, troughcolor=SURFACE,
                        sliderlength=14, sliderrelief="flat")
        s.pack(fill="x", pady=(0, 4))
        _update()
        return s

    def _check_row(self, parent, label, var):
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill="x", pady=2)
        cb = tk.Checkbutton(row, text=label, variable=var,
                            fg=TEXT, bg=PANEL, selectcolor=SURFACE,
                            activebackground=PANEL, activeforeground=TEXT,
                            font=("Courier New", 8), anchor="w",
                            highlightthickness=0, bd=0)
        cb.pack(side="left")

    def _build_controls(self, parent):
        tk.Label(parent, text="SETTINGS", fg=TEXT, bg=PANEL,
                 font=("Courier New", 10, "bold")).pack(pady=(16, 0))

        # ── Alpha / Transparency ──
        f = self._section(parent, "TRANSPARENCY")
        self.v_alpha_thresh  = tk.IntVar(value=128)
        self._slider_row(f, "Alpha threshold", self.v_alpha_thresh, 1, 254)

        # ── Edge cleaning ──
        f = self._section(parent, "EDGE CLEANING")
        self.v_smooth_passes = tk.IntVar(value=2)
        self._slider_row(f, "Smooth passes", self.v_smooth_passes, 0, 6)

        self.v_aa_strength   = tk.DoubleVar(value=0.5)
        self._slider_row(f, "Anti-alias str.", self.v_aa_strength,
                         0.0, 1.0, resolution=0.05, fmt="{:.2f}")

        # ── Artifact removal ──
        f = self._section(parent, "ARTIFACT REMOVAL")
        self.v_remove_iso    = tk.BooleanVar(value=True)
        self._check_row(f, "Remove isolated pixels", self.v_remove_iso)

        self.v_iso_radius    = tk.IntVar(value=1)
        self._slider_row(f, "Isolation radius", self.v_iso_radius, 1, 3)

        self.v_fill_holes    = tk.BooleanVar(value=True)
        self._check_row(f, "Fill small holes", self.v_fill_holes)

        self.v_despeckle     = tk.BooleanVar(value=True)
        self._check_row(f, "Despeckle edges", self.v_despeckle)

        # ── Output ──
        f = self._section(parent, "OUTPUT")
        self.v_suffix        = tk.StringVar(value="_clean")
        row = tk.Frame(f, bg=PANEL)
        row.pack(fill="x", pady=2)
        tk.Label(row, text="Filename suffix", fg=TEXT, bg=PANEL,
                 font=("Courier New", 8)).pack(anchor="w")
        tk.Entry(row, textvariable=self.v_suffix, bg=SURFACE,
                 fg=TEXT, insertbackground=TEXT,
                 relief="flat", font=("Courier New", 9),
                 highlightthickness=1, highlightbackground=BORDER).pack(
                     fill="x", pady=(2, 0))

        dir_btn = tk.Button(f, text="📁  Set output folder",
                            bg=SURFACE, fg=SUBTEXT,
                            activebackground=HOVER, activeforeground=TEXT,
                            relief="flat", font=("Courier New", 8),
                            cursor="hand2", pady=4,
                            command=self._pick_output_dir)
        dir_btn.pack(fill="x", pady=(6, 0))

        self.dir_lbl = tk.Label(f, text="(same folder as source)",
                                fg=SUBTEXT, bg=PANEL,
                                font=("Courier New", 7), wraplength=200)
        self.dir_lbl.pack(anchor="w", pady=(2, 0))

        # ── Buttons ──
        spacer = tk.Frame(parent, bg=PANEL)
        spacer.pack(fill="both", expand=True)

        clr_btn = tk.Button(parent, text="Clear Queue",
                            bg=SURFACE, fg=SUBTEXT,
                            activebackground=HOVER, activeforeground=TEXT,
                            relief="flat", font=("Courier New", 9),
                            cursor="hand2", pady=8,
                            command=self._clear_queue)
        clr_btn.pack(fill="x", padx=14, pady=(0, 4))

        self.run_btn = tk.Button(parent, text="▶  CLEAN IMAGES",
                                 bg=ACCENT, fg="#ffffff",
                                 activebackground="#3a7de8",
                                 activeforeground="#ffffff",
                                 relief="flat",
                                 font=("Courier New", 11, "bold"),
                                 cursor="hand2", pady=10,
                                 command=self._run)
        self.run_btn.pack(fill="x", padx=14, pady=(0, 8))

        # Progress bar
        style = ttk.Style()
        style.configure("Clean.Horizontal.TProgressbar",
                        troughcolor=SURFACE, background=ACCENT,
                        thickness=4)
        self.progress = ttk.Progressbar(parent, style="Clean.Horizontal.TProgressbar",
                                        orient="horizontal", mode="determinate")
        self.progress.pack(fill="x", padx=14, pady=(0, 10))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _add_files(self, paths):
        existing = {c.path for c in self._cards}
        added = 0
        for p in paths:
            p = str(p)
            if p in existing:
                continue
            ext = Path(p).suffix.lower()
            if ext not in {".png", ".webp", ".gif", ".bmp", ".jpg", ".jpeg"}:
                continue
            card = ImageCard(self.card_frame, p, self._remove_card)
            card.pack(fill="x", pady=3)
            self._cards.append(card)
            existing.add(p)
            added += 1
        self._update_count()

    def _remove_card(self, card: ImageCard):
        card.destroy()
        self._cards.remove(card)
        self._update_count()

    def _clear_queue(self):
        for c in self._cards:
            c.destroy()
        self._cards.clear()
        self._update_count()

    def _update_count(self):
        n = len(self._cards)
        self.count_lbl.configure(text=f"{n} image{'s' if n != 1 else ''}")

    def _pick_output_dir(self):
        d = filedialog.askdirectory(title="Select output folder")
        if d:
            self._output_dir = d
            short = d if len(d) < 32 else "…" + d[-30:]
            self.dir_lbl.configure(text=short)

    # ── Processing ────────────────────────────────────────────────────────────

    def _run(self):
        if not self._cards:
            messagebox.showinfo("Nothing to do", "Add some images first!")
            return
        self.run_btn.configure(state="disabled", text="⏳  WORKING…")
        self.progress["value"] = 0
        self.progress["maximum"] = len(self._cards)
        t = threading.Thread(target=self._process_all, daemon=True)
        t.start()

    def _process_all(self):
        params = dict(
            alpha_threshold   = self.v_alpha_thresh.get(),
            edge_smooth_passes= self.v_smooth_passes.get(),
            remove_isolated   = self.v_remove_iso.get(),
            isolated_radius   = self.v_iso_radius.get(),
            anti_alias_strength=self.v_aa_strength.get(),
            fill_holes        = self.v_fill_holes.get(),
            despeckle         = self.v_despeckle.get(),
        )
        suffix = self.v_suffix.get() or "_clean"

        for i, card in enumerate(self._cards):
            self.after(0, card.set_status, "processing")
            try:
                img    = Image.open(card.path)
                result = clean_pixel_art(img, **params)

                src    = Path(card.path)
                if self._output_dir:
                    out_dir = Path(self._output_dir)
                else:
                    out_dir = src.parent

                out_path = out_dir / (src.stem + suffix + ".png")
                result.save(out_path, "PNG")
                self.after(0, card.set_status, "done")
            except Exception as ex:
                print(f"[ERROR] {card.path}: {ex}")
                self.after(0, card.set_status, "error")

            self.after(0, self._tick_progress)

        self.after(0, self._done)

    def _tick_progress(self):
        self.progress["value"] += 1

    def _done(self):
        self.run_btn.configure(state="normal", text="▶  CLEAN IMAGES")
        done  = sum(1 for c in self._cards if c.status == "done")
        error = sum(1 for c in self._cards if c.status == "error")
        msg   = f"Finished!\n\n✓ {done} cleaned"
        if error:
            msg += f"\n✗ {error} failed"
        messagebox.showinfo("Done", msg)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Try to use tkinterdnd2 for real drag-and-drop
    try:
        import tkinterdnd2
        root = tkinterdnd2.Tk()
        root.withdraw()
        root.destroy()
    except ImportError:
        pass

    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()