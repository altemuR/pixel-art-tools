"""
Batch Content-Aware Inpainting GUI
====================================
Content-aware fill using OpenCV (built-in, fast) or LaMa AI (optional, high quality).

Requirements — CORE (always needed):
    pip install Pillow opencv-python tkinterdnd2 numpy

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
#  INPAINTING BACKENDS
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
    src_bgr   = cv2.cvtColor(rgba_arr[:, :, :3], cv2.COLOR_RGB2BGR)
    alpha_ch  = rgba_arr[:, :, 3]

    mask_np = np.array(pil_mask.convert("L"))
    if mask_np.shape[:2] != src_bgr.shape[:2]:
        mask_np = cv2.resize(mask_np,
                             (src_bgr.shape[1], src_bgr.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
    _, mask_bin = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)

    flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    result_bgr = cv2.inpaint(src_bgr, mask_bin, inpaintRadius=radius, flags=flag)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    result_pil = Image.fromarray(result_rgb)

    if has_alpha:
        result_pil = result_pil.convert("RGBA")
        result_pil.putalpha(Image.fromarray(alpha_ch))

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
        result.putalpha(pil_img.split()[3])

    return result


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
        self.geometry("920x840")
        self.minsize(800, 700)

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
        self._cancel_flag   = threading.Event()

        self._build_ui()
        self._on_engine_change()
        self.after(200, self._check_deps)

    # ── Dependency check ──────────────────────────────────────────────────────
    def _check_deps(self):
        if not CV2_AVAILABLE:
            self._log("⚠  opencv-python not found → pip install opencv-python", "warn")
        else:
            self._log("✓  OpenCV ready  (Telea & Navier-Stokes available)", "good")

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
              text="mask-driven fill  ·  OpenCV Telea / Navier-Stokes  ·  LaMa AI",
              font=FONT_SMALL, bg=BG, fg=SUBTEXT).pack(anchor=W)
        Frame(self, bg=BORDER, height=1).pack(fill=X, padx=28, pady=(10, 0))

        # Two-column body
        body = Frame(self, bg=BG)
        body.pack(fill=BOTH, expand=True, padx=28, pady=14)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2, minsize=300)
        body.rowconfigure(0, weight=1)

        left  = Frame(body, bg=BG)
        right = Frame(body, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        right.grid(row=0, column=1, sticky="nsew")

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

        SectionLabel(left, "MASK IMAGE").pack(fill=X, pady=(0, 4))
        Label(left, text="White = fill region     Black = keep region",
              font=FONT_SMALL, bg=BG, fg=SUBTEXT).pack(anchor=W, pady=(0, 4))
        self.mask_drop = DropZone(left, "Drop mask here",
                                  multi=False, on_file=self._on_mask, height=100)
        self.mask_drop.pack(fill=X)

        mask_preview_row = Frame(left, bg=BG)
        mask_preview_row.pack(fill=X, pady=(6, 0))
        self.mask_thumb_lbl = Label(mask_preview_row, bg=BG)
        self.mask_thumb_lbl.pack(side=LEFT)
        self.mask_info_lbl  = Label(mask_preview_row, text="", font=FONT_SMALL,
                                    bg=BG, fg=SUBTEXT, justify=LEFT)
        self.mask_info_lbl.pack(side=LEFT, padx=10)

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
        self.run_btn = Button(btn_row, text="▶  RUN  INPAINTING",
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
        # Auto-set output to a subfolder of the first image's directory
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

    def _on_engine_change(self):
        if self._backend.get() == "lama":
            self.radius_frame.pack_forget()
        else:
            # re-show after the engine radio section
            self.radius_frame.pack(fill=X, pady=(10, 0))

    def _browse_output(self):
        d = filedialog.askdirectory(title="Select output folder")
        if d:
            self._output_dir.set(d)

    # ── Run ───────────────────────────────────────────────────────────────────
    def _run(self):
        if not self._image_paths:
            messagebox.showwarning("No images", "Please load source images first.")
            return
        if not self._mask_path:
            messagebox.showwarning("No mask", "Please load a mask image.")
            return
        backend = self._backend.get()
        if not CV2_AVAILABLE and backend != "lama":
            messagebox.showerror("Missing dependency",
                                 "opencv-python is required for this engine.\n"
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
        out_dir  = Path(self._output_dir.get())
        out_dir.mkdir(parents=True, exist_ok=True)
        backend  = self._backend.get()
        radius   = self._radius.get()
        dilation = self._mask_dilation.get()

        # Load and optionally dilate the mask once
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
                m   = mask_pil
                if m.size != src.size:
                    m = m.resize(src.size, Image.Resampling.NEAREST)

                self.after(0, self.status_lbl.configure,
                           {"text": f"[{i+1}/{len(self._image_paths)}] {Path(p).name}…",
                            "fg": ACCENT})

                if backend in ("telea", "ns"):
                    result = inpaint_opencv(src, m, method=backend, radius=radius)
                else:
                    result = inpaint_lama(src, m)

                out_name = Path(p).stem + "_inpainted.png"
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
        self.run_btn.configure(state=NORMAL, text="▶  RUN  INPAINTING")
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