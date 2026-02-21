"""
Image Cleaner - Game Asset Preparation Tool
Requires: PyQt6, ImageMagick (magick in PATH), tinify
Install:  pip install PyQt6 tinify
"""

import sys
import os
import json
import subprocess
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QLabel, QPushButton, QSpinBox, QDoubleSpinBox,
    QCheckBox, QGroupBox, QFileDialog, QListWidget, QListWidgetItem,
    QProgressBar, QFrame, QSizePolicy,
    QMessageBox, QComboBox, QGridLayout, QLineEdit, QRadioButton, QButtonGroup
)
from PyQt6.QtCore import Qt, QThread, QEvent, pyqtSignal
from PyQt6.QtGui import QPixmap, QDragEnterEvent, QDragMoveEvent, QDropEvent

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG_PATH = Path(__file__).parent / "config.json"

def load_config():
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text())
        except Exception:
            pass
    return {}

def save_config(data: dict):
    CONFIG_PATH.write_text(json.dumps(data, indent=2))


# ──────────────────────────────────────────────────────────────────────────────
#  Shared tinify helper
# ──────────────────────────────────────────────────────────────────────────────
def tinified_path(p: str, base_dir: str | None = None) -> str:
    """Return a path inside an 'Output-Tinified' subfolder with '_tinified' appended.

    base_dir: the folder that contains the *original* source image.  When given,
    Output-Tinified sits next to the originals rather than next to a processed
    intermediate (used by the pipeline where *p* is already inside Output/).
    """
    pp     = Path(p)
    folder = (Path(base_dir) if base_dir else pp.parent) / "Output-Tinified"
    os.makedirs(folder, exist_ok=True)
    return str(folder / (pp.stem + "_tinified" + pp.suffix))


def tinify_file(src: str, dst: str, api_key: str):
    """Compress *src* with TinyPNG and save the result to *dst*. Returns (ok, error_str)."""
    try:
        import tinify
    except ImportError:
        return False, "'tinify' not installed — run: pip install tinify"
    try:
        tinify.key = api_key
        tinify.from_file(src).to_file(dst)
        return True, None
    except tinify.AccountError as e:
        return False, f"TinyPNG account error: {e.message}"
    except tinify.ClientError as e:
        return False, f"TinyPNG client error: {e.message}"
    except tinify.ServerError as e:
        return False, f"TinyPNG server error: {e.message}"
    except tinify.ConnectionError:
        return False, "TinyPNG: no internet connection"
    except Exception as ex:
        return False, f"TinyPNG: {ex}"


# ──────────────────────────────────────────────────────────────────────────────
#  Worker — full ImageMagick + optional TinyPNG pipeline
# ──────────────────────────────────────────────────────────────────────────────
class ProcessWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list)          # [(src, dst, ok, err), ...]
    log      = pyqtSignal(str)

    def __init__(self, jobs, settings):
        super().__init__()
        self.jobs     = jobs
        self.settings = settings
        self._abort   = False

    def abort(self):
        self._abort = True

    def run(self):
        results = []
        total   = len(self.jobs)
        for i, (src, dst) in enumerate(self.jobs):
            if self._abort:
                break
            self.progress.emit(int(i / total * 100), Path(src).name)
            ok, err = self._process(src, dst)
            results.append((src, dst, ok, err))
            kb_str = f"  ({os.path.getsize(dst)/1024:.1f} KB)" if ok and os.path.exists(dst) else ""
            self.log.emit(f"{'✓' if ok else '✗'}  {Path(src).name}{kb_str}" + (f"  —  {err}" if err else ""))
        self.progress.emit(100, "Done")
        self.finished.emit(results)

    def _process(self, src, dst):
        s   = self.settings
        cmd = ["magick", src]

        if s["remove_bg"]:
            cmd += [
                "-alpha", "set", "-fuzz", f"{s['fuzz']}%", "-fill", "none",
                "-draw", "color 0,0 floodfill",
                "-draw", "color %[fx:w-1],0 floodfill",
                "-draw", "color 0,%[fx:h-1] floodfill",
                "-draw", "color %[fx:w-1],%[fx:h-1] floodfill",
                "-alpha", "set",
            ]
            if s["erode"] > 0:
                cmd += ["-morphology", "erode", f"square:{s['erode']}"]

        if s["resize"] and s["resize_pct"] != 100:
            cmd += ["-filter", s["resize_filter"], "-resize", f"{s['resize_pct']}%"]

        if s["quantize"]:
            if not s["dither"]:
                cmd += ["+dither"]
            cmd += ["-colors", str(s["colors"])]

        cmd.append(f"{s['output_format']}:{dst}")

        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if r.returncode != 0:
                return False, r.stderr.strip()[:200]
        except FileNotFoundError:
            return False, "ImageMagick not found — is 'magick' in your PATH?"
        except subprocess.TimeoutExpired:
            return False, "Timeout (ImageMagick)"
        except Exception as ex:
            return False, str(ex)

        # Optional TinyPNG step inside the pipeline
        if s["tinypng_in_pipeline"] and s["tinypng_key"]:
            tiny_dst = tinified_path(dst, base_dir=str(Path(src).parent))
            ok, err  = tinify_file(dst, tiny_dst, s["tinypng_key"])
            if not ok:
                return False, err
            # Remove the plain ImageMagick output; keep only the _tinified file
            try:
                os.remove(dst)
            except OSError:
                pass

        return True, None


# ──────────────────────────────────────────────────────────────────────────────
#  Worker — standalone TinyPNG-only compression
# ──────────────────────────────────────────────────────────────────────────────
class TinyWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list)
    log      = pyqtSignal(str)

    def __init__(self, paths, api_key):
        super().__init__()
        self.paths   = paths
        self.api_key = api_key
        self._abort  = False

    def abort(self):
        self._abort = True

    def run(self):
        results = []
        total   = len(self.paths)
        for i, path in enumerate(self.paths):
            if self._abort:
                break
            name = Path(path).name
            self.progress.emit(int(i / total * 100), name)
            before_kb = os.path.getsize(path) / 1024
            out_path  = tinified_path(path)
            ok, err   = tinify_file(path, out_path, self.api_key)
            if ok:
                after_kb  = os.path.getsize(out_path) / 1024
                saved     = before_kb - after_kb
                pct       = saved / before_kb * 100 if before_kb else 0
                out_name  = Path(out_path).name
                self.log.emit(f"✓  {out_name}  {before_kb:.1f} KB → {after_kb:.1f} KB  (−{pct:.0f}%)")
            else:
                self.log.emit(f"✗  {name}  —  {err}")
            results.append((out_path if ok else path, ok, err))
        self.progress.emit(100, "Done")
        self.finished.emit(results)


# ──────────────────────────────────────────────────────────────────────────────
#  Drag-and-drop list
# ──────────────────────────────────────────────────────────────────────────────
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tga", ".tiff", ".webp")

class DropListWidget(QListWidget):
    files_dropped = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        # In Qt6, QAbstractScrollArea routes drag/drop events through the
        # viewport child, not the outer widget.  Install an event filter on
        # the viewport so we intercept them reliably.
        self.setAcceptDrops(True)
        vp = self.viewport()
        if vp is not None:
            vp.setAcceptDrops(True)
            vp.installEventFilter(self)

    # ------------------------------------------------------------------
    def eventFilter(self, object, event):
        if object is self.viewport() and event is not None:
            t = event.type()
            if t == QEvent.Type.DragEnter and isinstance(event, QDragEnterEvent):
                mime = event.mimeData()
                if mime is not None and mime.hasUrls():
                    event.acceptProposedAction()
                    return True
            elif t == QEvent.Type.DragMove and isinstance(event, QDragMoveEvent):
                mime = event.mimeData()
                if mime is not None and mime.hasUrls():
                    event.acceptProposedAction()
                    return True
            elif t == QEvent.Type.Drop and isinstance(event, QDropEvent):
                mime = event.mimeData()
                if mime is not None and mime.hasUrls():
                    paths = []
                    for u in mime.urls():
                        local = u.toLocalFile()
                        if os.path.isdir(local):
                            for fname in os.listdir(local):
                                if fname.lower().endswith(IMAGE_EXTS):
                                    paths.append(os.path.join(local, fname))
                        elif local.lower().endswith(IMAGE_EXTS):
                            paths.append(local)
                    if paths:
                        self.files_dropped.emit(paths)
                    event.acceptProposedAction()
                    return True
        return super().eventFilter(object, event)


# ──────────────────────────────────────────────────────────────────────────────
#  Preview panel
# ──────────────────────────────────────────────────────────────────────────────
class PreviewPanel(QFrame):
    def __init__(self, label_text):
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)

        title = QLabel(label_text)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-weight:bold;color:#aaa;font-size:11px;letter-spacing:1px;")

        self.image_label = QLabel("—")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_label.setStyleSheet("background:#1a1a2e;border-radius:4px;")

        self.info = QLabel("")
        self.info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info.setStyleSheet("color:#666;font-size:10px;")

        lay.addWidget(title)
        lay.addWidget(self.image_label, 1)
        lay.addWidget(self.info)
        self._px = None

    def set_image(self, path):
        px = QPixmap(path)
        if px.isNull():
            self.image_label.setText("Cannot preview"); self.info.setText(""); return
        self._px = px
        self._refresh()
        self.info.setText(f"{px.width()} × {px.height()} px  •  {os.path.getsize(path)/1024:.1f} KB")

    def clear(self):
        self._px = None
        self.image_label.setText("—")
        self.info.setText("")

    def resizeEvent(self, a0):
        super().resizeEvent(a0)
        self._refresh()

    def _refresh(self):
        if self._px:
            self.image_label.setPixmap(self._px.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))


# ──────────────────────────────────────────────────────────────────────────────
#  Settings panel
# ──────────────────────────────────────────────────────────────────────────────
def make_group(title):
    g = QGroupBox(title)
    g.setStyleSheet("""
        QGroupBox {
            font-weight:bold; font-size:11px; color:#c0c8e0;
            border:1px solid #2a2d4a; border-radius:6px;
            margin-top:8px; padding-top:4px;
        }
        QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 4px; }
    """)
    return g


class SettingsPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedWidth(275)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # ── Background removal ─────────────────────────────────────────────────
        bg = make_group("Background Removal")
        gl = QGridLayout(bg); gl.setSpacing(6)
        self.cb_remove_bg = QCheckBox("Remove white background")
        self.cb_remove_bg.setChecked(True)
        gl.addWidget(self.cb_remove_bg, 0, 0, 1, 2)
        gl.addWidget(QLabel("Fuzz %"), 1, 0)
        self.spin_fuzz = QDoubleSpinBox()
        self.spin_fuzz.setRange(0, 30); self.spin_fuzz.setValue(5.0)
        self.spin_fuzz.setSingleStep(0.5); self.spin_fuzz.setSuffix(" %")
        gl.addWidget(self.spin_fuzz, 1, 1)
        gl.addWidget(QLabel("Erode (px)"), 2, 0)
        self.spin_erode = QSpinBox()
        self.spin_erode.setRange(0, 5); self.spin_erode.setValue(1)
        gl.addWidget(self.spin_erode, 2, 1)
        layout.addWidget(bg)

        # ── Resize ────────────────────────────────────────────────────────────
        rs = make_group("Resize")
        gl = QGridLayout(rs); gl.setSpacing(6)
        self.cb_resize = QCheckBox("Enable resize")
        self.cb_resize.setChecked(True)
        gl.addWidget(self.cb_resize, 0, 0, 1, 2)
        # Quick-select radio buttons
        self._resize_btn_group = QButtonGroup(self)
        radio_row = QHBoxLayout()
        radio_row.setSpacing(6)
        for pct in (25, 50, 75):
            rb = QRadioButton(f"{pct}%")
            if pct == 25:
                rb.setChecked(True)
            self._resize_btn_group.addButton(rb, pct)
            radio_row.addWidget(rb)
        radio_row.addStretch()
        gl.addLayout(radio_row, 1, 0, 1, 2)
        # Manual spin box
        gl.addWidget(QLabel("Custom %"), 2, 0)
        self.spin_resize = QSpinBox()
        self.spin_resize.setRange(1, 400); self.spin_resize.setValue(25); self.spin_resize.setSuffix(" %")
        gl.addWidget(self.spin_resize, 2, 1)
        # Keep radio and spin in sync
        self._resize_btn_group.idClicked.connect(self.spin_resize.setValue)
        self.spin_resize.valueChanged.connect(self._sync_resize_radio)
        gl.addWidget(QLabel("Filter"), 3, 0)
        self.combo_filter = QComboBox()
        self.combo_filter.addItems(["Point", "Box", "Lanczos", "Mitchell", "Sinc"])
        gl.addWidget(self.combo_filter, 3, 1)
        layout.addWidget(rs)

        # ── Color quantization ─────────────────────────────────────────────────
        q = make_group("Color Quantization")
        gl = QGridLayout(q); gl.setSpacing(6)
        self.cb_quantize = QCheckBox("Enable quantization")
        self.cb_quantize.setChecked(True)
        gl.addWidget(self.cb_quantize, 0, 0, 1, 2)
        gl.addWidget(QLabel("Colors"), 1, 0)
        self.spin_colors = QSpinBox()
        self.spin_colors.setRange(2, 256); self.spin_colors.setValue(256)
        gl.addWidget(self.spin_colors, 1, 1)
        self.cb_dither = QCheckBox("Enable dithering")
        self.cb_dither.setChecked(False)
        gl.addWidget(self.cb_dither, 2, 0, 1, 2)
        layout.addWidget(q)

        # ── TinyPNG ───────────────────────────────────────────────────────────
        tp = make_group("TinyPNG Compression")
        vl = QVBoxLayout(tp); vl.setSpacing(6)

        # API key row
        kr = QHBoxLayout()
        kr.addWidget(QLabel("API Key"))
        self.edit_apikey = QLineEdit()
        self.edit_apikey.setPlaceholderText("Paste your key here…")
        self.edit_apikey.setEchoMode(QLineEdit.EchoMode.Password)
        kr.addWidget(self.edit_apikey)
        vl.addLayout(kr)

        # Show/hide
        self.cb_show_key = QCheckBox("Show key")
        self.cb_show_key.stateChanged.connect(
            lambda s: self.edit_apikey.setEchoMode(
                QLineEdit.EchoMode.Normal if s else QLineEdit.EchoMode.Password))
        vl.addWidget(self.cb_show_key)

        # Pipeline checkbox
        self.cb_tiny_pipeline = QCheckBox("Include in pipeline (after ImageMagick)")
        self.cb_tiny_pipeline.setChecked(False)
        vl.addWidget(self.cb_tiny_pipeline)

        # Usage label
        self.lbl_usage = QLabel("Compressions this month: —")
        self.lbl_usage.setStyleSheet("color:#667;font-size:10px;")
        vl.addWidget(self.lbl_usage)

        link = QLabel('<a href="https://tinypng.com/developers" style="color:#4a80d0;">Get a free API key ↗</a>')
        link.setOpenExternalLinks(True)
        link.setStyleSheet("font-size:10px;")
        vl.addWidget(link)

        layout.addWidget(tp)

        # ── Output ─────────────────────────────────────────────────────────────
        out = make_group("Output")
        gl  = QGridLayout(out); gl.setSpacing(6)
        gl.addWidget(QLabel("Format"), 0, 0)
        self.combo_format = QComboBox()
        self.combo_format.addItems(["PNG32", "PNG", "TGA", "BMP"])
        gl.addWidget(self.combo_format, 0, 1)
        self.cb_overwrite = QCheckBox("Overwrite originals")
        self.cb_overwrite.setChecked(False)
        gl.addWidget(self.cb_overwrite, 1, 0, 1, 2)
        layout.addWidget(out)

        layout.addStretch()

        # Restore saved config
        cfg = load_config()
        if cfg.get("tinypng_key"):
            self.edit_apikey.setText(cfg["tinypng_key"])
        self.cb_tiny_pipeline.setChecked(cfg.get("tiny_in_pipeline", False))

    def get_settings(self):
        return {
            "remove_bg":          self.cb_remove_bg.isChecked(),
            "fuzz":               self.spin_fuzz.value(),
            "erode":              self.spin_erode.value(),
            "resize":             self.cb_resize.isChecked(),
            "resize_pct":         self.spin_resize.value(),
            "resize_filter":      self.combo_filter.currentText(),
            "quantize":           self.cb_quantize.isChecked(),
            "colors":             self.spin_colors.value(),
            "dither":             self.cb_dither.isChecked(),
            "tinypng_key":        self.edit_apikey.text().strip(),
            "tinypng_in_pipeline":self.cb_tiny_pipeline.isChecked(),
            "output_format":      self.combo_format.currentText(),
            "overwrite":          self.cb_overwrite.isChecked(),
        }

    def save_prefs(self):
        cfg = load_config()
        cfg["tinypng_key"]      = self.edit_apikey.text().strip()
        cfg["tiny_in_pipeline"] = self.cb_tiny_pipeline.isChecked()
        save_config(cfg)

    def _sync_resize_radio(self, value: int):
        btn = self._resize_btn_group.button(value)
        if btn:
            btn.setChecked(True)

    def refresh_usage(self):
        key = self.edit_apikey.text().strip()
        if not key:
            return
        try:
            import tinify
            tinify.key = key
            tinify.validate()
            count = tinify.compression_count or 0
            over  = count >= 500
            self.lbl_usage.setText(f"Compressions this month: {count} / 500" + ("  ⚠" if over else ""))
            self.lbl_usage.setStyleSheet(
                "color:#e05050;font-size:10px;" if over else "color:#50a070;font-size:10px;")
        except ImportError:
            self.lbl_usage.setText("Run: pip install tinify")
            self.lbl_usage.setStyleSheet("color:#a06030;font-size:10px;")
        except Exception as ex:
            self.lbl_usage.setText(f"Usage fetch failed ({ex})")
            self.lbl_usage.setStyleSheet("color:#a06030;font-size:10px;")


# ──────────────────────────────────────────────────────────────────────────────
#  Dark stylesheet
# ──────────────────────────────────────────────────────────────────────────────
DARK_STYLE = """
QWidget {
    background-color:#12131f; color:#d0d8f0;
    font-family:"Segoe UI","SF Pro Display",sans-serif; font-size:12px;
}
QPushButton {
    background:#1e2140; border:1px solid #2e3260;
    border-radius:5px; padding:6px 14px; color:#c0caea;
}
QPushButton:hover   { background:#282c55; border-color:#5060c0; }
QPushButton:pressed { background:#1a1e3a; }
QPushButton:disabled{ color:#444; border-color:#222; }
QPushButton#btn_run {
    background:#2a4a9a; border-color:#3a5aba;
    color:#e0eaff; font-weight:bold; font-size:13px; padding:8px 20px;
}
QPushButton#btn_run:hover    { background:#3555b0; }
QPushButton#btn_run:disabled { background:#1a2040; color:#445; }
QPushButton#btn_tiny {
    background:#1a3a2a; border-color:#2a6040;
    color:#80e0a0; font-weight:bold; padding:8px 14px;
}
QPushButton#btn_tiny:hover    { background:#1e4a34; border-color:#3a8060; }
QPushButton#btn_tiny:disabled { background:#121e18; color:#2a4030; }
QListWidget {
    background:#1a1b2e; border:1px solid #2a2d4a; border-radius:5px;
    alternate-background-color:#1e1f35;
}
QListWidget::item:selected { background:#2a3a7a; color:#fff; }
QGroupBox { background:transparent; }
QSpinBox, QDoubleSpinBox, QLineEdit, QComboBox {
    background:#1a1b2e; border:1px solid #2a2d4a;
    border-radius:4px; padding:3px 6px; color:#c8d4f0;
}
QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus, QComboBox:focus { border-color:#4060c0; }
QComboBox::drop-down { border:none; }
QComboBox QAbstractItemView { background:#1e2040; selection-background-color:#3050a0; }
QCheckBox { spacing:6px; }
QCheckBox::indicator {
    width:14px; height:14px; border:1px solid #3a3d60; border-radius:3px; background:#1a1b2e;
}
QCheckBox::indicator:checked { background:#3a5aba; border-color:#5070d0; }
QProgressBar {
    background:#1a1b2e; border:1px solid #2a2d4a;
    border-radius:4px; text-align:center; color:#8090c0;
}
QProgressBar::chunk { background:#3050b0; border-radius:3px; }
QScrollBar:vertical { background:#16172a; width:8px; margin:0; }
QScrollBar::handle:vertical { background:#2e3260; border-radius:4px; min-height:20px; }
QLabel { color:#c0c8e0; }
QStatusBar { background:#0e0f1c; color:#606888; font-size:11px; }
QSplitter::handle { background:#1e2040; }
"""


# ──────────────────────────────────────────────────────────────────────────────
#  Main window
# ──────────────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Cleaner — Game Asset Tool")
        self.setMinimumSize(1040, 700)
        self.resize(1240, 780)

        self._worker     = None   # current background worker
        self._files      = []

        self._build_ui()
        self._check_magick()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # Settings sidebar
        self.sp = SettingsPanel()
        self.sp.edit_apikey.editingFinished.connect(self._on_key_changed)
        self.sp.cb_tiny_pipeline.stateChanged.connect(self._on_key_changed)
        root.addWidget(self.sp)

        # Right column
        right = QVBoxLayout()
        right.setSpacing(8)

        # ── Toolbar row 1: file management ────────────────────────────────────
        tb1 = QHBoxLayout(); tb1.setSpacing(8)
        btn_add = QPushButton("＋ Add Images")
        btn_add.clicked.connect(self._add_files)
        btn_clear = QPushButton("✕ Clear List")
        btn_clear.clicked.connect(self._clear_files)
        tb1.addWidget(btn_add)
        tb1.addWidget(btn_clear)
        tb1.addStretch(1)
        right.addLayout(tb1)

        # ── Toolbar row 2: action buttons ─────────────────────────────────────
        tb2 = QHBoxLayout(); tb2.setSpacing(8)

        self.btn_run = QPushButton("▶  Process All")
        self.btn_run.setObjectName("btn_run")
        self.btn_run.clicked.connect(self._run_pipeline)

        self.btn_tiny = QPushButton("🗜  TinyPNG Selected")
        self.btn_tiny.setObjectName("btn_tiny")
        self.btn_tiny.setToolTip(
            "Compress the selected file(s) in the list directly with TinyPNG.\n"
            "Works on already-processed outputs or any PNG/JPG in the list.\n"
            "Files are compressed in-place (overwritten)."
        )
        self.btn_tiny.clicked.connect(self._run_tinypng_standalone)

        btn_tiny_all = QPushButton("🗜  TinyPNG All")
        btn_tiny_all.setObjectName("btn_tiny")
        btn_tiny_all.setToolTip("Compress ALL files in the list with TinyPNG in-place.")
        btn_tiny_all.clicked.connect(self._run_tinypng_all)

        btn_abort = QPushButton("■ Stop")
        btn_abort.setFixedWidth(65)
        btn_abort.clicked.connect(self._abort)

        tb2.addWidget(self.btn_run)
        tb2.addSpacing(12)
        tb2.addWidget(self.btn_tiny)
        tb2.addWidget(btn_tiny_all)
        tb2.addStretch()
        tb2.addWidget(btn_abort)
        right.addLayout(tb2)

        # ── Splitter: file list | before/after preview ─────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)

        list_w = QWidget()
        ll = QVBoxLayout(list_w); ll.setContentsMargins(0,0,0,0); ll.setSpacing(4)
        _hint = QLabel("Drag & drop images or folders, or use ＋ Add Images")
        _hint.setStyleSheet("color:#445;font-size:11px;")
        ll.addWidget(_hint)
        self.file_list = DropListWidget()
        self.file_list.files_dropped.connect(self._add_paths)
        self.file_list.currentItemChanged.connect(self._on_selection)
        ll.addWidget(self.file_list)
        splitter.addWidget(list_w)

        pv_w = QWidget()
        pv = QHBoxLayout(pv_w); pv.setContentsMargins(0,0,0,0); pv.setSpacing(8)
        self.prev_before = PreviewPanel("ORIGINAL")
        self.prev_after  = PreviewPanel("PROCESSED")
        pv.addWidget(self.prev_before)
        pv.addWidget(self.prev_after)
        splitter.addWidget(pv_w)

        splitter.setSizes([300, 700])
        right.addWidget(splitter, 1)

        # ── Progress + log ────────────────────────────────────────────────────
        self.progress = QProgressBar(); self.progress.setValue(0)
        right.addWidget(self.progress)

        self.log_list = QListWidget()
        self.log_list.setFixedHeight(90)
        self.log_list.setStyleSheet("font-size:11px;font-family:'Consolas','Courier New',monospace;")
        right.addWidget(self.log_list)

        root.addLayout(right, 1)
        self._status("Ready")

    def _status(self, msg: str) -> None:
        """Safe wrapper around statusBar().showMessage()."""
        sb = self.statusBar()
        if sb is not None:
            sb.showMessage(msg)

    # ── ImageMagick check ──────────────────────────────────────────────────────
    def _check_magick(self):
        try:
            r = subprocess.run(["magick", "-version"], capture_output=True, text=True, timeout=5)
            ver = r.stdout.split("\n")[0] if r.returncode == 0 else "?"
            self._status(f"✓  {ver}")
        except FileNotFoundError:
            self._status("⚠  ImageMagick not found")
            QMessageBox.warning(self, "ImageMagick Missing",
                "Could not find 'magick' on PATH.\n\nDownload: https://imagemagick.org/script/download.php")

    # ── Key persistence ────────────────────────────────────────────────────────
    def _on_key_changed(self):
        self.sp.save_prefs()
        if self.sp.edit_apikey.text().strip():
            self.sp.refresh_usage()

    # ── File management ────────────────────────────────────────────────────────
    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tga *.tiff *.webp)")
        self._add_paths(paths)

    def _add_paths(self, paths):
        existing = set(self._files)
        for p in paths:
            if p not in existing:
                self._files.append(p)
                item = QListWidgetItem(Path(p).name)
                item.setToolTip(p)
                self.file_list.addItem(item)
        self._status(f"{len(self._files)} image(s) loaded")

    def _clear_files(self):
        self._files.clear(); self.file_list.clear()
        self.prev_before.clear(); self.prev_after.clear()
        self.log_list.clear(); self.progress.setValue(0)

    # ── Preview ────────────────────────────────────────────────────────────────
    def _on_selection(self, current, _prev):
        if not current: return
        src = self._files[self.file_list.row(current)]
        self.prev_before.set_image(src)
        dst = self._make_dst(src, self.sp.get_settings())
        self.prev_after.set_image(dst) if os.path.exists(dst) else self.prev_after.clear()

    def _make_dst(self, src, s):
        p   = Path(src)
        ext = ".tga" if "TGA" in s["output_format"] else ".bmp" if "BMP" in s["output_format"] else ".png"
        if s["overwrite"]:
            return str(p.parent / (p.stem + ext))
        pct    = s["resize_pct"] if s.get("resize") else None
        suffix = f"_{pct}percent" if pct is not None else "_clean"
        folder = p.parent / "Output"
        os.makedirs(folder, exist_ok=True)
        return str(folder / (p.stem + suffix + ext))

    # ── Guard: no active worker ────────────────────────────────────────────────
    def _busy(self):
        if self._worker and self._worker.isRunning():
            QMessageBox.information(self, "Busy", "A job is already running. Stop it first.")
            return True
        return False

    # ── Pipeline run ───────────────────────────────────────────────────────────
    def _run_pipeline(self):
        if self._busy(): return
        if not self._files:
            QMessageBox.information(self, "No Files", "Add some images first."); return
        s = self.sp.get_settings()
        if s["tinypng_in_pipeline"] and not s["tinypng_key"]:
            QMessageBox.warning(self, "No API Key",
                "TinyPNG is enabled in the pipeline but no API key is set.\n"
                "Add your key in the TinyPNG section, or uncheck 'Include in pipeline'."); return
        jobs = [(src, self._make_dst(src, s)) for src in self._files]
        self._start_worker(ProcessWorker(jobs, s))

    # ── Standalone TinyPNG — selected items ───────────────────────────────────
    def _run_tinypng_standalone(self):
        if self._busy(): return
        selected = [self._files[self.file_list.row(i)] for i in self.file_list.selectedItems()]
        if not selected:
            QMessageBox.information(self, "Nothing Selected",
                "Select one or more files in the list first.\n"
                "Tip: the standalone TinyPNG button compresses files in-place."); return
        self._start_tinypng(selected)

    # ── Standalone TinyPNG — all items ────────────────────────────────────────
    def _run_tinypng_all(self):
        if self._busy(): return
        if not self._files:
            QMessageBox.information(self, "No Files", "Add some images first."); return
        self._start_tinypng(list(self._files))

    def _start_tinypng(self, paths):
        key = self.sp.edit_apikey.text().strip()
        if not key:
            QMessageBox.warning(self, "No API Key",
                "Enter your TinyPNG API key in the sidebar.\n"
                "Get one free at: https://tinypng.com/developers"); return
        self._start_worker(TinyWorker(paths, key))

    # ── Generic worker launcher ────────────────────────────────────────────────
    def _start_worker(self, worker):
        self.log_list.clear(); self.progress.setValue(0)
        self._set_buttons(False)
        self._worker = worker
        worker.progress.connect(self._on_progress)
        worker.log.connect(self._on_log)
        worker.finished.connect(self._on_finished)
        worker.start()

    def _abort(self):
        if self._worker: self._worker.abort()

    def _set_buttons(self, enabled):
        self.btn_run.setEnabled(enabled)
        self.btn_tiny.setEnabled(enabled)

    # ── Worker callbacks ───────────────────────────────────────────────────────
    def _on_progress(self, pct, name):
        self.progress.setValue(pct)
        self._status(f"Processing: {name}")

    def _on_log(self, msg):
        self.log_list.addItem(msg); self.log_list.scrollToBottom()

    def _on_finished(self, results):
        self._set_buttons(True)
        ok  = sum(1 for *_, o, _ in results if o)
        err = len(results) - ok
        self._status(f"Done — {ok} succeeded, {err} failed")
        self.sp.refresh_usage()
        cur = self.file_list.currentItem()
        if cur: self._on_selection(cur, None)


# ──────────────────────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_STYLE)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()