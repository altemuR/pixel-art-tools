"""
Image Cleaner - Game Asset Preparation Tool
Requires: PyQt6, ImageMagick (magick in PATH)
Install:  pip install PyQt6
"""

import sys
import os
import subprocess
import shutil
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QLabel, QPushButton, QSlider, QSpinBox, QDoubleSpinBox,
    QCheckBox, QGroupBox, QFileDialog, QListWidget, QListWidgetItem,
    QProgressBar, QStatusBar, QScrollArea, QFrame, QSizePolicy,
    QMessageBox, QComboBox, QTabWidget, QGridLayout, QLineEdit
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer
from PyQt6.QtGui import QPixmap, QIcon, QFont, QColor, QPalette, QDragEnterEvent, QDropEvent


# ──────────────────────────────────────────────
#  Worker thread — runs magick commands off-UI
# ──────────────────────────────────────────────
class ProcessWorker(QThread):
    progress   = pyqtSignal(int, str)   # (percent, filename)
    finished   = pyqtSignal(list)       # list of (src, dst, ok, error)
    log        = pyqtSignal(str)

    def __init__(self, jobs, settings):
        super().__init__()
        self.jobs     = jobs      # list of (input_path, output_path)
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
            pct  = int(i / total * 100)
            name = Path(src).name
            self.progress.emit(pct, name)
            ok, err = self._process(src, dst)
            results.append((src, dst, ok, err))
            self.log.emit(f"{'✓' if ok else '✗'}  {name}" + (f"  —  {err}" if err else ""))
        self.progress.emit(100, "Done")
        self.finished.emit(results)

    def _process(self, src, dst):
        s   = self.settings
        cmd = ["magick", src]

        # 1. Remove white background
        if s["remove_bg"]:
            fuzz = s["fuzz"]
            cmd += [
                "-alpha", "set",
                "-fuzz",  f"{fuzz}%",
                "-fill",  "none",
                "-draw",  "color 0,0 floodfill",
                "-draw",  "color %[fx:w-1],0 floodfill",
                "-draw",  "color 0,%[fx:h-1] floodfill",
                "-draw",  "color %[fx:w-1],%[fx:h-1] floodfill",
                "-alpha", "set",
            ]
            if s["erode"] > 0:
                cmd += ["-morphology", "erode", f"square:{s['erode']}"]

        # 2. Resize
        if s["resize"] and s["resize_pct"] != 100:
            pct    = s["resize_pct"]
            filter_ = s["resize_filter"]
            cmd += ["-filter", filter_, "-resize", f"{pct}%"]

        # 3. Color quantization
        if s["quantize"]:
            if not s["dither"]:
                cmd += ["+dither"]
            cmd += ["-colors", str(s["colors"])]

        # Output format
        fmt = s["output_format"]
        cmd.append(f"{fmt}:{dst}")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60
            )
            if result.returncode != 0:
                return False, result.stderr.strip()[:200]
            return True, None
        except FileNotFoundError:
            return False, "ImageMagick not found — is 'magick' in your PATH?"
        except subprocess.TimeoutExpired:
            return False, "Timeout"
        except Exception as ex:
            return False, str(ex)


# ──────────────────────────────────────────────
#  Drag-and-drop list widget
# ──────────────────────────────────────────────
class DropListWidget(QListWidget):
    files_dropped = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setDragDropMode(QListWidget.DragDropMode.NoDragDrop)
        self.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)

    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dragMoveEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e: QDropEvent):
        paths = []
        for url in e.mimeData().urls():
            p = url.toLocalFile()
            if p.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tga", ".tiff", ".webp")):
                paths.append(p)
        if paths:
            self.files_dropped.emit(paths)


# ──────────────────────────────────────────────
#  Image preview panel
# ──────────────────────────────────────────────
class PreviewPanel(QFrame):
    def __init__(self, label_text):
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setSpacing(4)

        self.title = QLabel(label_text)
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title.setStyleSheet("font-weight: bold; color: #aaa; font-size: 11px; letter-spacing: 1px;")

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_label.setStyleSheet("background: #1a1a2e; border-radius: 4px;")
        self.image_label.setText("—")

        self.info = QLabel("")
        self.info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info.setStyleSheet("color: #666; font-size: 10px;")

        self._layout.addWidget(self.title)
        self._layout.addWidget(self.image_label, 1)
        self._layout.addWidget(self.info)

        self._pixmap = None

    def set_image(self, path):
        px = QPixmap(path)
        if px.isNull():
            self.image_label.setText("Cannot preview")
            self.info.setText("")
            return
        self._pixmap = px
        self._refresh()
        mb = os.path.getsize(path) / 1024
        self.info.setText(f"{px.width()} × {px.height()} px  •  {mb:.1f} KB")

    def clear(self):
        self._pixmap = None
        self.image_label.setText("—")
        self.info.setText("")

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._refresh()

    def _refresh(self):
        if self._pixmap:
            scaled = self._pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.image_label.setPixmap(scaled)


# ──────────────────────────────────────────────
#  Settings panel
# ──────────────────────────────────────────────
def make_group(title):
    g = QGroupBox(title)
    g.setStyleSheet("""
        QGroupBox {
            font-weight: bold;
            font-size: 11px;
            color: #c0c8e0;
            border: 1px solid #2a2d4a;
            border-radius: 6px;
            margin-top: 8px;
            padding-top: 4px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px;
        }
    """)
    return g


class SettingsPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedWidth(260)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # ── Background removal ──────────────────
        bg_group = make_group("Background Removal")
        bg_layout = QGridLayout(bg_group)
        bg_layout.setSpacing(6)

        self.cb_remove_bg = QCheckBox("Remove white background")
        self.cb_remove_bg.setChecked(True)
        bg_layout.addWidget(self.cb_remove_bg, 0, 0, 1, 2)

        bg_layout.addWidget(QLabel("Fuzz %"), 1, 0)
        self.spin_fuzz = QDoubleSpinBox()
        self.spin_fuzz.setRange(0, 30)
        self.spin_fuzz.setValue(5.0)
        self.spin_fuzz.setSingleStep(0.5)
        self.spin_fuzz.setSuffix(" %")
        bg_layout.addWidget(self.spin_fuzz, 1, 1)

        bg_layout.addWidget(QLabel("Erode (px)"), 2, 0)
        self.spin_erode = QSpinBox()
        self.spin_erode.setRange(0, 5)
        self.spin_erode.setValue(1)
        bg_layout.addWidget(self.spin_erode, 2, 1)

        layout.addWidget(bg_group)

        # ── Resize ─────────────────────────────
        rs_group = make_group("Resize")
        rs_layout = QGridLayout(rs_group)
        rs_layout.setSpacing(6)

        self.cb_resize = QCheckBox("Enable resize")
        self.cb_resize.setChecked(True)
        rs_layout.addWidget(self.cb_resize, 0, 0, 1, 2)

        rs_layout.addWidget(QLabel("Scale %"), 1, 0)
        self.spin_resize = QSpinBox()
        self.spin_resize.setRange(1, 400)
        self.spin_resize.setValue(50)
        self.spin_resize.setSuffix(" %")
        rs_layout.addWidget(self.spin_resize, 1, 1)

        rs_layout.addWidget(QLabel("Filter"), 2, 0)
        self.combo_filter = QComboBox()
        self.combo_filter.addItems(["Point", "Box", "Lanczos", "Mitchell", "Sinc"])
        rs_layout.addWidget(self.combo_filter, 2, 1)

        layout.addWidget(rs_group)

        # ── Color quantization ──────────────────
        q_group = make_group("Color Quantization")
        q_layout = QGridLayout(q_group)
        q_layout.setSpacing(6)

        self.cb_quantize = QCheckBox("Enable quantization")
        self.cb_quantize.setChecked(True)
        q_layout.addWidget(self.cb_quantize, 0, 0, 1, 2)

        q_layout.addWidget(QLabel("Colors"), 1, 0)
        self.spin_colors = QSpinBox()
        self.spin_colors.setRange(2, 256)
        self.spin_colors.setValue(128)
        q_layout.addWidget(self.spin_colors, 1, 1)

        self.cb_dither = QCheckBox("Enable dithering")
        self.cb_dither.setChecked(False)
        q_layout.addWidget(self.cb_dither, 2, 0, 1, 2)

        layout.addWidget(q_group)

        # ── Output ─────────────────────────────
        out_group = make_group("Output")
        out_layout = QGridLayout(out_group)
        out_layout.setSpacing(6)

        out_layout.addWidget(QLabel("Format"), 0, 0)
        self.combo_format = QComboBox()
        self.combo_format.addItems(["PNG32", "PNG", "TGA", "BMP"])
        out_layout.addWidget(self.combo_format, 0, 1)

        out_layout.addWidget(QLabel("Suffix"), 1, 0)
        self.edit_suffix = QLineEdit("_clean")
        out_layout.addWidget(self.edit_suffix, 1, 1)

        self.cb_overwrite = QCheckBox("Overwrite originals")
        self.cb_overwrite.setChecked(False)
        out_layout.addWidget(self.cb_overwrite, 2, 0, 1, 2)

        layout.addWidget(out_group)
        layout.addStretch()

    def get_settings(self):
        return {
            "remove_bg":     self.cb_remove_bg.isChecked(),
            "fuzz":          self.spin_fuzz.value(),
            "erode":         self.spin_erode.value(),
            "resize":        self.cb_resize.isChecked(),
            "resize_pct":    self.spin_resize.value(),
            "resize_filter": self.combo_filter.currentText(),
            "quantize":      self.cb_quantize.isChecked(),
            "colors":        self.spin_colors.value(),
            "dither":        self.cb_dither.isChecked(),
            "output_format": self.combo_format.currentText(),
            "suffix":        self.edit_suffix.text().strip(),
            "overwrite":     self.cb_overwrite.isChecked(),
        }


# ──────────────────────────────────────────────
#  Main window
# ──────────────────────────────────────────────
DARK_STYLE = """
QWidget {
    background-color: #12131f;
    color: #d0d8f0;
    font-family: "Segoe UI", "SF Pro Display", sans-serif;
    font-size: 12px;
}
QPushButton {
    background: #1e2140;
    border: 1px solid #2e3260;
    border-radius: 5px;
    padding: 6px 14px;
    color: #c0caea;
}
QPushButton:hover   { background: #282c55; border-color: #5060c0; }
QPushButton:pressed { background: #1a1e3a; }
QPushButton:disabled{ color: #444; border-color: #222; }
QPushButton#btn_run {
    background: #2a4a9a;
    border-color: #3a5aba;
    color: #e0eaff;
    font-weight: bold;
    font-size: 13px;
    padding: 8px 20px;
}
QPushButton#btn_run:hover   { background: #3555b0; }
QPushButton#btn_run:disabled{ background: #1a2040; color: #445; }
QListWidget {
    background: #1a1b2e;
    border: 1px solid #2a2d4a;
    border-radius: 5px;
    alternate-background-color: #1e1f35;
}
QListWidget::item:selected { background: #2a3a7a; color: #fff; }
QGroupBox { background: transparent; }
QSpinBox, QDoubleSpinBox, QLineEdit, QComboBox {
    background: #1a1b2e;
    border: 1px solid #2a2d4a;
    border-radius: 4px;
    padding: 3px 6px;
    color: #c8d4f0;
}
QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus, QComboBox:focus {
    border-color: #4060c0;
}
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView { background: #1e2040; selection-background-color: #3050a0; }
QCheckBox { spacing: 6px; }
QCheckBox::indicator {
    width: 14px; height: 14px;
    border: 1px solid #3a3d60;
    border-radius: 3px;
    background: #1a1b2e;
}
QCheckBox::indicator:checked { background: #3a5aba; border-color: #5070d0; }
QProgressBar {
    background: #1a1b2e;
    border: 1px solid #2a2d4a;
    border-radius: 4px;
    text-align: center;
    color: #8090c0;
}
QProgressBar::chunk { background: #3050b0; border-radius: 3px; }
QScrollBar:vertical {
    background: #16172a; width: 8px; margin: 0;
}
QScrollBar::handle:vertical {
    background: #2e3260; border-radius: 4px; min-height: 20px;
}
QLabel { color: #c0c8e0; }
QStatusBar { background: #0e0f1c; color: #606888; font-size: 11px; }
QSplitter::handle { background: #1e2040; }
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Cleaner — Game Asset Tool")
        self.setMinimumSize(1000, 680)
        self.resize(1200, 740)

        self._worker    = None
        self._files     = []      # list of absolute paths
        self._output_dir = None

        self._build_ui()
        self._check_magick()

    # ── Build UI ───────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # Settings sidebar
        self.settings_panel = SettingsPanel()
        root.addWidget(self.settings_panel)

        # Main area (splitter: file list + previews)
        right = QVBoxLayout()
        right.setSpacing(8)

        # Toolbar row
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        btn_add    = QPushButton("＋ Add Images")
        btn_add.clicked.connect(self._add_files)
        btn_clear  = QPushButton("✕ Clear List")
        btn_clear.clicked.connect(self._clear_files)
        btn_outdir = QPushButton("📁 Output Folder…")
        btn_outdir.clicked.connect(self._pick_outdir)
        self.lbl_outdir = QLabel("Same folder as input")
        self.lbl_outdir.setStyleSheet("color: #556; font-size: 11px;")

        toolbar.addWidget(btn_add)
        toolbar.addWidget(btn_clear)
        toolbar.addWidget(btn_outdir)
        toolbar.addWidget(self.lbl_outdir, 1)

        self.btn_run = QPushButton("▶  Process All")
        self.btn_run.setObjectName("btn_run")
        self.btn_run.setFixedWidth(150)
        self.btn_run.clicked.connect(self._run)
        btn_abort = QPushButton("■ Stop")
        btn_abort.setFixedWidth(70)
        btn_abort.clicked.connect(self._abort)
        toolbar.addWidget(self.btn_run)
        toolbar.addWidget(btn_abort)
        right.addLayout(toolbar)

        # Splitter: file list (left) | preview (right)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # File list
        list_widget = QWidget()
        list_layout = QVBoxLayout(list_widget)
        list_layout.setContentsMargins(0, 0, 0, 0)
        list_layout.setSpacing(4)
        lbl_drop = QLabel("Drag & drop images here, or use ＋ Add Images")
        lbl_drop.setStyleSheet("color: #445; font-size: 11px;")
        self.file_list = DropListWidget()
        self.file_list.files_dropped.connect(self._add_paths)
        self.file_list.currentItemChanged.connect(self._on_selection)
        list_layout.addWidget(lbl_drop)
        list_layout.addWidget(self.file_list)
        splitter.addWidget(list_widget)

        # Before / After previews
        preview_widget = QWidget()
        pv_layout = QHBoxLayout(preview_widget)
        pv_layout.setContentsMargins(0, 0, 0, 0)
        pv_layout.setSpacing(8)
        self.preview_before = PreviewPanel("ORIGINAL")
        self.preview_after  = PreviewPanel("PROCESSED")
        pv_layout.addWidget(self.preview_before)
        pv_layout.addWidget(self.preview_after)
        splitter.addWidget(preview_widget)

        splitter.setSizes([320, 680])
        right.addWidget(splitter, 1)

        # Progress + log
        self.progress = QProgressBar()
        self.progress.setValue(0)
        right.addWidget(self.progress)

        self.log_list = QListWidget()
        self.log_list.setFixedHeight(90)
        self.log_list.setStyleSheet("font-size: 11px; font-family: 'Consolas','Courier New',monospace;")
        right.addWidget(self.log_list)

        root.addLayout(right, 1)

        self.statusBar().showMessage("Ready  •  ImageMagick required")

    # ── ImageMagick check ─────────────────────
    def _check_magick(self):
        try:
            r = subprocess.run(["magick", "-version"], capture_output=True, text=True, timeout=5)
            ver = r.stdout.split("\n")[0] if r.returncode == 0 else "?"
            self.statusBar().showMessage(f"✓  {ver}")
        except FileNotFoundError:
            self.statusBar().showMessage("⚠  ImageMagick not found — install it and ensure 'magick' is in PATH")
            QMessageBox.warning(self, "ImageMagick Missing",
                "Could not find 'magick' on PATH.\n\nDownload from: https://imagemagick.org/script/download.php")

    # ── File management ───────────────────────
    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tga *.tiff *.webp)"
        )
        self._add_paths(paths)

    def _add_paths(self, paths):
        existing = set(self._files)
        for p in paths:
            if p not in existing:
                self._files.append(p)
                item = QListWidgetItem(Path(p).name)
                item.setToolTip(p)
                self.file_list.addItem(item)
        self.statusBar().showMessage(f"{len(self._files)} image(s) loaded")

    def _clear_files(self):
        self._files.clear()
        self.file_list.clear()
        self.preview_before.clear()
        self.preview_after.clear()
        self.log_list.clear()
        self.progress.setValue(0)

    def _pick_outdir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if d:
            self._output_dir = d
            self.lbl_outdir.setText(f"→ {d}")

    # ── Preview on selection ──────────────────
    def _on_selection(self, current, _previous):
        if not current:
            return
        idx = self.file_list.row(current)
        src = self._files[idx]
        self.preview_before.set_image(src)
        # Look for a previously processed output
        dst = self._make_dst(src, self.settings_panel.get_settings())
        if os.path.exists(dst):
            self.preview_after.set_image(dst)
        else:
            self.preview_after.clear()

    # ── Build output path ─────────────────────
    def _make_dst(self, src, s):
        p      = Path(src)
        suffix = s["suffix"] if not s["overwrite"] else ""
        ext    = ".png" if "PNG" in s["output_format"] else \
                 ".tga" if "TGA" in s["output_format"] else \
                 ".bmp" if "BMP" in s["output_format"] else ".png"
        name   = p.stem + suffix + ext
        folder = Path(self._output_dir) if self._output_dir else p.parent
        return str(folder / name)

    # ── Run ───────────────────────────────────
    def _run(self):
        if not self._files:
            QMessageBox.information(self, "No Files", "Add some images first.")
            return
        s    = self.settings_panel.get_settings()
        jobs = [(src, self._make_dst(src, s)) for src in self._files]
        self.log_list.clear()
        self.progress.setValue(0)
        self.btn_run.setEnabled(False)

        self._worker = ProcessWorker(jobs, s)
        self._worker.progress.connect(self._on_progress)
        self._worker.log.connect(self._on_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _abort(self):
        if self._worker:
            self._worker.abort()

    # ── Worker signals ────────────────────────
    def _on_progress(self, pct, name):
        self.progress.setValue(pct)
        self.statusBar().showMessage(f"Processing: {name}")

    def _on_log(self, msg):
        self.log_list.addItem(msg)
        self.log_list.scrollToBottom()

    def _on_finished(self, results):
        self.btn_run.setEnabled(True)
        ok_count  = sum(1 for _, _, ok, _ in results if ok)
        err_count = len(results) - ok_count
        self.statusBar().showMessage(
            f"Done — {ok_count} succeeded, {err_count} failed"
        )
        # Refresh after-preview for selected item
        cur = self.file_list.currentItem()
        if cur:
            self._on_selection(cur, None)


# ──────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_STYLE)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
