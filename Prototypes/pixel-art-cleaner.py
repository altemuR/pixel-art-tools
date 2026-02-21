import sys
import os
from datetime import datetime

import numpy as np
from PIL import Image
from PIL import ImageCms

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QFileDialog, QSpinBox, QVBoxLayout, QHBoxLayout,
    QMessageBox
)

# -----------------------------
# Pillow resampling compatibility
# -----------------------------
try:
    RESAMPLE = Image.Resampling.NEAREST
except AttributeError:
    RESAMPLE = Image.NEAREST


def load_palette(palette_path):
    """Extract unique RGB colors from palette image"""
    img = Image.open(palette_path).convert("RGB")
    pixels = np.array(img).reshape(-1, 3)
    unique = np.unique(pixels, axis=0)
    return unique


def rgb_to_lab(image):
    srgb = ImageCms.createProfile("sRGB")
    lab = ImageCms.createProfile("LAB")
    transform = ImageCms.buildTransformFromOpenProfiles(
        srgb, lab, "RGB", "LAB"
    )
    return np.array(ImageCms.applyTransform(image, transform))


def apply_palette(image, palette_rgb):
    # Convert image to LAB
    img_lab = rgb_to_lab(image).reshape(-1, 3).astype(np.int16)

    # Convert palette to LAB
    palette_img = Image.fromarray(
        palette_rgb.reshape(1, -1, 3).astype(np.uint8),
        "RGB"
    )
    palette_lab = rgb_to_lab(palette_img).reshape(-1, 3).astype(np.int16)

    # LAB distance
    distances = ((img_lab[:, None] - palette_lab[None, :]) ** 2).sum(axis=2)
    nearest = np.argmin(distances, axis=1)

    new_pixels = palette_rgb[nearest]

    return Image.fromarray(
        new_pixels.reshape(image.size[1], image.size[0], 3),
        "RGB"
    )



class PixelArtCleaner(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pixel Art Cleaner (Custom Palette)")
        self.setMinimumWidth(420)

        self.input_path = None
        self.output_dir = None
        self.palette_path = None

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Input image
        self.input_label = QLabel("Input image: none")
        input_btn = QPushButton("Select Input Image")
        input_btn.clicked.connect(self.select_input)

        # Palette image
        self.palette_label = QLabel("Palette image: none")
        palette_btn = QPushButton("Select Palette Image")
        palette_btn.clicked.connect(self.select_palette)

        # Output directory
        self.output_label = QLabel("Output directory: none")
        output_btn = QPushButton("Select Output Directory")
        output_btn.clicked.connect(self.select_output)

        # Target size
        size_layout = QHBoxLayout()
        self.width_spin = QSpinBox()
        self.width_spin.setRange(8, 512)
        self.width_spin.setValue(64)

        self.height_spin = QSpinBox()
        self.height_spin.setRange(8, 512)
        self.height_spin.setValue(64)

        size_layout.addWidget(QLabel("Width"))
        size_layout.addWidget(self.width_spin)
        size_layout.addWidget(QLabel("Height"))
        size_layout.addWidget(self.height_spin)

        # Run
        run_btn = QPushButton("Create Pixel Art (Using Palette)")
        run_btn.clicked.connect(self.run)

        # Assemble
        layout.addWidget(self.input_label)
        layout.addWidget(input_btn)
        layout.addSpacing(6)
        layout.addWidget(self.palette_label)
        layout.addWidget(palette_btn)
        layout.addSpacing(6)
        layout.addWidget(self.output_label)
        layout.addWidget(output_btn)
        layout.addSpacing(12)
        layout.addLayout(size_layout)
        layout.addSpacing(16)
        layout.addWidget(run_btn)

        self.setLayout(layout)

    def select_input(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if path:
            self.input_path = path
            self.input_label.setText(f"Input image: {os.path.basename(path)}")

    def select_palette(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Palette Image", "", "Images (*.png *.bmp)"
        )
        if path:
            self.palette_path = path
            self.palette_label.setText(f"Palette image: {os.path.basename(path)}")

    def select_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.output_dir = path
            self.output_label.setText(f"Output directory: {path}")

    def run(self):
        if not self.input_path or not self.output_dir or not self.palette_path:
            QMessageBox.warning(
                self,
                "Missing information",
                "Please select input image, palette image, and output directory."
            )
            return

        try:
            self.process_image()
            QMessageBox.information(self, "Done", "Pixel art created successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def process_image(self):
        w = self.width_spin.value()
        h = self.height_spin.value()

        img = Image.open(self.input_path).convert("RGB")
        img = img.resize((w, h), resample=RESAMPLE)

        palette = load_palette(self.palette_path)
        if len(palette) == 0:
            raise RuntimeError("Palette image contains no colors.")

        final_img = apply_palette(img, palette)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.splitext(os.path.basename(self.input_path))[0]

        out_path = os.path.join(
            self.output_dir,
            f"{base}_pixel_palette_{timestamp}.png"
        )

        final_img.save(out_path)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PixelArtCleaner()
    window.show()
    sys.exit(app.exec_())
