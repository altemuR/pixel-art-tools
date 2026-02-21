import sys
import os
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QProgressBar, QTextEdit, QSpinBox, QGroupBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree


class PixelArtProcessor(QThread):
    """Thread for processing images without blocking the GUI"""
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, input_path, palette_path, output_folder, cluster_count):
        super().__init__()
        self.input_path = input_path
        self.palette_path = palette_path
        self.output_folder = output_folder
        self.cluster_count = cluster_count
        
    def load_palette(self, palette_path):
        """Load color palette from image"""
        self.status.emit("Loading color palette...")
        palette_img = Image.open(palette_path).convert('RGB')
        palette_array = np.array(palette_img)
        
        # Extract unique colors from palette
        colors = palette_array.reshape(-1, 3)
        unique_colors = np.unique(colors, axis=0)
        
        self.status.emit(f"Loaded {len(unique_colors)} colors from palette")
        return unique_colors
    
    def cluster_colors(self, img_array, n_clusters):
        """Cluster similar colors together using K-Means in LAB color space"""
        self.status.emit(f"Clustering colors into {n_clusters} groups...")
        self.progress.emit(20)
        
        # Convert RGB to LAB for perceptually uniform clustering
        # This prevents grey and green from being grouped together
        from skimage import color
        
        h, w = img_array.shape[:2]
        
        # Normalize to 0-1 range for skimage
        img_normalized = img_array / 255.0
        
        # Convert to LAB color space
        img_lab = color.rgb2lab(img_normalized)
        pixels_lab = img_lab.reshape(-1, 3)
        
        # Sample for faster clustering on large images
        if len(pixels_lab) > 50000:
            indices = np.random.choice(len(pixels_lab), 50000, replace=False)
            sample = pixels_lab[indices]
        else:
            sample = pixels_lab
        
        # Cluster in LAB space
        n_clusters = min(n_clusters, len(np.unique(sample, axis=0)))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(sample)
        
        self.progress.emit(40)
        
        # Map all pixels to cluster centers
        labels = kmeans.predict(pixels_lab)
        clustered_lab = kmeans.cluster_centers_[labels].reshape(h, w, 3)
        
        # Convert back to RGB
        clustered_rgb = color.lab2rgb(clustered_lab) * 255.0
        
        return clustered_rgb
    
    def map_to_palette(self, img_array, palette):
        """Map each pixel to nearest color in palette using LAB color space"""
        self.status.emit("Mapping to palette colors...")
        self.progress.emit(60)
        
        from skimage import color
        
        # Reshape for processing
        h, w = img_array.shape[:2]
        pixels = img_array.reshape(-1, 3) / 255.0
        
        # Convert both to LAB for perceptually accurate matching
        pixels_lab = color.rgb2lab(pixels.reshape(1, -1, 3)).reshape(-1, 3)
        palette_normalized = palette / 255.0
        palette_lab = color.rgb2lab(palette_normalized.reshape(1, -1, 3)).reshape(-1, 3)
        
        # Build KD-Tree in LAB space
        tree = cKDTree(palette_lab)
        
        # Find nearest palette color for each pixel
        _, indices = tree.query(pixels_lab)
        mapped_pixels = palette[indices]
        
        self.progress.emit(80)
        
        return mapped_pixels.reshape(h, w, 3)
    
    def run(self):
        """Main processing pipeline"""
        try:
            # Load input image
            self.status.emit("Loading input image...")
            self.progress.emit(0)
            img = Image.open(self.input_path).convert('RGB')
            img_array = np.array(img, dtype=np.float32)
            
            # Load palette
            palette = self.load_palette(self.palette_path)
            
            # Cluster similar colors
            clustered = self.cluster_colors(img_array, self.cluster_count)
            
            # Map to palette
            final = self.map_to_palette(clustered, palette)
            
            # Convert back to uint8
            final = final.astype(np.uint8)
            
            # Create output filename with timestamp
            self.status.emit("Saving output...")
            input_name = Path(self.input_path).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"{input_name}_{timestamp}_pixelart.png"
            output_path = os.path.join(self.output_folder, output_name)
            
            # Save with no filtering (NEAREST neighbor - point sampling)
            output_img = Image.fromarray(final, mode='RGB')
            output_img.save(output_path, 'PNG')
            
            self.progress.emit(100)
            self.status.emit(f"Success! Saved to: {output_name}")
            self.finished.emit(True, output_path)
            
        except Exception as e:
            self.status.emit(f"Error: {str(e)}")
            self.finished.emit(False, str(e))


class PixelArtGeneratorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.input_path = None
        self.palette_path = None
        self.output_folder = None
        self.processor = None
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Perfect Pixel Art Generator & Cleaner")
        self.setGeometry(100, 100, 700, 600)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        main_widget.setLayout(layout)
        
        # Title
        title = QLabel("Perfect Pixel Art Generator")
        title.setFont(QFont('Arial', 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Input image section
        input_group = QGroupBox("Input Image")
        input_layout = QVBoxLayout()
        
        self.input_label = QLabel("No file selected")
        self.input_label.setWordWrap(True)
        input_layout.addWidget(self.input_label)
        
        input_btn = QPushButton("Select Input Image")
        input_btn.clicked.connect(self.select_input)
        input_layout.addWidget(input_btn)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # Palette section
        palette_group = QGroupBox("Color Palette")
        palette_layout = QVBoxLayout()
        
        self.palette_label = QLabel("No palette selected")
        self.palette_label.setWordWrap(True)
        palette_layout.addWidget(self.palette_label)
        
        palette_btn = QPushButton("Select Palette Image")
        palette_btn.clicked.connect(self.select_palette)
        palette_layout.addWidget(palette_btn)
        
        palette_group.setLayout(palette_layout)
        layout.addWidget(palette_group)
        
        # Cluster settings
        cluster_group = QGroupBox("Clustering Settings")
        cluster_layout = QHBoxLayout()
        
        cluster_label = QLabel("Number of color clusters:")
        cluster_layout.addWidget(cluster_label)
        
        self.cluster_spin = QSpinBox()
        self.cluster_spin.setMinimum(2)
        self.cluster_spin.setMaximum(256)
        self.cluster_spin.setValue(32)
        self.cluster_spin.setToolTip("Higher values preserve more color variation")
        cluster_layout.addWidget(self.cluster_spin)
        
        cluster_group.setLayout(cluster_layout)
        layout.addWidget(cluster_group)
        
        # Output folder section
        output_group = QGroupBox("Output Folder")
        output_layout = QVBoxLayout()
        
        self.output_label = QLabel("No folder selected")
        self.output_label.setWordWrap(True)
        output_layout.addWidget(self.output_label)
        
        output_btn = QPushButton("Select Output Folder")
        output_btn.clicked.connect(self.select_output)
        output_layout.addWidget(output_btn)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # Process button
        self.process_btn = QPushButton("Generate Pixel Art")
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self.process_image)
        self.process_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
            QPushButton:hover:enabled {
                background-color: #45a049;
            }
        """)
        layout.addWidget(self.process_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Status log
        status_group = QGroupBox("Status Log")
        status_layout = QVBoxLayout()
        
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(150)
        status_layout.addWidget(self.status_text)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        layout.addStretch()
        
    def select_input(self):
        """Select input image file"""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Input Image", "", 
            "Images (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        if path:
            self.input_path = path
            self.input_label.setText(f"Selected: {Path(path).name}")
            self.log_status(f"Input image selected: {Path(path).name}")
            self.check_ready()
            
    def select_palette(self):
        """Select palette image file"""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Palette Image", "", 
            "Images (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        if path:
            self.palette_path = path
            self.palette_label.setText(f"Selected: {Path(path).name}")
            self.log_status(f"Palette selected: {Path(path).name}")
            self.check_ready()
            
    def select_output(self):
        """Select output folder"""
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path:
            self.output_folder = path
            self.output_label.setText(f"Selected: {path}")
            self.log_status(f"Output folder selected: {path}")
            self.check_ready()
            
    def check_ready(self):
        """Check if all inputs are ready"""
        if hasattr(self, 'process_btn') and self.process_btn is not None:
            ready = bool(self.input_path and self.palette_path and self.output_folder)
            self.process_btn.setEnabled(ready)
        
    def log_status(self, message):
        """Add message to status log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_text.append(f"[{timestamp}] {message}")
        
    def process_image(self):
        """Start image processing"""
        self.process_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        self.log_status("Starting pixel art generation...")
        
        # Create processor thread
        self.processor = PixelArtProcessor(
            self.input_path,
            self.palette_path,
            self.output_folder,
            self.cluster_spin.value()
        )
        
        # Connect signals
        self.processor.progress.connect(self.update_progress)
        self.processor.status.connect(self.log_status)
        self.processor.finished.connect(self.processing_finished)
        
        # Start processing
        self.processor.start()
        
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
        
    def processing_finished(self, success, message):
        """Handle processing completion"""
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        
        if success:
            self.log_status(f"✓ Processing complete!")
            self.log_status(f"Output saved: {Path(message).name}")
        else:
            self.log_status(f"✗ Processing failed: {message}")


def main():
    app = QApplication(sys.argv)
    window = PixelArtGeneratorGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()