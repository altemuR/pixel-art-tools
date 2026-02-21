import os
import customtkinter as ctk
from tkinter import filedialog, messagebox
from wand.image import Image
from wand.color import Color
from wand.drawing import Drawing

class PixelArtPipeline(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Pixel Art Asset Prep Tool")
        self.geometry("700x650")
        ctk.set_appearance_mode("dark")
        
        # State
        self.input_path = ctk.StringVar(value="Not Selected")
        self.output_path = ctk.StringVar(value="Not Selected")
        
        self.setup_ui()

    def setup_ui(self):
        container = ctk.CTkFrame(self)
        container.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkLabel(container, text="SPRITE CLEANER & PREP", font=("Courier", 24, "bold")).pack(pady=15)

        # Folder Selection
        dir_frame = ctk.CTkFrame(container)
        dir_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkButton(dir_frame, text="Select Input", width=120, command=self.select_input).grid(row=0, column=0, padx=10, pady=5)
        ctk.CTkLabel(dir_frame, textvariable=self.input_path, text_color="gray", wraplength=400).grid(row=0, column=1, sticky="w")

        ctk.CTkButton(dir_frame, text="Select Output", width=120, command=self.select_output).grid(row=1, column=0, padx=10, pady=5)
        ctk.CTkLabel(dir_frame, textvariable=self.output_path, text_color="gray", wraplength=400).grid(row=1, column=1, sticky="w")

        # Adjustment Sliders
        adjust_frame = ctk.CTkFrame(container)
        adjust_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Fuzz
        self.fuzz_label = ctk.CTkLabel(adjust_frame, text="Transparency Fuzz: 5%", font=("Arial", 14))
        self.fuzz_label.pack(pady=(10, 0))
        self.fuzz_slider = ctk.CTkSlider(adjust_frame, from_=0, to=100, command=self.update_labels)
        self.fuzz_slider.set(5)
        self.fuzz_slider.pack(fill="x", padx=30, pady=5)

        # Resize Scale
        self.scale_label = ctk.CTkLabel(adjust_frame, text="Output Scale: 50%", font=("Arial", 14))
        self.scale_label.pack(pady=(10, 0))
        self.scale_slider = ctk.CTkSlider(adjust_frame, from_=10, to=200, command=self.update_labels)
        self.scale_slider.set(50)
        self.scale_slider.pack(fill="x", padx=30, pady=5)

        # Max Colors
        self.color_label = ctk.CTkLabel(adjust_frame, text="Max Palette Colors: 128", font=("Arial", 14))
        self.color_label.pack(pady=(10, 0))
        self.color_slider = ctk.CTkSlider(adjust_frame, from_=2, to=256, number_of_steps=254, command=self.update_labels)
        self.color_slider.set(128)
        self.color_slider.pack(fill="x", padx=30, pady=5)

        # Execute
        self.run_btn = ctk.CTkButton(container, text="START PROCESSING", 
                                     fg_color="#2c5f2d", hover_color="#1e3f1e",
                                     width=250, height=50, font=("Arial", 16, "bold"),
                                     command=self.start_processing)
        self.run_btn.pack(pady=20)

    def update_labels(self, _=None):
        self.fuzz_label.configure(text=f"Transparency Fuzz: {int(self.fuzz_slider.get())}%")
        self.scale_label.configure(text=f"Output Scale: {int(self.scale_slider.get())}%")
        self.color_label.configure(text=f"Max Palette Colors: {int(self.color_slider.get())}")

    def select_input(self):
        path = filedialog.askdirectory() or filedialog.askopenfilename()
        if path: self.input_path.set(path)

    def select_output(self):
        path = filedialog.askdirectory()
        if path: self.output_path.set(path)

    def process_logic(self, in_file, out_file):
        with Image(filename=in_file) as img:
            f_val = (self.fuzz_slider.get() / 100) * img.quantum_range
            scale_p = self.scale_slider.get() / 100.0
            colors = int(self.color_slider.get())
            
            # --- COMPATIBILITY FLOODFILL ---
            # In 0.6.13, we set fuzz on the image and use a Drawing context
            img.fuzz = f_val
            with Drawing() as draw:
                # Define coordinates for the 4 corners
                points = [(0,0), (img.width-1, 0), (0, img.height-1), (img.width-1, img.height-1)]
                
                for x, y in points:
                    try:
                        # ImageMagick 7 style
                        draw.alpha(int(x), int(y), paint_method='floodfill')
                    except AttributeError:
                        # ImageMagick 6 style fallback
                        draw.matte(int(x), int(y), paint_method='floodfill')
                
                draw(img)

            # --- PIXEL ART REFINEMENT ---
            # Remove the "white halo" usually left by floodfills
            img.morphology(method='erode', kernel='square:1')
            
            # Point resize to keep pixels sharp for your 2.5D game
            new_w = max(1, int(img.width * scale_p))
            new_h = max(1, int(img.height * scale_p))
            img.resize(width=new_w, height=new_h, filter='point')

            # Palette reduction without dithering
            img.quantize(number_colors=colors, dither=False)
            
            # Force output to PNG32 to ensure alpha channel is saved
            img.format = 'png32'
            img.save(filename=out_file)

    def start_processing(self):
        in_p = self.input_path.get()
        out_p = self.output_path.get()

        if in_p == "Not Selected" or out_p == "Not Selected":
            messagebox.showwarning("Missing Paths", "Please set input and output first.")
            return

        files = [in_p] if not os.path.isdir(in_p) else \
                [os.path.join(in_p, f) for f in os.listdir(in_p) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        try:
            for f in files:
                self.process_logic(f, os.path.join(out_p, f"prep_{os.path.basename(f)}"))
            messagebox.showinfo("Success", f"Done! Processed {len(files)} sprites.")
        except Exception as e:
            messagebox.showerror("Error", f"ImageMagick Error: {str(e)}")

if __name__ == "__main__":
    app = PixelArtPipeline()
    app.mainloop()