import os
from PIL import Image, ImageTk
import tkinter as tk

# Pfade zu den Ordnern (anpassen!)
jpg_folder = "Samples/Pictures"
png_folder = "Samples/Masks"

jpg_files = sorted(
    [f for f in os.listdir(jpg_folder) if f.endswith('.jpg')],
    key=lambda x: int(os.path.splitext(x)[0])
)

png_files = sorted(
    [f for f in os.listdir(png_folder) if f.endswith('.png')],
    key=lambda x: int(os.path.splitext(x)[0])
)

class ImageViewer:
    def __init__(self, root):
        self.root = root
        self.index = 0

        self.label = tk.Label(root)
        self.label.pack()

        btn_next = tk.Button(root, text="Weiter", command=self.next_image)
        btn_weirdBild = tk.Button(root, text="Weird Bild", command=self.weirdBild)
        btn_weirdMaske = tk.Button(root, text="Weird Maske", command=self.weirdMaske)
        btn_next.pack()
        btn_weirdBild.pack()
        btn_weirdMaske.pack()

        self.show_image()

    def show_image(self):
        # Bilder laden ohne automatische EXIF-Rotation
        jpg_path = os.path.join(jpg_folder, jpg_files[self.index])
        png_path = os.path.join(png_folder, png_files[self.index])

        # JPG laden, keine automatische Drehung (Pillow macht das nicht automatisch)
        jpg_img = Image.open(jpg_path)

        # PNG laden
        png_img = Image.open(png_path)

        # Optional: Bildgröße anpassen, falls zu groß
        max_size = (500, 500)
        jpg_img.thumbnail(max_size)
        png_img.thumbnail(max_size)

        # Bilder nebeneinander zusammenfügen (horizontal)
        total_width = jpg_img.width + png_img.width
        max_height = max(jpg_img.height, png_img.height)

        combined = Image.new('RGB', (total_width, max_height))
        combined.paste(jpg_img, (0, 0))
        combined.paste(png_img, (jpg_img.width, 0))

        # Bild für tkinter vorbereiten
        self.tk_img = ImageTk.PhotoImage(combined)

        # Bild im Label anzeigen
        self.label.config(image=self.tk_img)

    def next_image(self):
        self.index = (self.index + 1) % len(jpg_files)
        self.show_image()
    def weirdMaske(self):
        print(f"Weirdes Maske:  {png_files[self.index]}")
    def weirdBild(self):
        print(f"Weirdes Bild: {jpg_files[self.index]} ")

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Bilderanzeige ohne EXIF-Rotation")
    viewer = ImageViewer(root)
    root.mainloop()
