'''#CineTrack  
Tracking the Cinema

CineTrack ist ein KI-gestütztes System zur Objektverfolgung im Videostream – entwickelt für Anwendungen im Bereich Filmproduktion, Live-Bildverarbeitung und automatisiertem Kamera-Tracking.

---

## Projektübersicht

Das Projekt besteht aus zwei zentralen Python-Dateien:

### 1. `Cinetrack_submission.py`
Diese Datei enthält die ausführbare KI-Anwendung, die mithilfe eines UNet-Modells ein Objekt im Kamerabild erkennt und automatisch den Fokus (Crop-Fenster) auf den vorhergesagten Mittelpunkt der Maske richtet.

Funktionen:
- Live-Kameraeingang (Webcam)
- Interaktiver Startpunkt via Mausklick
- UNet-Modellvorhersage der Objektmaske
- Automatischer Zuschnitt (Crop) auf festen Bereich (`CROP_WIDTH` x `CROP_HEIGHT`), zentriert auf dem vorhergesagten Mittelpunkt
- Darstellung des Ausschnitts in Echtzeit

---

### 2. `main_training.py`
Dieses Skript dient dem Training des UNet-Modells auf einem annotierten Datensatz mit Heatmaps.

Nach dem Training wird das Modell gespeichert und kann mit `predicting_ai_with_focus.py` geladen werden.

---

## Modell

Das verwendete UNet-Modell hat eine Größe von circa 125 MB und ist damit zu groß für einen Upload auf GitHub oder die BW-KI-Abgabeseite.

## Download des Modells:  

GitHub-Repo:
https://github.com/Mxitermi/CineTrack <--- Sehr Empfehlenswert, da dort auch der gesamte Code liegt und nicht in einer Datei wie dieser hier unübersichtlich zusammengestellt.

Google Drive (nur Modell)
https://drive.google.com/file/d/1vKMGtvg0zM8HsO5BcHBJTYJHDb-G0B_R/view?usp=drive_link

Pfad im Code anpassen:
MODEL_PATH = "C:/data/python/CineTrack_Extras/Models/unet_model_03.pth"
'''



import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms


click_point = None

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "C:/data/python/CineTrack_Extras/Models/unet_model_03.pth"

CROP_WIDTH = 400
CROP_HEIGHT = 400


# Model laden (einmalig)
model = UNet(in_channels=4, num_classes=1).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("Modell geladen.")
def crop_around_center(frame, center, width, height):
    """
    Schneidet ein Rechteck der Größe (width, height) um das gegebene Zentrum aus dem Bild aus.
    Achtet darauf, dass die Grenzen nicht überschritten werden.
    """
    h, w = frame.shape[:2]
    cx, cy = center

    x1 = max(cx - width // 2, 0)
    y1 = max(cy - height // 2, 0)
    x2 = min(x1 + width, w)
    y2 = min(y1 + height, h)

    # Anpassung, falls wir am Rand schneiden
    x1 = max(x2 - width, 0)
    y1 = max(y2 - height, 0)

    return frame[y1:y2, x1:x2]


def create_input_tensor(frame, x, y):
    """
    Wandelt das OpenCV-Bild (BGR) und Klick-Koordinaten in den Input-Tensor (C=4) für das Model um.
    """
    # BGR -> RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Bild -> Tensor (C,H,W), float 0..1
    img_tensor = transforms.ToTensor()(frame_rgb)

    h, w = frame.shape[:2]

    # Heatmap (1,H,W)
    heatmap = cgh(h, w, x, y, sigma=5).float()

    # Kombiniere Kanäle: 3 RGB + 1 Heatmap
    input_tensor = torch.cat((img_tensor, heatmap), dim=0)
    return input_tensor

def overlay_prediction(frame, input_tensor, pred_mask):
    pred_mask = torch.sigmoid(pred_mask).cpu().numpy()

    if pred_mask.ndim == 3:
        pred_mask = pred_mask.squeeze(0)

    pred_mask_bin = (pred_mask > 0.3).astype(np.uint8) * 255

    mask_color = np.zeros_like(frame)
    mask_color[:, :, 1] = pred_mask_bin  # Grün

    overlay = cv2.addWeighted(frame, 0.7, mask_color, 0.3, 0)

    center, bbox = find_mask_center(pred_mask, frame, threshold=0.3)

    if center:
        print(f"Mittelpunkt der Maske: {center}")
        cropped = crop_around_center(overlay, center, CROP_WIDTH, CROP_HEIGHT)
        cv2.imshow("Kamera", cropped)
    else:
        cv2.imshow("Kamera", overlay)


def mouse_callback(event, x, y, flags, param):
    global click_point
    if event == cv2.EVENT_LBUTTONDOWN:
        click_point = (x, y)
        print(f"Klick bei {click_point}")

def main():
    global click_point

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow("Kamera")
    cv2.setMouseCallback("Kamera", mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if click_point is not None:
            # Input-Tensor für den Klickpunkt erstellen
            input_tensor = create_input_tensor(frame, click_point[0], click_point[1]).unsqueeze(0).to(device)

            with torch.no_grad():
                pred_mask = model(input_tensor)

            # Overlay mit Maske anzeigen
            overlay_prediction(frame, input_tensor.squeeze(0), pred_mask.squeeze(0))
        else:
            # Kein Klick, nur normales Bild zeigen
            cv2.imshow("Kamera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

    import os
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import ImageOps
import numpy as np
import cv2

def create_gaussian_heatmap(h, w, x, y, sigma=10):
    """Erzeugt eine 2D-Gaussian-Heatmap mit Peak bei (x, y)."""
    heatmap = np.zeros((h, w), dtype=np.float32)
    if 0 <= x < w and 0 <= y < h:
        heatmap[int(y), int(x)] = 1
        heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=sigma, sigmaY=sigma)
        heatmap /= heatmap.max()  # normalize to [0,1]
    return torch.from_numpy(heatmap).unsqueeze(0)  # (1, H, W)


class dataloader(Dataset):
    def __init__(self, root_path, test=False):
        self.root_path = root_path
        if test:
            self.images = sorted([os.path.join(root_path, "manual_test", i) for i in os.listdir(os.path.join(root_path, "manual_test"))], key=lambda x: int(os.path.basename(x)[0]))
            self.masks = sorted([os.path.join(root_path, "manual_test_masks", i) for i in os.listdir(os.path.join(root_path, "manual_test_masks"))], key=lambda x: int(os.path.basename(x)[0]))
        else:
            self.images = sorted([os.path.join(root_path, "Pictures", i) for i in os.listdir(os.path.join(root_path, "Pictures")) if i.lower().endswith(".jpg")], key=lambda x: int(os.path.basename(x)[0]))
            self.masks = sorted([os.path.join(root_path, "Masks", i) for i in os.listdir(os.path.join(root_path, "Masks")) if i.lower().endswith(".png")], key=lambda x: int(os.path.basename(x)[0]))
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()])


    def __getitem__(self, index):
        img_path = self.images[index]
        img = Image.open(img_path)
        #img = ImageOps.exif_transpose(img).convert("RGB") weder masken noch bilder duerfen rotiert werden
        mask = Image.open(self.masks[index]).convert("L")

        img = self.transform(img) 
        mask = self.transform(mask)

        h, w = img.shape[1], img.shape[2]
        if "manual_test" or "Pictures" in img_path:
            txt_path = img_path.replace(".jpg", "_1.txt")

            with open(txt_path, 'r') as f:
                lines = f.readlines()
                val1 = float(lines[0].strip())
                val2 = float(lines[1].strip()) 
            
            orig_w, orig_h = Image.open(img_path).size
            scale_x = w / orig_w
            scale_y = h / orig_h

            x_rescaled = val1 * scale_x
            y_rescaled = val2 * scale_y

            # Heatmap erzeugen
            heatmap = create_gaussian_heatmap(h, w, x_rescaled, y_rescaled, sigma=5)

            img = torch.cat((img, heatmap), dim=0)
        return img, mask
    def __len__(self):
        return len(self.images)

import os
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import ImageOps
import numpy as np
import cv2

def create_gaussian_heatmap(h, w, x, y, sigma=10):
    """Erzeugt eine 2D-Gaussian-Heatmap mit Peak bei (x, y)."""
    heatmap = np.zeros((h, w), dtype=np.float32)
    if 0 <= x < w and 0 <= y < h:
        heatmap[int(y), int(x)] = 1
        heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=sigma, sigmaY=sigma)
        heatmap /= heatmap.max()  # normalize to [0,1]
    return torch.from_numpy(heatmap).unsqueeze(0)  # (1, H, W)


class dataloader(Dataset):
    def __init__(self, root_path, test=False):
        self.root_path = root_path
        if test:
            self.images = sorted([os.path.join(root_path, "manual_test", i) for i in os.listdir(os.path.join(root_path, "manual_test"))], key=lambda x: int(os.path.basename(x)[0]))
            self.masks = sorted([os.path.join(root_path, "manual_test_masks", i) for i in os.listdir(os.path.join(root_path, "manual_test_masks"))], key=lambda x: int(os.path.basename(x)[0]))
        else:
            self.images = sorted([os.path.join(root_path, "Pictures", i) for i in os.listdir(os.path.join(root_path, "Pictures")) if i.lower().endswith(".jpg")], key=lambda x: int(os.path.basename(x)[0]))
            self.masks = sorted([os.path.join(root_path, "Masks", i) for i in os.listdir(os.path.join(root_path, "Masks")) if i.lower().endswith(".png")], key=lambda x: int(os.path.basename(x)[0]))
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()])


    def __getitem__(self, index):
        img_path = self.images[index]
        img = Image.open(img_path)
        #img = ImageOps.exif_transpose(img).convert("RGB") weder masken noch bilder duerfen rotiert werden
        mask = Image.open(self.masks[index]).convert("L")

        img = self.transform(img) 
        mask = self.transform(mask)

        h, w = img.shape[1], img.shape[2]
        if "manual_test" or "Pictures" in img_path:
            txt_path = img_path.replace(".jpg", "_1.txt")

            with open(txt_path, 'r') as f:
                lines = f.readlines()
                val1 = float(lines[0].strip())
                val2 = float(lines[1].strip()) 
            
            orig_w, orig_h = Image.open(img_path).size
            scale_x = w / orig_w
            scale_y = h / orig_h

            x_rescaled = val1 * scale_x
            y_rescaled = val2 * scale_y

            # Heatmap erzeugen
            heatmap = create_gaussian_heatmap(h, w, x_rescaled, y_rescaled, sigma=5)

            img = torch.cat((img, heatmap), dim=0)
        return img, mask
    def __len__(self):
        return len(self.images)

import cv2
import numpy as np

def find_mask_center(pred_mask, frame=None, threshold=0.3):
    """
    Findet die Mitte der größten Maske in pred_mask (numpy array, Werte 0..1).
    Optional: Zeichnet Rechteck und Mittelpunkt in das frame (OpenCV Bild).

    Args:
        pred_mask (np.ndarray): predicted Maske als float numpy array (H,W), Werte 0..1
        frame (np.ndarray): optional, BGR Bild, um Zeichnungen vorzunehmen
        threshold (float): Schwellenwert für Binarisierung
        draw (bool): ob Rechteck und Punkt gezeichnet werden sollen

    Returns:
        center (tuple or None): Mittelpunkt (x,y) oder None, wenn keine Maske gefunden
        bbox (tuple or None): bounding box (x,y,w,h) oder None
    """
    pred_mask_bin = (pred_mask > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(pred_mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    center = (x + w // 2, y + h // 2)
    return center, (x, y, w, h)

import torch
import torch.nn as nn

from unet_parts import DoubleConv, DownSample, UpSample


class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.down_convolution_1 = DownSample(in_channels, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        self.down_convolution_4 = DownSample(256, 512)

        self.bottle_neck = DoubleConv(512, 1024)

        self.up_convolution_1 = UpSample(1024, 512)
        self.up_convolution_2 = UpSample(512, 256)
        self.up_convolution_3 = UpSample(256, 128)
        self.up_convolution_4 = UpSample(128, 64)
        
        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
       down_1, p1 = self.down_convolution_1(x)
       down_2, p2 = self.down_convolution_2(p1)
       down_3, p3 = self.down_convolution_3(p2)
       down_4, p4 = self.down_convolution_4(p3)

       b = self.bottle_neck(p4)

       up_1 = self.up_convolution_1(b, down_4)
       up_2 = self.up_convolution_2(up_1, down_3)
       up_3 = self.up_convolution_3(up_2, down_2)
       up_4 = self.up_convolution_4(up_3, down_1)

       out = self.out(up_4)
       return out

import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_op(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)

        return down, p


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
       x1 = self.up(x1)
       x = torch.cat([x1, x2], 1)
       return self.conv(x)