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
from torchvision import transforms
import torch.nn as nn

# === Globale Variablen ===
click_point = None
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "C:/data/python/CineTrack_Extras/Models/unet_model_03.pth"

CROP_WIDTH = 400
CROP_HEIGHT = 400


# === Hilfsfunktionen ===
def create_gaussian_heatmap(h, w, x, y, sigma=10):
    heatmap = np.zeros((h, w), dtype=np.float32)
    if 0 <= x < w and 0 <= y < h:
        heatmap[int(y), int(x)] = 1
        heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=sigma, sigmaY=sigma)
        heatmap /= heatmap.max()
    return torch.from_numpy(heatmap).unsqueeze(0)  # (1, H, W)


def crop_around_center(frame, center, width, height):
    h, w = frame.shape[:2]
    cx, cy = center
    x1 = max(cx - width // 2, 0)
    y1 = max(cy - height // 2, 0)
    x2 = min(x1 + width, w)
    y2 = min(y1 + height, h)
    x1 = max(x2 - width, 0)
    y1 = max(y2 - height, 0)
    return frame[y1:y2, x1:x2]


def create_input_tensor(frame, x, y):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = transforms.ToTensor()(frame_rgb)
    h, w = frame.shape[:2]
    heatmap = create_gaussian_heatmap(h, w, x, y, sigma=5).float()
    input_tensor = torch.cat((img_tensor, heatmap), dim=0)
    return input_tensor


def find_mask_center(pred_mask, frame=None, threshold=0.3):
    pred_mask_bin = (pred_mask > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(pred_mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    center = (x + w // 2, y + h // 2)
    return center, (x, y, w, h)


def overlay_prediction(frame, input_tensor, pred_mask):
    pred_mask = torch.sigmoid(pred_mask).cpu().numpy()
    if pred_mask.ndim == 3:
        pred_mask = pred_mask.squeeze(0)

    pred_mask_bin = (pred_mask > 0.3).astype(np.uint8) * 255
    mask_color = np.zeros_like(frame)
    mask_color[:, :, 1] = pred_mask_bin  # Grün
    overlay = cv2.addWeighted(frame, 0.7, mask_color, 0.3, 0)

    center, _ = find_mask_center(pred_mask, frame, threshold=0.3)

    if center:
        print(f"[INFO] Mittelpunkt der Maske: {center}")
        cropped = crop_around_center(overlay, center, CROP_WIDTH, CROP_HEIGHT)
        cv2.imshow("Kamera", cropped)
    else:
        cv2.imshow("Kamera", overlay)


def mouse_callback(event, x, y, flags, param):
    global click_point
    if event == cv2.EVENT_LBUTTONDOWN:
        click_point = (x, y)
        print(f"[INFO] Klick bei {click_point}")


# === UNet Architektur ===
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
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.down1 = DownSample(in_channels, 64)
        self.down2 = DownSample(64, 128)
        self.down3 = DownSample(128, 256)
        self.down4 = DownSample(256, 512)

        self.bottleneck = DoubleConv(512, 1024)

        self.up1 = UpSample(1024, 512)
        self.up2 = UpSample(512, 256)
        self.up3 = UpSample(256, 128)
        self.up4 = UpSample(128, 64)

        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        d1, p1 = self.down1(x)
        d2, p2 = self.down2(p1)
        d3, p3 = self.down3(p2)
        d4, p4 = self.down4(p3)

        b = self.bottleneck(p4)

        u1 = self.up1(b, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)

        return self.out(u4)


# === Hauptfunktion ===
def main():
    global click_point

    # Modell laden
    model = UNet(in_channels=4, num_classes=1).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("[INFO] Modell geladen.")

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
            input_tensor = create_input_tensor(frame, click_point[0], click_point[1]).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_mask = model(input_tensor)
            overlay_prediction(frame, input_tensor.squeeze(0), pred_mask.squeeze(0))
        else:
            cv2.imshow("Kamera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
