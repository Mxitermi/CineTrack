import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

from unet import UNet
from dataset import create_gaussian_heatmap as cgh  # Heatmap-Funktion
from ai_parts import find_mask_center

click_point = None

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "C:/data/python/CineTrack_Extras/Models/unet_model_03.pth"

# Model laden (einmalig)
model = UNet(in_channels=4, num_classes=1).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("Modell geladen.")

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
    
    # Sicherstellen, dass pred_mask 2D ist (z.B. (H,W))
    if pred_mask.ndim == 3:
        pred_mask = pred_mask.squeeze(0)  # oder passende Achse
    
    pred_mask_bin = (pred_mask > 0.3).astype(np.uint8) * 255

    mask_color = np.zeros_like(frame)
    mask_color[:, :, 1] = pred_mask_bin  # Grün

    overlay = cv2.addWeighted(frame, 0.7, mask_color, 0.3, 0)
    cv2.imshow("Kamera", overlay)

    center, bbox = find_mask_center(pred_mask, frame, threshold=0.3)

    if center:
        print(f"Mittelpunkt der Maske: {center}")


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
