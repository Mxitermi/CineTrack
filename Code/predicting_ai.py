import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

from unet import UNet
from dataset import create_gaussian_heatmap as cgh  # Heatmap-Funktion

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

def visualize_prediction(input_tensor, pred_mask):
    """
    Zeigt RGB-Bild, Heatmap, und predicted Mask als Matplotlib-Figure an.
    """
    input_tensor = input_tensor.cpu()
    pred_mask = pred_mask.squeeze().cpu()

    # RGB Bild: Kanäle 0-2
    rgb_img = input_tensor[:3].permute(1, 2, 0).numpy()

    # Heatmap: Kanal 3
    heatmap = input_tensor[3].numpy()

    # Sigmoid + Binarisierung der Vorhersage
    pred_mask = torch.sigmoid(pred_mask)
    pred_mask_bin = (pred_mask > 0.3).float().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(15,5))

    axs[0].imshow(rgb_img)
    axs[0].set_title("Input RGB")
    axs[0].axis("off")

    axs[1].imshow(heatmap, cmap="hot")
    axs[1].set_title("Gaussian Heatmap")
    axs[1].axis("off")

    axs[2].imshow(pred_mask_bin, cmap="gray")
    axs[2].set_title("Vorhergesagte Maske")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        frame = param.copy()
        input_tensor = create_input_tensor(frame, x, y).unsqueeze(0).to(device)  # (1,4,H,W)

        with torch.no_grad():
            pred_mask = model(input_tensor)

        visualize_prediction(input_tensor.squeeze(0), pred_mask.squeeze(0))

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow("Kamera")
    cv2.setMouseCallback("Kamera", mouse_callback, param=None)  # param wird in der Schleife gesetzt

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Aktuelles Frame an Callback übergeben
        cv2.setMouseCallback("Kamera", mouse_callback, param=frame)

        cv2.imshow("Kamera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()