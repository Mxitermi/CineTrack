import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from dataset import dataloader as dl
from unet import UNet

def single_image_inference(img_index, DATA_PATH, model_pth, device):
    model = UNet(in_channels=4, num_classes=1).to(device)  # Achtung: anpassen je nach Kanalanzahl
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))
    model.eval()

    dataset = dl(DATA_PATH)
    input_tensor, gt_mask = dataset[img_index]  # Eingabe + Ground Truth Maske

    # Für Vorhersage
    input_batch = input_tensor.unsqueeze(0).to(device)  # (1, C, H, W)
    with torch.no_grad():
        pred_mask = model(input_batch)

    


    # Alles auf CPU für Visualisierung
    input_tensor = input_tensor.cpu()
    pred_mask = pred_mask.squeeze(0).squeeze(0).cpu()  # (H, W)
    gt_mask = gt_mask.squeeze(0).cpu()  # (H, W)

    pred_mask = torch.sigmoid(pred_mask)  # Wahrscheinlichkeiten
    pred_mask = (pred_mask > 0.3).float()  # Binarisierung

    # RGB-Bild (Kanäle 0, 1, 2)
    rgb_img = input_tensor[:3].permute(1, 2, 0)  # (H, W, 3)

    # Gaussian-Heatmap (letzter Kanal, also -1)
    heatmap = input_tensor[-1]  # (H, W)

    # Visualisierung
    fig, axs = plt.subplots(1, 4, figsize=(18, 6))

    axs[0].imshow(rgb_img)
    axs[0].set_title("Input RGB")
    axs[0].axis("off")

    axs[1].imshow(heatmap, cmap="hot")
    axs[1].set_title("Gaussian Heatmap (Koordinaten)")
    axs[1].axis("off")

    axs[2].imshow(gt_mask, cmap="gray")
    axs[2].set_title("Ground Truth Maske")
    axs[2].axis("off")

    axs[3].imshow(pred_mask, cmap="gray")
    axs[3].set_title("Vorhergesagte Maske")
    axs[3].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    DATA_PATH = "Samples"  # ← hier auf das Verzeichnis, nicht das Einzelbild zeigen
    MODEL_PATH = "Models/unet_model_03.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    single_image_inference(60, DATA_PATH, MODEL_PATH, device)
