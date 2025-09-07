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