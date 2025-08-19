import os
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

class dataloader(Dataset):
    def __init__(self, root_path, test=False):
        self.root_path = root_path
        if test:
            self.images = sorted([root_path+"/manual_test/"+i for i in os.listdir(root_path+"/manual_test/")])
            self.masks = sorted([root_path+"/manual_test_masks/"+i for i in os.listdir(root_path+"/manual_test_masks/")])
        else:
            self.images = sorted([root_path+"/train/"+i for i in os.listdir(root_path+"/train/")])
            self.masks = sorted([root_path+"/train_masks/"+i for i in os.listdir(root_path+"/train_masks/")])
        
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        img_path = self.images[index]
        img = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("L")

        img = self.transform(img) 
        mask = self.transform(mask)

        h, w = img.shape[1], img.shape[2]
        if "manual_test" or "train" in img_path:
            txt_path = img_path.replace(".jpeg", "_1.txt")

            with open(txt_path, 'r') as f:
                lines = f.readlines()
                val1 = float(lines[0].strip())
                val2 = float(lines[1].strip()) 
            
            extra_channel_1 = torch.full((1, h, w), val1)
            extra_channel_2 = torch.full((1, h, w), val2)

            img = torch.cat((img, extra_channel_1, extra_channel_2), dim=0)
        return img, mask
    def __len__(self):
        return len(self.images)