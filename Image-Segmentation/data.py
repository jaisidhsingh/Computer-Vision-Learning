import os
import PIL
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class CarvanaDataset(Dataset):
    def __init__(self, imageDir, maskDir, transform=None):
        self.imageDir = imageDir
        self.maskDir = maskDir
        self.transform = transform
        self.images = os.listdir(imageDir)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, i):
        imgPath = os.path.join(self.imageDir, self.images[i])
        maskPath = os.path.join(self.maskDir, self.images[i].replace("jpg", "_mask.gif"))
        image = np.array(Image.open(imgPath).convert("RGB"))
        mask = np.array(Image.open(maskPath).convert("L"), dtype=np.float32)
        mask[mask==255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        return image, mask