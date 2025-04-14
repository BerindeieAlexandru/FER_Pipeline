import torch
import pandas as pd
import os
import cv2
from torch.utils.data import Dataset

class DataProcessor(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        df = pd.read_csv(csv_file)
        self.image_names = df.iloc[:, 0].values
        self.labels = df.iloc[:, 1].values
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = os.path.join(self.img_dir, self.image_names[idx])
        label = self.labels[idx]
        image = cv2.imread(img_name)

        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.merge([grayscale_image, grayscale_image, grayscale_image])

        if self.transform:
            image = self.transform(image=image)["image"]
    
        return image, label