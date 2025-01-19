from torch.utils.data import Dataset
import cv2
import torch

class MRIDataset(Dataset):
    def __init__(self, image_paths, labels):
        self.img_paths = image_paths
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Normalize pixels in [0,1]
        image = image.astype('float32') / 255.0
        
        # Add one dimension
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)