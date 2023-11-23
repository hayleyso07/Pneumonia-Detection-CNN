from torch.utils.data import Dataset
from PIL import Image
import torch

class XrayDataset(Dataset):
    def __init__(self, paths, labels, transformer=None):
        self.paths = paths
        self.labels = labels
        self.transformer = transformer

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        image = Image.open(path).convert("RGB")

        if self.transformer:
            image = self.transformer(image)
        
        label = self.labels[index]
        label = torch.as_tensor(label, dtype=torch.float32).view(-1)

        return image, label

    