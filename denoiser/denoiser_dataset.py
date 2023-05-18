import os

from torch.utils.data import Dataset
from PIL import Image


class DenoiserDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, transform=None):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.transform = transform

        self.clean_files = os.listdir(clean_dir)
        self.noisy_files = os.listdir(noisy_dir)

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, index):
        clean_img_path = os.path.join(self.clean_dir, self.clean_files[index])
        noisy_img_path = os.path.join(self.noisy_dir, self.noisy_files[index])

        clean_img = Image.open(clean_img_path).convert("RGB")
        noisy_img = Image.open(noisy_img_path).convert("RGB")

        if self.transform:
            clean_img = self.transform(clean_img)
            noisy_img = self.transform(noisy_img)

        return noisy_img, clean_img
