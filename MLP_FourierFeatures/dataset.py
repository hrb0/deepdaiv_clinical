import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import imageio
import numpy as np
from kornia import create_meshgrid
from einops import rearrange
import cv2
from typing import Tuple


class ImageDataset(Dataset):
    def __init__(self, image_path: str, img_wh: Tuple[int, int], split: str):
        image = imageio.imread(image_path)[..., :3]/255.
        image = cv2.resize(image, img_wh)
        # image = np.load('images/data_2d_text.npz')['test_data'][6]/255.

        self.uv = create_meshgrid(*image.shape[:2], True)[0]
        self.rgb = torch.FloatTensor(image)

        if split == 'train':
            self.uv = self.uv[::2, ::2]
            self.rgb = self.rgb[::2, ::2]
        elif split == 'val':
            self.uv = self.uv[1::2, 1::2]
            self.rgb = self.rgb[1::2, 1::2]

        self.uv = rearrange(self.uv, 'h w c -> (h w) c')
        self.rgb = rearrange(self.rgb, 'h w c -> (h w) c')

    def __len__(self):
        return len(self.uv)

    def __getitem__(self, idx: int):
        return {"uv": self.uv[idx], "rgb": self.rgb[idx]}
