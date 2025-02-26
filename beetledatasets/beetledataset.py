from pathlib import Path
import pandas as pd
import ast
import numpy as np
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize, Resize


def global_to_local(global_coords, beetle_box_coords):
    x1, y1, x2, y2 = beetle_box_coords
    x, y = global_coords
    return x - x1, y - y1

class beetleIndividualDataset(Dataset):
    def __init__(self, image_base_dir, gt_file_path, input_size=224, horizontal_flip=None, vertical_flip=None):
        self.image_base_dir = Path(image_base_dir)
        self.image_paths = list(self.image_base_dir.rglob('*.png'))
        self.gt_file_path = Path(gt_file_path)
        self.input_size = input_size
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip

        self.transform = Compose([
            Resize((self.input_size, self.input_size)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.df = pd.read_csv(self.gt_file_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        # image = Image.open(image_path)
        # w, h = image.size
        image = cv2.imread(str(image_path))
        h, w, _ = image.shape

        image_name = image_path.name
        # Check if the image exists in the ground truth file
        if image_name not in self.df['BeetleID'].values:
            raise ValueError(f'{image_name} is not in the ground truth file')
        
        # Get the ground truth information
        row = self.df[self.df['BeetleID'] == image_name].iloc[0]
        dim_ori_img = ast.literal_eval(row["dim_ori_img"])
        coords_beetle_box = ast.literal_eval(row["coords_beetle_box"])
        coords_len = ast.literal_eval(row["coords_len"])
        coords_width = ast.literal_eval(row["coords_width"])

        # Compute the local coordinates of coords_len, coords_width
        coords_len_local = (global_to_local(coords_len[:2], coords_beetle_box), global_to_local(coords_len[2:], coords_beetle_box))
        coords_width_local = (global_to_local(coords_width[:2], coords_beetle_box), global_to_local(coords_width[2:], coords_beetle_box))

        # Horizontal flip
        if self.horizontal_flip:
            if np.random.rand() < self.horizontal_flip:
                image = cv2.flip(image, 1)
                coords_len_local = ((w - coords_len_local[1][0], coords_len_local[0][1]), (w - coords_len_local[0][0], coords_len_local[1][1]))
                coords_width_local = ((w - coords_width_local[1][0], coords_width_local[0][1]), (w - coords_width_local[0][0], coords_width_local[1][1]))

        # Vertical flip
        if self.vertical_flip:
            if np.random.rand() < self.vertical_flip:
                image = cv2.flip(image, 0)
                coords_len_local = ((coords_len_local[0][0], h - coords_len_local[1][1]), (coords_len_local[1][0], h - coords_len_local[0][1]))
                coords_width_local = ((coords_width_local[0][0], h - coords_width_local[1][1]), (coords_width_local[1][0], h - coords_width_local[0][1]))

        max_dim = max(w, h)
        # Pad the image to square add zero padding to the right or bottom
        if w > h:
            pad = np.zeros((w, w, 3), dtype=np.uint8)
            pad[:h, :, :] = image
            image = pad
        elif h > w:
            pad = np.zeros((h, h, 3), dtype=np.uint8)
            pad[:, :w, :] = image
            image = pad

        input_image = Image.fromarray(image)

        # Image to tensor
        input_image_tensor = self.transform(input_image)

        # Make ground truth one-dimensional vector
        (x1, y1), (x2, y2) = coords_len_local
        (x3, y3), (x4, y4) = coords_width_local

        x1, y1, x2, y2, x3, y3, x4, y4 = x1 / max_dim, y1 / max_dim, x2 / max_dim, y2 / max_dim, x3 / max_dim, y3 / max_dim, x4 / max_dim, y4 / max_dim

        # Make ground truth tensor to 0-1 scale referring to the input size
        # x1, y1, x2, y2, x3, y3, x4, y4 = x1 / self.input_size, y1 / self.input_size, x2 / self.input_size, y2 / self.input_size, x3 / self.input_size, y3 / self.input_size, x4 / self.input_size, y4 / self.input_size

        gt = torch.tensor([x1, y1, x2, y2, x3, y3, x4, y4], dtype=torch.float32)

        return {'image_name': image_name, 'image': input_image_tensor, 'gt': gt}


if __name__ == '__main__':
    image_base_dir = '/data-archive/2018-NEON-beetles/predictbeetle/individual_images_train'
    gt_file_path = '/data-archive/2018-NEON-beetles/predictbeetle/individual_images.csv'

    dataset = beetleIndividualDataset(image_base_dir, gt_file_path)
    print(len(dataset))
    sample = dataset[0]
    print(sample['image'].shape)
    print(sample['gt'])

    for i in range(len(dataset)):
        print(dataset[i]['gt'])