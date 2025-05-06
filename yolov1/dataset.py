"""
Creates a Pytorch dataset to load the Pascal VOC dataset
"""

import torch
import os
import numpy as np
import pandas as pd
from PIL import Image


class VOCDataset(torch.utils.data.Dataset):
    def __init__(
        self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        class_labels = []

        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.strip().split()
                ]
                boxes.append([x, y, width, height])
                class_labels.append(int(class_label))

        # img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        # image = np.array(Image.open(img_path).convert("RGB"))

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        try:
            image = np.array(Image.open(img_path).convert("RGB"))
        except Exception as e:
            print(f"Skipping unreadable image: {img_path} ({e})")
            return self.__getitem__((index + 1) % len(self))


        if self.transform:
            transformed = self.transform(
                image=image,
                bboxes=boxes,
                class_labels=class_labels,
            )
            image = transformed["image"]
            boxes = transformed["bboxes"]
            class_labels = transformed["class_labels"]

        # Convert To Cells
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box, class_label in zip(boxes, class_labels):
            x, y, width, height = box
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            width_cell, height_cell = width * self.S, height * self.S

            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] = 1
                box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                label_matrix[i, j, 21:25] = box_coordinates
                label_matrix[i, j, int(class_label)] = 1

        return image, label_matrix
