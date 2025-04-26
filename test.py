import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import Yolov1
from resnet18 import Yolov1_Resnet18
from dataset import VOCDataset
from utils import (
    non_max_suppression,
    mean_average_precision,
    get_bboxes,
    load_checkpoint,
    plot_image,
    cellboxes_to_boxes
)
import matplotlib.pyplot as plt
import numpy as np
import os
import time

# Hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL_FILE = "checkpoints/latest.pth"
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"
use_resnet18 = True

transform = A.Compose([
    A.Resize(448, 448),
    A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def main():
    # Load model
    if use_resnet18:
        model = Yolov1_Resnet18(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    else:
        model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    
    # Load checkpoint
    try:
        checkpoint = torch.load(LOAD_MODEL_FILE)
        load_checkpoint(checkpoint, model, None)
        print(f"Successfully loaded checkpoint from epoch {checkpoint['epoch']+1}")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    # Setup test dataset and loader
    test_dataset = VOCDataset(
        "data/test.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )
    
    test_pred_boxes, test_target_boxes = get_bboxes(
        test_loader, model, iou_threshold=0.5, threshold=0.4
    )
    test_mAP = mean_average_precision(
        test_pred_boxes, test_target_boxes, iou_threshold=0.5, box_format="midpoint"
    )
    
    print(f"mAP: {test_mAP:.4f}")
    
    # Save validation results
    os.makedirs("metrics", exist_ok=True)
    epoch_num = checkpoint['epoch']
    with open('metrics/test_results.txt', 'w') as f:
        f.write(f'mAP: {test_mAP:.4f}\n')
        f.write(f'Model checkpoint epoch: {epoch_num}\n')
        f.write(f'Batch size: {BATCH_SIZE}\n')
    model.eval()
    for x, y in test_loader:
        x = x.to(DEVICE)
        for idx in range(8):
            bboxes = cellboxes_to_boxes(model(x))
            bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
            plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes, f"test_output_{idx}.png")
        break

if __name__ == "__main__":
    main() 