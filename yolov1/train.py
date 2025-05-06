import copy
from torch.optim.lr_scheduler import LambdaLR
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms as T
from tqdm import tqdm
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.dropout.coarse_dropout import CoarseDropout
from model import Yolov1
from resnet18 import Yolov1_Resnet18
from dataset import VOCDataset
import os
import time
import matplotlib.pyplot as plt
from datetime import datetime
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)
from loss import YoloLoss

from sklearn.model_selection import train_test_split
import pandas as pd

seed = 123
torch.manual_seed(seed)

# Hyperparameters etc. 
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 16
WEIGHT_DECAY = 1e-5
EPOCHS = 150
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = True
LOAD_MODEL_FILE = "checkpoints/latest.pth"
SAVE_LAST = False
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"

use_resnet18 = True
validate = True



train_transform = A.Compose([
    A.Resize(448, 448),
    A.Affine(scale=(0.9, 1.1), translate_percent=(0.05, 0.1), rotate=(-10, 10), shear=(-5, 5), p=0.4),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.HueSaturationValue(p=0.2),
    CoarseDropout(holes=8, max_height=32, max_width=32, fill_value=0, p=0.5),
    A.Blur(blur_limit=3, p=0.1),
    A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

val_transform = A.Compose([
    A.Resize(448, 448),
    A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")


def main():
    global LOAD_MODEL
    
    if use_resnet18:
        model = Yolov1_Resnet18(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    else:
        model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.SGD(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=0.9
    )

    early_stopping_patience = 10
    early_stopping_counter = 0
    best_model_state = None

    loss_fn = YoloLoss()

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)
    
    start_epoch = 0

    # Lists to store metrics
    epoch_times = []
    mAP_values = []
    val_mAP_values = []
    epochs = []

    if LOAD_MODEL:
        try:
            checkpoint = torch.load(LOAD_MODEL_FILE)
            load_checkpoint(checkpoint, model, optimizer)
            start_epoch = checkpoint["epoch"]+1
            epochs = checkpoint["epoch_list"]
            mAP_values = checkpoint["mAP_list"]
            val_mAP_values = checkpoint["val_mAP"]
            epoch_times = checkpoint["times"]

            for param_group in optimizer.param_groups:
                param_group.setdefault('initial_lr', LEARNING_RATE)

            print(f"Successfully loaded checkpoint from epoch {start_epoch}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print("Starting fresh...")
            LOAD_MODEL = False
            start_epoch = 0

    def lr_lambda(epoch):
        if epoch <= 5: return 1 + 9 * (epoch / 5)     # linearly from 1× to 10×
        elif epoch <= 50: return 10                   # constant 10×
        elif epoch <= 110: return 1                   # back to 1×
        elif epoch <= 130: return 0.1                 # decay to 0.1x
        else: return 0.01                              # decay to 0.01×
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=start_epoch if start_epoch > 0 else -1)

    train_dataset = VOCDataset(
        "data/train.csv",
        transform=train_transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    if validate:
        val_dataset = VOCDataset(
            "data/val.csv",
            transform=val_transform,
            img_dir=IMG_DIR,
            label_dir=LABEL_DIR
        )
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            shuffle=False,
            drop_last=False,
        )

    best_val_mAP = max(val_mAP_values) if len(val_mAP_values) > 0 else 0
    for epoch in range(start_epoch, EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        
        epoch_start_time = time.time()
        train_fn(train_loader, model, optimizer, loss_fn)
        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        
        epoch_time = time.time() - epoch_start_time
        # Calculate epoch time
        epoch_times.append(epoch_time)
        mAP_values.append(mean_avg_prec)
        epochs.append(epoch + 1)
        
        
        # save last epoch
        if epoch == EPOCHS-1 and SAVE_LAST:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "mAP": mean_avg_prec,
                "epoch_list": epochs,
                "mAP_list": mAP_values,
                "val_mAP": val_mAP_values,
                "times": epoch_times
            }
            save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
            print(f"Checkpoint saved at epoch {epoch+1} with mAP: {mean_avg_prec:.4f}")



        print(f"Train mAP: {mean_avg_prec:.4f}")
        if validate:
            with torch.no_grad():
                val_pred_boxes, val_target_boxes = get_bboxes(
                    val_loader, model, iou_threshold=0.5, threshold=0.4
                )

                val_map = mean_average_precision(
                    val_pred_boxes,
                    val_target_boxes,
                    iou_threshold=0.5,
                    box_format="midpoint",
                )

                val_mAP_values.append(val_map)

                # Step the LR scheduler
                scheduler.step(val_map)

                # Save best model
                if val_map > best_val_mAP:
                    best_val_mAP = val_map
                    best_model_state = {
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "mAP": mean_avg_prec,
                        "epoch_list": epochs,
                        "mAP_list": mAP_values,
                        "val_mAP": val_mAP_values,
                        "times": epoch_times
                    }
                    save_checkpoint(best_model_state, filename=LOAD_MODEL_FILE)
                    print(f"New best model saved with val mAP: {val_map:.4f}")
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    print(f"No improvement. Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")

                # Check early stopping
                if early_stopping_counter >= early_stopping_patience:
                    print("Early stopping triggered. Ending training...")
                    break
            print(f"Val mAP: {val_map:.4f}")
        print(f"Epoch time: {epoch_time:.2f} seconds")
        
        # Write this epoch's metrics to file
        with open('metrics/training_metrics.txt', 'a') as f:
            if epoch == 0:  # First time writing, add headers
                f.write('Epoch\tTime(s)\tmAP\tval mAP\n')
            
            val_str = f"{val_map:.4f}" if validate else "N/A"
            f.write(f'{epoch+1}\t{epoch_time:.2f}\t{mean_avg_prec:.4f}\t{val_str}\n')
        


    # save final best model
    if best_model_state:
        save_checkpoint(best_model_state, filename=LOAD_MODEL_FILE)
        print("Final best model saved.")

    # Create and save the plots
    plt.figure(figsize=(12, 5))
    
    # Plot mAP
    plt.subplot(1, 2, 1)
    plt.plot(epochs, mAP_values, label='Train mAP', color='blue')
    if val_mAP_values:
        # Align val mAPs with correct epochs
        val_epochs = [e for i, e in enumerate(epochs) if i < len(val_mAP_values)]
        plt.plot(val_epochs, val_mAP_values, label='Val mAP', color='green')
    plt.title('mAP vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.legend()
    plt.grid(True)
    
    # Plot training time
    plt.subplot(1, 2, 2)
    plt.plot(epochs, epoch_times, 'r-')
    plt.title('Training Time vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('metrics/training_metrics.png')
    plt.close()

    # Save metrics to a file
    with open('metrics/training_metrics.txt', 'w') as f:
        f.write('Epoch\tTime(s)\tmAP\tval mAP\n')
        for e, t, m, v in zip(epochs, epoch_times, mAP_values, val_mAP_values):
            f.write(f'{e}\t{t:.2f}\t{m:.4f}\t{v:.4f}\n')

if __name__ == "__main__":
    main()