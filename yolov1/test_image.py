import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
import albumentations as A

from model import Yolov1
from resnet18 import Yolov1_Resnet18
from utils import load_checkpoint, non_max_suppression, cellboxes_to_boxes, VOC_CLASSES, plot_image

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_PATH = "example/input.jpeg"
MODEL_PATH = "checkpoints/latest.pth"
SAVE_PATH = "example/output.png"

# ==== TRANSFORM (same as used in dataset) ====
transform = A.Compose([
    A.Resize(448, 448),
    A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
    ToTensorV2(),
])

# ==== Load Image ====
def load_image(path):
    image = np.array(Image.open(path).convert("RGB"))
    transformed = transform(image=image)
    return transformed["image"].unsqueeze(0).to(DEVICE), image  # tensor, original image

# ==== Load Model ====
model = Yolov1_Resnet18(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
checkpoint = torch.load(MODEL_PATH)
load_checkpoint(checkpoint, model, None)
model.eval()

# ==== Inference ====
with torch.no_grad():
    x, original_img = load_image(IMG_PATH)
    predictions = model(x)
    bboxes = cellboxes_to_boxes(predictions)
    bboxes = non_max_suppression(bboxes[0], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
    plot_image(x[0].permute(1, 2, 0).cpu(), bboxes, filename=SAVE_PATH)
