import torch
import torch.nn as nn
import torchvision.models as models
import ssl

class Yolov1_Resnet18(nn.Module):
    def __init__(self, split_size=7, num_boxes=2, num_classes=20, in_channels=3):
        super(Yolov1_Resnet18, self).__init__()

        # Load pretrained ResNet-18 backbone
        # resnet = models.resnet18(pretrained=True)
        ssl._create_default_https_context = ssl._create_unverified_context
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Freeze resnet layers
        # for param in resnet.parameters():
        #     param.requires_grad = False

        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # remove avgpool & fc

        # Reduce output from 14x14 to 7x7
        self.pool = nn.AdaptiveAvgPool2d((7, 7))  # output shape: [B, 512, 7, 7]

        # YOLO head
        self.fcs = nn.Sequential(
            nn.Flatten(),                          # shape: [B, 512*7*7]
            nn.Linear(512 * split_size * split_size, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, split_size * split_size * (num_classes + num_boxes * 5))
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)          # Make sure spatial size is 7x7
        x = self.fcs(x)
        return x