from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

class Net(nn.Module):
    def __init__(self):
        dropout_val = 0.04
        super(Net, self).__init__()

        ####### convolution block 1
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(3, 64, 3,bias=False,padding=1), # depthwise
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_val)
        ) # 64,32,32

        self.conv1_2 = nn.Sequential(
            nn.Conv2d(64, 64, 3,bias=False,padding=1,groups=64), # depthwise
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 1,bias=False), # pointwise
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_val)
        ) # 64,32,32

        self.conv1_3 = nn.Sequential(
            nn.Conv2d(64, 64, 3,bias=False,padding=1,stride=2), #dilation=8),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_val)
        ) # 64,16,16

        ####### convolution block 2
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(64, 64, 3,bias=False,padding=1,groups=64), # depthwise
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, 1,bias=False), # pointwise
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_val)
        ) # 32,16,16

        self.conv2_2 = nn.Sequential(
            nn.Conv2d(32, 32, 3,bias=False,padding=1,groups=32), # depthwise
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 1,bias=False), # pointwise
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_val)
        ) # 64,16,16

        self.conv2_3 = nn.Sequential(
            nn.Conv2d(64, 128, 3,bias=False,padding=1,dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_val)
        ) # 64,14,14

        ####### convolution block 3
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(128, 128, 3,bias=False,padding=1,groups=128), # depthwise
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 32, 1,bias=False), # pointwise
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_val)

        ) # 32,14,14

        self.conv3_2 = nn.Sequential(
            nn.Conv2d(32, 32, 3,bias=False,padding=1,groups=32), # depthwise
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 1,bias=False), # pointwise
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_val)

        ) # 32,14,14

        self.conv3_3 = nn.Sequential(
            nn.Conv2d(32, 64, 3,bias=False,padding=1,stride=2), #dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_val)
        ) # 64,7,7


        ####### convolution block 4 - DEPTHWISE
        self.conv4_1 = nn.Sequential(
            nn.Conv2d(64, 64, 3,bias=False,padding=1,groups=64), # depthwise
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, 1,bias=False), # pointwise
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_val)
        ) # 32,7,7

        self.conv4_2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, bias=False, padding=1, groups=32),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 128, 1, bias=False), # pointwise
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_val)
        ) # 64,7,7

        self.conv4_3 = nn.Sequential(
            nn.Conv2d(128, 128, 3,bias=False,padding=1,groups=128),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 1,bias=False), # pointwise
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(dropout_val)
        ) # 256,7,7


        ####### final block
        self.final_block = nn.Sequential(
            nn.AvgPool2d(7),
            nn.Conv2d(256, 10, 1,bias=False)  # 10
        )

    def forward(self, x):
        x = self.conv1_1(x) # 16,32,32
        x = self.conv1_2(x) # 32,32,32
        x = self.conv1_3(x) # 64,16,16

        x = self.conv2_1(x) # 16,16,16
        x = self.conv2_2(x) # 32,16,16
        x = self.conv2_3(x) # 64,8,8

        x = self.conv3_1(x) # 16,8,8
        x = self.conv3_2(x) # 32,8,8
        x = self.conv3_3(x) # 64,4,4

        x = self.conv4_1(x) # 16,4,4
        x = self.conv4_2(x) # 32,4,4
        x = self.conv4_3(x) # 64,4,4

        x = self.final_block(x)
        x = x.view(-1,10)
        return F.log_softmax(x, dim=-1)
