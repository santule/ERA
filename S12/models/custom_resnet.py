from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()


        self.prep_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        ) # output_size = 64,32,32

        self.layer1_x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        ) # output_size = 128,16,16

        self.layer1_r = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        ) #output_size = 128,16,16

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        ) # output_size = 256,8,8


        self.layer3_x = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        ) # output_size = 512,4,4

        self.layer3_r = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        ) #output_size = 512,4,4

        self.last_maxpool = nn.Sequential(
           nn.MaxPool2d(4, 4), #512

        )
        self.last_fc =  nn.Sequential(
            nn.Linear(512,10,bias=False)
        )

    def forward(self, x):
        x = self.prep_layer(x)
        x = self.layer1_x(x)
        x_layer1_identity = x.clone()
        x = self.layer1_r(x)
        #x = F.relu(x + x_layer1_identity)
        x = x + x_layer1_identity
        x = self.layer2(x)
        x = self.layer3_x(x)
        x_layer3_identity = x.clone()
        x = self.layer3_r(x)
        #x = F.relu(x + x_layer3_identity)
        x = x + x_layer3_identity
        x = self.last_maxpool(x)
        x = x.view(-1,512)
        x = self.last_fc(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
