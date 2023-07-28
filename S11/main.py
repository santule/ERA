import utils as utils
from models import resnet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch_lr_finder import LRFinder
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch.nn.functional as F
from torchsummary import summary
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image,scale_cam_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import cv2
from PIL import Image



EPOCHS = 20

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

print(f"----------Load and Transform Images----------")
train_loader,test_loader = utils.load_dataset()


print(f"----------Build Model----------")
model_check = resnet.ResNet18().to(device)
utils.summarise_model(model_check)

print(f"----------Training Model----------")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_check.parameters(), lr=0.03, weight_decay=1e-4)
scheduler = OneCycleLR(optimizer,
                       max_lr = 4.02E-02,
                       #pct_start = 5/EPOCHS,
                       div_factor = 100,
                       epochs=EPOCHS,
                       steps_per_epoch=len(train_loader),
                       verbose = False,three_phase=False)
                       #final_div_factor= 100,anneal_strategy='linear')

train_losses   = []
test_losses    = []
train_acc_list = []
test_acc_list  = []

for epoch in range(EPOCHS):
    print("EPOCH:", epoch)
    train_loss,train_acc = utils.train(model_check, device, train_loader, optimizer, epoch,criterion,scheduler)
    train_losses.append(train_loss)
    train_acc_list.append(train_acc)

    test_loss,test_acc = utils.test(model_check, device, test_loader,criterion)
    test_losses.append(test_loss)
    test_acc_list.append(test_acc)

print(f"----------Training and Testing Loss/Accuracy----------")
utils.plot_losses(train_losses,train_acc_list,test_losses,test_acc_list)

print(f"----------Misclassified Images----------")
utils.misclassified_images(10,model_check,test_loader,device)

print(f"----------Grad Cam on Images----------")
utils.grad_cam_images(model_check,test_loader,device,9,10)
