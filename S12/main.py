#import utils as utils
#from models import custom_resnet

import ERA.S12.utils as utils
from ERA.S12.models import custom_resnet

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch_lr_finder import LRFinder
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
import torchvision
import torch.nn.functional as F
from torchsummary import summary
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image,scale_cam_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib.pyplot as plt
import math
from PIL import Image
from pl_bolts.datamodules import CIFAR10DataModule
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import Accuracy
import seaborn as sn
import pandas as pd


# parameters
SEED = 1
EPOCHS = 20
BATCH_SIZE = 512
transparency = 0.60 # gradcam
torch.manual_seed(SEED)
cuda    = torch.cuda.is_available()
if cuda:
  torch.cuda.manual_seed(SEED)
device  = torch.device("cuda" if cuda else "cpu")
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


print(f"----------Display Sample Images -------------")

# load original images
train = datasets.CIFAR10('./data',train=True,download=True,transform=transforms.ToTensor())
utils.display_sample_images(12,train,classes)

print(f"----------Display Sample Augmented Images -------------")
utils.display_augmented_images(classes,train)

print(f"----------Load and Transform Images----------")
train_loader,test_loader = utils.load_dataset()

print(f"----------Find Max LR----------")
model_max_lr = custom_resnet.Net().to(device)
found_max_lr = utils.find_max_lr(model_max_lr,train_loader)
print(f"Max LR {found_max_lr}")

print(f"----------Build Model and Train using pytorch Lightning ----------")
class LitResnet(LightningModule):
    def __init__(self, num_classes=10, lr=0.05, max_lr=5.38E-02):
        super().__init__()

        self.save_hyperparameters()
        self.model = custom_resnet.Net()
        self.criterion = nn.CrossEntropyLoss()
        self.BATCH_SIZE = 512
        self.torchmetrics_accuracy = Accuracy(task="multiclass", num_classes= self.hparams.num_classes)

    def forward(self, x):
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        acc  = self.torchmetrics_accuracy(y_pred, y)

        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss


    def evaluate(self, batch, stage=None):
        x, y = batch
        y_test_pred = self(x)
        loss = self.criterion(y_test_pred, y)
        acc  = self.torchmetrics_accuracy(y_test_pred, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        scheduler = OneCycleLR(
                optimizer,
                max_lr= self.hparams.max_lr, #5.38E-02, #self.hparams.lr,
                pct_start = 5/self.trainer.max_epochs,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=len(train_loader),
                div_factor=100,verbose=False,
                three_phase=False
            )
        return ([optimizer],[scheduler])

model   = LitResnet(num_classes=10,lr=0.03,max_lr=found_max_lr)
trainer = Trainer(log_every_n_steps=1,
                  auto_lr_find=True,
    enable_model_summary=True,
    max_epochs=20,
    precision=16,
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    logger=CSVLogger(save_dir="logs/"), # TensorBoardLogger("logs/")
    default_root_dir="model/save/",
    callbacks=[LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=10)],)

trainer.fit(model, train_loader,test_loader)
trainer.test(model, test_loader)
trainer.save_checkpoint("cifar10_customresnet_20_epoch.ckpt")

print(f"----------Training / Tesing loss and accuracy curves ----------")
metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
del metrics["step"]
metrics.set_index("epoch", inplace=True)
#display(metrics.dropna(axis=1, how="all").head())
sn.relplot(data=metrics, kind="line")
plt.show()

# check the model's test accuracy by loading from save_checkpoint
print(f"----------Model Performance after loading from saved checkpoint----------")
inference_model = LitResnet.load_from_checkpoint("cifar10_customresnet_20_epoch.ckpt")
trainer.test(inference_model, test_loader)

print(f"----------Misclassified Images----------")
inference_model.to(device)
misclassified_examples,misclassified_labels,correct_labels = utils.get_misclassified_examples(inference_model,test_loader,device)
utils.display_misclassified_images(12,misclassified_examples,misclassified_labels,correct_labels,classes)
#
# print(f"----------Grad Cam on Images----------")
# target_layers = [model_check.layer4[-1]]
# utils.display_grad_cam_images(12,model_check,target_layers,test_loader,device,misclassified_examples,transparency,classes,correct_labels,misclassified_labels)
