import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchsummary import summary
from tqdm.autonotebook import tqdm
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image,scale_cam_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import cv2
from PIL import Image


train_losses = []
test_losses = []
train_acc = []
test_acc = []
means = [0.4914, 0.4822, 0.4465]
stds  = [0.2470, 0.2435, 0.2616]


# Train Phase transformations
train_transforms = A.Compose(
    [
        A.Normalize(mean=means, std=stds, always_apply=True),
        A.PadIfNeeded(min_height=36, min_width=36, always_apply=True),
        A.RandomCrop(height=32, width=32, always_apply=True),
        A.HorizontalFlip(),
        A.CoarseDropout(max_holes=1, max_height=8, max_width=8, min_holes=1, min_height=8, min_width=8, fill_value=means),
        ToTensorV2(),
    ]
)

test_transforms = A.Compose(
    [
        A.Normalize(mean=means, std=stds, always_apply=True),
        ToTensorV2(),
    ]
)

# dataset for CIFAR10 dataset
class CIFAR10Dataset(datasets.CIFAR10):

 def __init__(self,root="./data",train=True,download=True,transform=None):
   super().__init__(root=root,train=train,download=download,transform=transform)

 def __getitem__(self,index):
   image, label = self.data[index], self.targets[index]

   if self.transform is not None:
     transformed = self.transform(image=image)
     image = transformed["image"]
   return image,label


def load_dataset():
    SEED = 1
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)

    # For reproducibility
    torch.manual_seed(SEED)

    if cuda:
     torch.cuda.manual_seed(SEED)

    # dataloader arguments
    dataloader_args = dict(shuffle=True, batch_size=512, num_workers=2, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

    # train dataloader
    train = CIFAR10Dataset(transform = train_transforms)
    train_loader = torch.utils.data.DataLoader(train, **dataloader_args)

    # test dataloader
    test = CIFAR10Dataset(transform = test_transforms,train=False)
    test_loader = torch.utils.data.DataLoader(test, **dataloader_args)

    return train_loader,test_loader

def train(model, device, train_loader, optimizer, epoch,criterion,scheduler):

  model.train()
  pbar = tqdm(train_loader)
  correct    = 0
  processed  = 0
  train_loss = 0

  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)
    # Init
    optimizer.zero_grad()

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = criterion(y_pred, target)
    train_loss += loss.item()

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update pbar-tqdm
    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    scheduler.step()
    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100 * correct/processed:0.2f}')

  train_loss = train_loss / processed
  train_acc = 100 * correct / processed

  print('\nTraining set set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        train_loss, correct, len(train_loader.dataset),train_acc))

  return train_loss, train_acc



def test(model, device, test_loader,criterion):
    model.eval()
    test_loss = 0
    correct   = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = 100 * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct, len(test_loader.dataset),test_acc))

    return test_loss, test_acc


def summarise_model(m):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = m.to(device)
    summary(model, input_size=(3,32,32))

def plot_losses(train_losses,train_acc,test_losses,test_acc):
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def misclassified_images(num_misclassified,model_bn,test_loader,device):

    misclassified_examples = []
    misclassified_labels = []
    correct_labels = []

    model_bn.eval()
    with torch.no_grad():
      for data, target in test_loader:
          data, target = data.to(device), target.to(device)
          output = model_bn(data)
          pred = output.argmax(dim=1, keepdim=True)

          ids_mask = ((pred == target.view_as(pred)) ==False).view(-1)
          misclassified_examples.append(data[ids_mask].squeeze().cpu().numpy())
          misclassified_labels.append(target[ids_mask].squeeze().cpu().numpy())
          correct_labels.append(pred[ids_mask].squeeze().cpu().numpy())

          if len(misclassified_examples[0]) >= num_misclassified:
            break

    fig = plt.figure(figsize=(20,8))
    for idx in np.arange(num_misclassified):
      ax = fig.add_subplot(2,5,idx + 1,xticks=[],yticks=[])
      img = misclassified_examples[0][idx]
      img = img/2 + 0.5
      img = np.clip(img,0,1)
      plt.imshow(img.T)
      ax.set_title(f"Correct/Predicted: {misclassified_labels[0][idx]} / {correct_labels[0][idx]}")

    plt.show()

def grad_cam_images(model,test_loader,class_to_see,max_images):

  target_layer = [model.layer3[-1]]
  input_tensor = test_loader
  test_images = [x[0] for x in test_loader.dataset]
  test_images = torch.stack(test_images[:max_images]).to(device) # test images to apply grad cam on
  model.eval() # model in eval
  cam = GradCAM(model= model, target_layers=target_layer, use_cuda=device) # cam
  target_class = [ClassifierOutputTarget(class_to_see)] # which class to see
  grayscale_cam = cam(input_tensor=test_images, targets=target_class) # grayscale activation

  class_cam = []
  for i in range(grayscale_cam.shape[0]):
    grayscale_cam_img = grayscale_cam[i,:]
    rgb_test_img = test_images[i,: ].cpu().numpy()
    rgb_test_img = rgb_test_img.reshape(32,32,3)
    rgb_test_img = np.float32(rgb_test_img) / 255
    cam_image = show_cam_on_image(rgb_test_img, grayscale_cam_img, use_rgb = True)
    cam = np.uint8(255 * grayscale_cam_img)
    cam = cv2.merge([cam, cam, cam])
    class_cam.append(cam_image)

  images = np.hstack(class_cam)
  return images
