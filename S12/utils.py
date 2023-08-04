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
import math

train_losses = []
test_losses  = []
train_acc    = []
test_acc     = []

#### Loading Datasets
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


# load the dataset in the dataloader
def load_dataset():
    cuda    = torch.cuda.is_available()
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

def plot_accuracy_losses(train_losses,train_acc,test_losses,test_acc):
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
    plt.show()

# get learning rate
def get_lr(optimizer):
  for param_group in optimizer.param_groups:
        return param_group['lr']

# functions to show an sample images
def display_sample_images(display_samples,train,classes):
    batch_data  = []
    batch_label = []

    for idx,train_sample in enumerate(train):
        if idx > display_samples:
            break
        batch_data.append(train_sample[0])
        batch_label.append(train_sample[1])
    batch_data = torch.stack(batch_data,dim=0).numpy()


    fig = plt.figure()
    x_width = 4
    y_width = math.ceil(display_samples / x_width)

    for i in range(display_samples):
      plt.subplot(y_width, x_width, i + 1)
      plt.tight_layout()
      plt.imshow(np.transpose(batch_data[i],(1,2,0)))
      plt.title(classes[batch_label[i]])
      plt.xticks([])
      plt.yticks([])


# display augmented images
def display_augmented_images(classes,train):

    augmentations_dict = {
    'randomcrop': A.RandomCrop(height=32, width=32, always_apply=True),
    'horizontalflip': A.HorizontalFlip(p=0.5),
    'cutout': A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=8, min_width=8, fill_value=means),
    'normalize': A.Normalize(mean=means, std=stds, always_apply=True),
    'standarize': ToTensorV2(),
    }

    fig = plt.figure()
    x_width = 4
    y_width = math.ceil(len(augmentations_dict) / x_width) # no of rows

    # sample image for albumentation transform
    aug_train = CIFAR10Dataset('./data', train=True, download=True)
    sample_img, sample_lbl = aug_train[np.random.randint(low=0,high=len(train))]

    print(f'IMAGE FOR CLASS -- {classes[sample_lbl]}')

    for i, (augment_key,augment_cmd) in enumerate(augmentations_dict.items()):
      plt.subplot(y_width,x_width,i + 1)
      plt.tight_layout()
      augmented_img = augment_cmd(image = sample_img)['image']
      if augment_key == 'standarize':
        augmented_img = np.transpose(augmented_img.numpy(),(1,2,0))
      plt.imshow(augmented_img)
      plt.title(augment_key)
      plt.xticks([])
      plt.yticks([])



def display_misclassified_images(number_misclassified_display,misclassified_examples,misclassified_labels,correct_labels,classes):

    inv_normalize = transforms.Normalize(
        mean=[-0.50/0.23, -0.50/0.23, -0.50/0.23],
        std=[1/0.23, 1/0.23, 1/0.23]
    )

    fig = plt.figure()
    x_width = 4
    y_width = math.ceil(number_misclassified_display / x_width)

    for i in range(number_misclassified_display):
      plt.subplot(y_width, x_width, i + 1)
      plt.tight_layout()
      # inverse normalisation
      img = inv_normalize(misclassified_examples[i])
      plt.imshow(np.transpose(img.cpu().numpy(),(1,2,0)))
      plt.title(f"Correct: {classes[correct_labels[i]]} \n Predicted: {classes[misclassified_labels[i]]}")
      plt.xticks([])
      plt.yticks([])
    plt.show()


# get misclassified examples
def get_misclassified_examples(model_bn,test_loader,device):
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
          misclassified_examples.append(data[ids_mask].squeeze())
          correct_labels.append(target[ids_mask].squeeze())
          misclassified_labels.append(pred[ids_mask].squeeze())


    # stack them all together
    misclassified_examples = torch.cat(misclassified_examples,dim=0)
    misclassified_labels = torch.cat(misclassified_labels,dim=0)
    correct_labels = torch.cat(correct_labels,dim=0)

    return misclassified_examples,misclassified_labels,correct_labels


# display grad cam images
def display_grad_cam_images(total_gradcam_samples,model,target_layers,test_loader,device,misclassified_examples,transparency,classes,correct_labels,misclassified_labels):

    inv_normalize = transforms.Normalize(
        mean=[-0.50/0.23, -0.50/0.23, -0.50/0.23],
        std=[1/0.23, 1/0.23, 1/0.23]
    )

    fig = plt.figure(figsize=(10, 10))
    x_width = 3
    y_width = math.ceil(total_gradcam_samples / x_width)

    cam = GradCAM(model=model,target_layers=target_layers,use_cuda=True)

    for i in range(total_gradcam_samples):
        plt.subplot(y_width,x_width, i + 1)
        plt.tight_layout()
        input_tensor = misclassified_examples[i].unsqueeze(0) #1,3,32,32

        grayscale_cam = cam(input_tensor = input_tensor,targets=None) #1,32,32
        grayscale_cam = grayscale_cam[0,:] # 32,32

        img = input_tensor.cpu().squeeze(0) # 3,32,32
        img = inv_normalize(img) # 3,32,32
        rgb_img = np.transpose(img.numpy(),(2,1,0)) # 32,32,3

        visualisation = show_cam_on_image(rgb_img,grayscale_cam,use_rgb=True,image_weight=transparency)
        plt.imshow(visualisation)
        plt.title(f"Correct: {classes[correct_labels[i]]} \n Predicted: {classes[misclassified_labels[i]]}")
        plt.xticks([])
        plt.yticks([])
    plt.show()
