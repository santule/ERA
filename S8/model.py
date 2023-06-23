from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

def norm_layer(norm_type, dimensions):
    if norm_type == "BN":
        return(nn.BatchNorm2d(dimensions[0]))
    elif norm_type == "LN":
        return(nn.GroupNorm(1, dimensions[0]))
    elif norm_type == "GN":
        return nn.GroupNorm(dimensions[0]//2, dimensions[0])
    else:
        raise ValueError('Options are BN / LN / GN')


class Net(nn.Module):
    def __init__(self,norm="BN"):
        dropout_val = 0.10
        super(Net, self).__init__()
        self.norm = norm

        # convolution block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, 3,bias=False,padding=1),
            nn.ReLU(),
            norm_layer(self.norm, [8, 32, 32]),
            nn.Dropout(dropout_val)
        ) # 8,32,32

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3,bias=False,padding=1),
            nn.ReLU(),
            norm_layer(self.norm, [16, 32, 32]),
            nn.Dropout(dropout_val)
        ) # 16,32,32

        # transition block 1
        self.trans_block1 = nn.Sequential(
            nn.Conv2d(16, 8, 1,bias=False), # 8,32,32 - reduce channels
            nn.MaxPool2d(2, 2),  # 8,16,16 - reduce output
        ) # 8,16,16


        # convolution block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 16, 3,bias=False,padding=1),
            nn.ReLU(),
            norm_layer(self.norm, [16, 16, 16]),
            nn.Dropout(dropout_val)
        ) # 16,16,16

        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 32, 3,bias=False,padding=1),
            nn.ReLU(),
            norm_layer(self.norm, [32, 16, 16]),
            nn.Dropout(dropout_val)
        ) # 32,16,16

        # convolution block 5
        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3,bias=False,padding=1),
            norm_layer(self.norm, [32, 16, 16]),
            nn.Dropout(dropout_val)
        ) # 32,16,16

        # transition block 2
        self.trans_block2 = nn.Sequential(
            nn.Conv2d(32, 16, 1,bias=False), # 32,16,16 - reduce channels
            nn.MaxPool2d(2, 2),  # 16,8,8 - reduce output
        ) # 16,8,8

        self.conv6 = nn.Sequential(
            nn.Conv2d(16, 32, 3,bias=False,padding=1),
            nn.ReLU(),
            norm_layer(self.norm, [32, 8, 8]),
            nn.Dropout(dropout_val)
        ) # 32,8,8

        self.conv7 = nn.Sequential(
            nn.Conv2d(32, 32, 3,bias=False,padding=1),
            nn.ReLU(),
            norm_layer(self.norm, [32, 8, 8]),
            nn.Dropout(dropout_val),
        ) # 32,8,8

        self.conv8 = nn.Sequential(
            nn.Conv2d(32, 32, 3,bias=False,padding=1),
            norm_layer(self.norm, [32, 8, 8]),
            nn.Dropout(dropout_val),
        ) # 32,8,8

        # final block
        self.final_block = nn.Sequential(
            nn.AvgPool2d(kernel_size=8), # 16
            nn.Conv2d(32, 10, 1,bias=False)  # 10
        )

        self.skip_conn1 = nn.Sequential(
            nn.Conv2d(8, 32, 1,bias=False)
        )

        self.skip_conn2 = nn.Sequential(
            nn.Conv2d(16, 32, 1,bias=False)
        )

    def forward(self, x):
        x = self.conv1(x) # 8,32,32
        x = self.conv2(x) # 16,32,32
        x = self.trans_block1(x) # 8,16,16
        x_clone1 = self.skip_conn1(x.clone()) # 32,16,16


        x = self.conv3(x) # 16,16,16
        x = self.conv4(x) # 16,16,16
        x = self.conv5(x) # 32,16,16 -- NO RELU HERE
        x = x + x_clone1  # skip connection 1 ,32,16,16
        x = F.relu(x)
        x = self.trans_block2(x) #16,8,8
        x_clone2 = self.skip_conn2(x.clone()) #32,8,8

        x = self.conv6(x) # 32,8,8
        x = self.conv7(x) # 32,8,8
        x = self.conv8(x) # 32,8,8 --- NO RELU HERE
        x = x + x_clone2 # skip connection 2 , 32,8,8
        x = F.relu(x)

        x = self.final_block(x)
        x = x.view(-1,10)
        return F.log_softmax(x, dim=-1)

def train(model, device, train_loader, optimizer, epoch):
        model.train()
        pbar = tqdm(train_loader)

        train_loss = 0
        correct = 0
        processed = 0
        for batch_idx, (data, target) in enumerate(pbar):
            # get samples
            data, target = data.to(device), target.to(device)

            # Init
            optimizer.zero_grad()
            # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
            # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

            # Predict
            y_pred = model(data)

            # Calculate loss
            loss = F.nll_loss(y_pred, target)
            #train_losses.append(loss)
            train_loss += loss.item()

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Update pbar-tqdm

            pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)

            #pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
            #train_acc.append(100*correct/processed)

        train_loss = train_loss/len(train_loader)
        train_acc = 100 * correct / processed

        print('Training set set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        train_loss, correct, len(train_loader.dataset),
        train_acc))

        return train_loss, train_acc


def test(model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        test_acc = 100. * correct / len(test_loader.dataset)

        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        test_acc))

        return test_loss, correct / len(test_loader.dataset)
