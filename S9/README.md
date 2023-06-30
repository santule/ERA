# Training CIFAR-10 with less than 200k parameters and 85% accuracy using advances convolutions.

In this repo, we train convolutional neural network on CIFAR10 dataset. We use depthwise convolutions and diluted convolutions.

### Model Architecture



### 1 - Load the dataset
CIFAR10 dataset can be loaded and augmented like so.

```sh
train_loader,test_loader = dataloader.load_dataset()
```

### 2 - Define and summarise model

```sh
model_check = model.Net().to(device)
utils.summarise_model(model_check)
```

### 3 - Train the model

```sh
train_losses = []
test_losses = []
train_acc_list = []
test_acc_list = []

optimizer = optim.SGD(model_check.parameters(), lr=0.01, momentum=0.9)
#scheduler = OneCycleLR(optimizer, max_lr=0.5, epochs=60, steps_per_epoch=len(train_loader),verbose = True)

EPOCHS = 80
for epoch in range(EPOCHS):
    print("EPOCH:", epoch)
    train_loss,train_acc = utils.train(model_check, device, train_loader, optimizer, epoch)
    train_losses.append(train_loss)
    train_acc_list.append(train_acc)
    #scheduler.step()


    test_loss,test_acc = utils.test(model_check, device, test_loader)
    test_losses.append(test_loss)
    test_acc_list.append(test_acc)
```

### 4 - Plot graph
```sh
utils.plot_losses(train_losses,train_acc_list,test_losses,test_acc_list)
```
