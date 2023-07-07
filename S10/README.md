# Training a custom resnet architecture for CIFAR-10 with 90% validation accuracy in 24 epochs.

In this repo, we train custom resnet convolutional neural network on CIFAR10 dataset. We use one cycle LR and cutouts.

### Model Architecture



### 1 - Load the dataset
CIFAR10 dataset can be loaded and augmented like so.

```sh
train_loader,test_loader = dataloader.load_dataset()
```

### 2 - Define and summarise model

```sh
model_check = custom_resnet.Net().to(device)
utils.summarise_model(model_check)
```

### 3 - Find max LR

```sh
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_check.parameters(), lr=0.03, weight_decay=1e-4)
lr_finder = LRFinder(model_check, optimizer, criterion, device="cuda")
lr_finder.range_test(train_loader, end_lr=10, num_iter=200,step_mode='exp')
lr_finder.plot()  # to inspect the loss-learning rate graph
lr_finder.reset()
```

### 4 - Train the model using one-cycle LR

```sh

EPOCHS = 24
scheduler = OneCycleLR(optimizer,
                       max_lr = 5.38E-02,
                       pct_start = 5/EPOCHS,
                       div_factor = 100,
                       epochs=EPOCHS,
                       steps_per_epoch=len(train_loader),
                       verbose = False,three_phase=False)
                       #final_div_factor= 100,anneal_strategy='linear')


for epoch in range(EPOCHS):
    print("EPOCH:", epoch)
    utils.train(model_check, device, train_loader, optimizer, epoch,criterion,scheduler)
    utils.test(model_check, device, test_loader,criterion)
```
