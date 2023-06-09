# Session 6 Assignment
## MNIST CLASSIFICATION USING CONVOLUTIONAL NEURAL NETWORK

Training convolutional neural network on MNIST data.
[![image.png](https://i.postimg.cc/2SqzfyDN/image.png)](https://postimg.cc/JsLwN10p)


## Using functions in Jupyter notebook

Load data:

```python
train_loader, test_loader = utils.load_data(batch_size = 512)
```

Visualise random images in the train data:

```python
utils.visualise_data(12,train_loader)
```

Load the model and print summary:

```python
device = 'cuda' if cuda else 'cpu'
mymodel = model.Net().to(device)
utils.summarise_model(mymodel)
```

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 28, 28]             160
              ReLU-2           [-1, 16, 28, 28]               0
           Dropout-3           [-1, 16, 28, 28]               0
       BatchNorm2d-4           [-1, 16, 28, 28]              32
         MaxPool2d-5           [-1, 16, 14, 14]               0
            Conv2d-6           [-1, 32, 14, 14]           4,640
              ReLU-7           [-1, 32, 14, 14]               0
           Dropout-8           [-1, 32, 14, 14]               0
       BatchNorm2d-9           [-1, 32, 14, 14]              64
        MaxPool2d-10             [-1, 32, 7, 7]               0
           Conv2d-11             [-1, 32, 7, 7]           9,248
             ReLU-12             [-1, 32, 7, 7]               0
          Dropout-13             [-1, 32, 7, 7]               0
      BatchNorm2d-14             [-1, 32, 7, 7]              64
           Conv2d-15             [-1, 10, 7, 7]           2,890
             ReLU-16             [-1, 10, 7, 7]               0
          Dropout-17             [-1, 10, 7, 7]               0
      BatchNorm2d-18             [-1, 10, 7, 7]              20
        MaxPool2d-19             [-1, 10, 3, 3]               0
        AvgPool2d-20             [-1, 10, 1, 1]               0
================================================================
Total params: 17,118
Trainable params: 17,118
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.67
Params size (MB): 0.07
Estimated Total Size (MB): 0.74
----------------------------------------------------------------

```

Train the model
```python
for epoch in range(1, num_epochs+1):
    print(f'Epoch {epoch}')
    train_loss,train_acc = model.train(mymodel, device, train_loader, optimizer, criterion)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
<<<<<<< HEAD

=======
    
>>>>>>> b5961f5473b07ba22b72b2bf1bc21e1fb99569b0
    test_loss,test_acc = model.test(mymodel, device, test_loader, criterion)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    scheduler.step()
```
Model training and testing performance

[![image.png](https://i.postimg.cc/VLhJt3s1/image.png)](https://postimg.cc/MvyZ23jr)

