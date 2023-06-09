# Session 5 Assignment
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
            Conv2d-1           [-1, 32, 26, 26]             320
            Conv2d-2           [-1, 64, 24, 24]          18,496
            Conv2d-3          [-1, 128, 10, 10]          73,856
            Conv2d-4            [-1, 256, 8, 8]         295,168
            Linear-5                   [-1, 50]         204,850
            Linear-6                   [-1, 10]             510
================================================================
Total params: 593,200
Trainable params: 593,200
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.67
Params size (MB): 2.26
Estimated Total Size (MB): 2.94
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

