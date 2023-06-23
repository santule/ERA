# Different normalisation for CIFAR-10 dataset

In this repo, we try different normalisation - batch.layer and group normalisation on CIFAR-10 dataset with less than 50k parameters with min test accuracy of 70%. We also add skip connections.

### Model Architecture
[![A.png](https://i.postimg.cc/Gt9CWcg4/A.png)](https://postimg.cc/LgdwfMpS)


### 1 - Batch Normalisation

##### 1.1 - Train Model
```sh
model_gn =  model.Net(norm="BN").to(device)
for epoch in range(EPOCHS):
    print("EPOCH:", epoch)
    train_loss,train_acc = model.train(model_gn, device, train_loader, optimizer, epoch)
    test_loss,test_acc = model.test_accuracy = model.test(model_gn, device, test_loader)
```


##### 1.2 - Training/Testing Loss/Accuracy graph.
[![bn-graph.png](https://i.postimg.cc/fb6qzMb3/bn-graph.png)](https://postimg.cc/WDwwSLgj)


##### 1.3 - 10 misclassified examples
[![bn-mis.png](https://i.postimg.cc/1XSCkPmp/bn-mis.png)](https://postimg.cc/34LFGsDR)


### 2 - Layer Normalisation

##### 2.1 - Train Model
```sh
model_gn =  model.Net(norm="LN").to(device)
for epoch in range(EPOCHS):
    print("EPOCH:", epoch)
    train_loss,train_acc = model.train(model_gn, device, train_loader, optimizer, epoch)
    test_loss,test_acc = model.test_accuracy = model.test(model_gn, device, test_loader)
```

##### 2.2 - Training/Testing Loss/Accuracy graph.
[![ln-graph.png](https://i.postimg.cc/1zCnwzFt/ln-graph.png)](https://postimg.cc/yJR8zsv4)

##### 2.3 - 10 misclassified examples
[![ln-mis.png](https://i.postimg.cc/P5C4FNWg/ln-mis.png)](https://postimg.cc/DSKLSvY6)

### 3 - Group Normalisation

##### 3.1 - Train Model
```sh
model_gn =  model.Net(norm="GN").to(device)
for epoch in range(EPOCHS):
    print("EPOCH:", epoch)
    train_loss,train_acc = model.train(model_gn, device, train_loader, optimizer, epoch)
    test_loss,test_acc = model.test_accuracy = model.test(model_gn, device, test_loader)
```

##### 3.2 - Training/Testing Loss/Accuracy graph.
[![gn-graph.png](https://i.postimg.cc/PJzdb674/gn-graph.png)](https://postimg.cc/64qstht2)

##### 3.3 - 10 misclassified examples
[![gn-mis.png](https://i.postimg.cc/fygdxjrJ/gn-mis.png)](https://postimg.cc/zHK3NW5r)
