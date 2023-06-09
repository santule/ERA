# Session 6 Assignment
## BACKPROPOGATION

Understanding backpropogation in neural networks.
[![image.png](https://i.postimg.cc/Vk6mKp69/Screen-Shot-2023-06-10-at-1-34-28-am.png)


## Forward Pass

Step 1 - Perform forward pass through the model:

```python
train_loader, test_loader = utils.load_data(batch_size = 512)
```

Visualise random images in the train data:

```python
utils.visualise_data(12,train_loader)
```



