# Building and Training Conditional VAE on CIFAR10 dataset


In this repo, we build CVAE on CIFAR10 dataset.

Model

```
VAE(
  (encConv1): Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1))
  (encConv2): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))
  (encConv3): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1))
  (fc_mu): Linear(in_features=173056, out_features=512, bias=True)
  (fc_var): Linear(in_features=173056, out_features=512, bias=True)
  (deFC1): Linear(in_features=522, out_features=173056, bias=True)
  (deConv1): ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(1, 1))
  (deConv2): ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(1, 1))
  (deConv3): ConvTranspose2d(64, 3, kernel_size=(3, 3), stride=(1, 1))
)
```

Training
```
****Training****
Epoch 20: Train loss -1041.243896484375
****Training****
Epoch 21: Train loss -1047.35302734375
****Training****
Epoch 22: Train loss -1055.383544921875
****Training****
Epoch 23: Train loss -1063.37646484375
****Training****
Epoch 24: Train loss -1073.676025390625
****Training****
Epoch 25: Train loss -1081.3480224609375
****Training****
Epoch 26: Train loss -1089.858642578125
****Training****
Epoch 27: Train loss -1098.6964111328125
****Training****
Epoch 28: Train loss -1105.6414794921875
****Training****
Epoch 29: Train loss -1115.9234619140625
```
Plotting Images with incorrect labels.



