# Building and Training Conditional VAE on MNIST dataset


In this repo, we build CVAE on MNIST dataset.

Model

```
VAE(
  (encConv1): Conv2d(2, 32, kernel_size=(3, 3), stride=(1, 1))
  (encConv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
  (fc_mu): Linear(in_features=36864, out_features=256, bias=True)
  (fc_var): Linear(in_features=36864, out_features=256, bias=True)
  (deFC1): Linear(in_features=266, out_features=36864, bias=True)
  (deConv1): ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=(1, 1))
  (deConv2): ConvTranspose2d(32, 1, kernel_size=(3, 3), stride=(1, 1))
)
```

Training
```
****Training****
Epoch 20: Train loss 870.9280395507812
****Training****
Epoch 21: Train loss 870.5869140625
****Training****
Epoch 22: Train loss 870.216552734375
****Training****
Epoch 23: Train loss 869.9364013671875
****Training****
Epoch 24: Train loss 869.6248168945312
****Training****
Epoch 25: Train loss 869.3941040039062
****Training****
Epoch 26: Train loss 869.1148071289062
****Training****
Epoch 27: Train loss 868.8385009765625
****Training****
Epoch 28: Train loss 868.6658935546875
****Training****
Epoch 29: Train loss 868.348388671875
```
Plotting Images with incorrect labels.

![Screenshot 2023-09-23 at 9 47 47 pm](https://github.com/santule/ERA/assets/20509836/dd367370-633d-4c04-a85b-25818d181cf1)


