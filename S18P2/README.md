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
Epoch 0: Train loss 743.94482421875
****Training****
Epoch 1: Train loss 537.4459228515625
****Training****
Epoch 2: Train loss 409.37493896484375
****Training****
Epoch 3: Train loss 290.4701232910156
****Training****
Epoch 4: Train loss 183.79696655273438
****Training****
Epoch 5: Train loss 93.75096893310547
****Training****
Epoch 6: Train loss 25.283349990844727
****Training****
Epoch 7: Train loss -18.43797492980957
****Training****
Epoch 8: Train loss -38.52235794067383
****Training****
Epoch 9: Train loss -44.07147979736328
```
Plotting Images with incorrect labels.

<img width="655" alt="Screenshot 2023-09-23 at 12 07 52 am" src="https://github.com/santule/ERA/assets/20509836/3a8d3b21-cfee-4097-950b-61e5070d3543">

