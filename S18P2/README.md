# Building and Training Conditional VAE on MNIST dataset trained for generating new images using image + incorrect label


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
  (chConv1): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1))
  (chout): Linear(in_features=43264, out_features=10, bias=True)
  (bce_loss): BCELoss()
)
```

Training
```
****Training****
****Validation****
Epoch 0: Validation loss 1.3629941940307617
****Training****
Epoch 0: Train loss 2.1524834632873535
****Validation****
Epoch 1: Validation loss 1.3311786651611328
****Training****
Epoch 1: Train loss 1.3482739925384521
****Validation****
Epoch 2: Validation loss 1.2945215702056885
****Training****
Epoch 2: Train loss 1.2846781015396118
****Validation****
Epoch 3: Validation loss 1.264923334121704
****Training****
Epoch 3: Train loss 1.2440893650054932
****Validation****
Epoch 4: Validation loss 1.2440366744995117
****Training****
Epoch 4: Train loss 1.2207486629486084
****Validation****
Epoch 5: Validation loss 1.1783515214920044
****Training****
Epoch 5: Train loss 1.1661055088043213
****Validation****
Epoch 6: Validation loss 1.148390293121338
****Training****
Epoch 6: Train loss 1.1287875175476074
****Validation****
Epoch 7: Validation loss 1.1226953268051147
****Training****
Epoch 7: Train loss 1.0994470119476318
****Validation****
Epoch 8: Validation loss 1.1502413749694824
****Training****
Epoch 8: Train loss 1.0763802528381348
****Validation****
Epoch 9: Validation loss 1.1147665977478027
****Training****
Epoch 9: Train loss 1.0576246976852417
****Validation****
Epoch 10: Validation loss 1.1046535968780518
****Training****
Epoch 10: Train loss 1.0429153442382812
****Validation****
Epoch 11: Validation loss 1.1034306287765503
****Training****
Epoch 11: Train loss 1.0285704135894775
****Validation****
Epoch 12: Validation loss 1.0828297138214111
****Training****
Epoch 12: Train loss 1.0215729475021362
****Validation****
Epoch 13: Validation loss 1.078284502029419
****Training****
Epoch 13: Train loss 1.0082311630249023
****Validation****
Epoch 14: Validation loss 1.0806478261947632
****Training****
Epoch 14: Train loss 1.0038723945617676
****Validation****
Epoch 15: Validation loss 1.0692613124847412
****Training****
Epoch 15: Train loss 0.9944864511489868
****Validation****
Epoch 16: Validation loss 1.0644495487213135
****Training****
Epoch 16: Train loss 0.9856837391853333
****Validation****
Epoch 17: Validation loss 1.070383906364441
****Training****
Epoch 17: Train loss 0.9819890856742859
****Validation****
Epoch 18: Validation loss 1.0578793287277222
****Training****
Epoch 18: Train loss 0.977859377861023
****Validation****
Epoch 19: Validation loss 1.0653613805770874
****Training****
Epoch 19: Train loss 0.9749959707260132
```
Plotting Images with incorrect labels.

<img width="648" alt="Screenshot 2023-09-25 at 11 42 36 pm" src="https://github.com/santule/ERA/assets/20509836/21ff59c0-f9a9-4c58-923e-dc883e371e85">

