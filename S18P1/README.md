# Building and Training U-NET model


In this repo, we build U-Net with 4 different strategies.

STRATEGY 1 - Max Pooling(Contracting) + Transpose Convolutions(Expanding) + Cross Entropy Loss

```
my_unet = Unet(in_channels=3, out_channels=3, StrConv=False, ConvTr=True).to(device)
train_unet(EPOCHS,my_unet,ce_loss_fn,device,train_dataloader,test_dataloader)

```
![Screenshot 2023-09-20 at 11 12 54 pm](https://github.com/santule/ERA/assets/20509836/456a33ee-138d-412a-988e-848c3b03b125)



STRATEGY 2 -  Max Pooling(Contracting) + Transpose Convolutions(Expanding) + Dice Loss

```
my_unet = Unet(in_channels=3, out_channels=3,StrConv=False,ConvTr=True).to(device)
train_unet(EPOCHS,my_unet,dice_loss_fn,device,train_dataloader,test_dataloader)

```

![Screenshot 2023-09-20 at 11 12 41 pm](https://github.com/santule/ERA/assets/20509836/a0855d89-b563-4ba4-8dbf-c0c2d1cfaee2)


STRATEGY 3 -  Strided Convolutions(Contracting) + Transpose Convolutions(Expanding) + Cross Entropy Loss

```
my_unet = Unet(in_channels=3, out_channels=3,StrConv=True,ConvTr=True).to(device)
train_unet(EPOCHS,my_unet,ce_loss_fn,device,train_dataloader,test_dataloader)

```
![Screenshot 2023-09-20 at 11 23 36 pm](https://github.com/santule/ERA/assets/20509836/b2a37cf1-acb7-4e6b-b013-3b3c737c3fa3)

STRATEGY 4 -  Strided Convolutions(Contracting) + Bilinear UpSampling(Expanding) + Dice Loss

```
my_unet = Unet(in_channels=3, out_channels=3,StrConv=True,ConvTr=False).to(device)
train_unet(EPOCHS,my_unet,dice_loss_fn,device,train_dataloader,test_dataloader)

```

