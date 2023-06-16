
## Building an effecient DNN - from 6M parameters to 8K parameters


### MODEL 1 - the basic

#### Targets:
1. Get the set-up right
2. Set Transforms
3. Set Data Loader
4. Set Basic Working Code
5. Set Basic Training  & Test Loop

#### Results:
1. Parameters: 6.3M
2. Best Training Accuracy: 99.9
3. Best Test Accuracy: 99.3
#### Analysis:
1. very heavy model for a simple MNIST dataset.
2. From epoch 5 to 15, the model is there is consistent gap between training and testing accuracy.

#### File:
Session_7_Model1.ipynb

--------------------------------------------

### MODEL 2 - the skeleton

#### Targets:
1. Basic skeleton model with convolutional and transition block
2. Making the model lighter

#### Results:
1. Parameters: 9 K
2. Best Training Accuracy: 99.2
3. Best Test Accuracy: 98.9
#### Analysis:
1. Both training and testing performance has gone down as the model is very small.
2. The gap between training and testing accuracy has reduced but started to get wider in the last 3-4 epochs, meaning the model is now overfitting the training data.
3. The model cannot get any better with this.

#### File:
Session_7_Model2.ipynb

--------------------------------------------

### MODEL 3 - the progressive

#### Targets:
1. Include batch normalisation
2. Add regularisation - dropout

#### Results:
1. Parameters: 9 k
2. Best Training Accuracy: 99.43
3. Best Test Accuracy: 99.0
#### Analysis:
1. Performance on both test and train improved.
2. Still see some overfitting between train and test accuracy.
3. Test accuracy is not consistent in performance.

#### File:
Session_7_Model3.ipynb

--------------------------------------------

### MODEL 4 - add the GAP

#### Targets:
1. Add gap layer in the output layer 

#### Results:
1. Parameters: 6.5K
2. Best Training Accuracy: 98.8
3. Best Test Accuracy: 98.5
#### Analysis:
1. Difference between training and test accuracy has decreased.
2. Overall model performance has dropped.

#### File:
Session_7_Model4.ipynb

--------------------------------------------

### MODEL 5 - dropout after each layer, add layer after GAP.

#### Targets:
1. Add dropout after each convolutional layer.
2. Add another convolutional layer after GAP.
3. Get rid of the second transition block, and increase model capacity.

#### Results:
1. Parameters: 7.5k
2. Best Training Accuracy: 
3. Best Test Accuracy: 
#### Analysis:
1. Model reaches good accuracy of 99.2 and it is stable in the last few epochs.

#### File:
Session_7_Model5.ipynb

--------------------------------------------


### MODEL 6 - Add image augmentation and StepLR
#### Targets:
1. Add image rotation, jitter and affine.
2. Add StepLR to adjust learning rate after half epochs of total epochs are done.
3. Include bias as False.

#### Results:
1. Parameters: 7416
2. Best Training Accuracy: 
3. Best Test Accuracy: 99.4
#### Analysis:
1. After epoch 8, there is bump in the accuracy. This concides with the LR adjusted through stepLR.
2. The model shows consistent performance of 99.4 in last 5 epochs.

#### File:
Session_7_Model6.ipynb

Model plots:
[![Screen-Shot-2023-06-16-at-11-10-08-pm.png](https://i.postimg.cc/j2hMcZpb/Screen-Shot-2023-06-16-at-11-10-08-pm.png)](https://postimg.cc/JHG34c76)
