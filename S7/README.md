
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
1. Parameters: 32 K
2. Best Training Accuracy: 99.19
3. Best Test Accuracy: 98.89
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
1. Parameters: 32 k
2. Best Training Accuracy: 99.89
3. Best Test Accuracy: 99.3
#### Analysis:
1. Performance on both test and train improved.
2. Still see some overfitting between train and test accuracy.
3. Test accuracy is not consistent in performance.

#### File:
Session_7_Model2.ipynb

--------------------------------------------

### MODEL 4 - the GAP

#### Targets:
1. Add gap layer in the output layer
2. 

#### Results:
1. Parameters: 
2. Best Training Accuracy: 
3. Best Test Accuracy: 
#### Analysis:
1. 

#### File:
