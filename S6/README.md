# Session 6 Assignment
## Understanding backpropogation in neural networks


[![Screen-Shot-2023-06-10-at-2-08-27-am.png](https://i.postimg.cc/BZHCT4bw/Screen-Shot-2023-06-10-at-2-08-27-am.png)](https://postimg.cc/hh4xgF99)


Here we have a simple neural network with 1 hidden layer, 1 output layer and the input layer. In this network we have 2 input features and 2 output predictions. All explanations below is tailored to this network.

##### Step 1 - Randomly initialise weights in neural network.
Before we start training our neural network, we first randomly initialise the weights of our network.
##### Step 2 - Perform a forward pass through the network using the weights and the input data.
With the weights and input features, we can now calculate the hidden layer output and output layer predictions.
##### Step 3 - Calculate the loss between ground truth prediction and output from the network.
The output predictions made by our network can be compared to the ground truth and the loss can be calculated. The higher the loss, the worse our network performs. We need to reduce the loss in the next epoch. For this we need to adjust the weights in the network in the direction of the loss.
##### Step 4 - Calculate the gradient of the loss with respect to each of the weights.
The graident of the loss with respect to the weights can be calculated using the partial derivatives.
##### Step 5 - Adjust the weight of the network based on the gradients.
The weights are now adjusted using weights = weights - learning_rate * partial derivative of the loss wrt to the weight.
##### Step 6 - Rerun step 1 to 6 again till number of epochs are finished.
We repeat the steps from 1 to 6 till number of epochs are finished.


Depending on the learning rate, the loss function graph can look very different.

Learning rate = 0.1
[![Screen-Shot-2023-06-10-at-1-41-11-am.png](https://i.postimg.cc/tTDPW74t/Screen-Shot-2023-06-10-at-1-41-11-am.png)](https://postimg.cc/gXXnPzRn)

Learning rate = 2
[![Screen-Shot-2023-06-10-at-1-43-39-am.png](https://i.postimg.cc/YCgW8dSS/Screen-Shot-2023-06-10-at-1-43-39-am.png)](https://postimg.cc/cvxCLMwq)


