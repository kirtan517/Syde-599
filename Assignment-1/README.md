## SYDE 599 Deep Learning Autodifferentiation Assignment
Completed by:
Clayton Haight
Danlei Geng
Kirtan Kanani
Liam Calder
Samuel Diaz

## How to Run the Code

The 'main.py' file contains the code related to the creation of the test network using our autodifferentiation package and all the resulting values from the test. Simply run the 'main.py' file to run the tests. You can set the number of epochs, batch size and learning rate at the beginning of the file.

## Test Code

Once the 'main.py' file is run the values for number of epochs, batch size, and learning rate will be assigned. These can be modified by the user and will set the conditions for the train() function.

The dataloader() function will read the training data from the pickle file provided to us with the assignment instructions, and it will create the training set ('trainset') and parameters ('params') for the network.

The train() function is responsible for performing all the network training, and displaying results.

Finally, the plot(function) will plot the losses across epochs.

## Package Code

Inside the train() function, the first step is to create the variables for the incoming training data.

[Image showcasing every operation just for one linear layer]

## Functions and Class

### Variable Class

The Variable class serves as a foundational building block for the project. It holds data, includes methods for forward and backward passes, and can optionally store gradients. The forward method returns the stored data, and the backward method computes and applies gradients.

**Methods in Variable Class:**

**forward():** This method returns the stored data value. It's the starting point of the forward pass.

backward(value, learning\_rate): The backward method receives a value and learning rate to update the variable's value. If requires\_grad is set to true, it computes gradients and updates the variable. If store\_gradient is enabled, the computed gradients are also stored for future analysis.

**\_\_str\_\_():** Provides a human-readable string representation of the variable, making it easier to inspect its state.

### MatrixMul Class

The MatrixMul class is responsible for matrix multiplication. It takes two Variable instances as input and computes both forward and backward passes:

**Methods in MatrixMul Class:**

**forward(matrixA, matrixB):** The forward method multiplies matrixA and matrixB. It stores the result in self.value.

**backward(value, learning\_rate):** The backward method computes gradients with respect to the inputs matrixA and matrixB and updates them using the provided learning rate. The calculated gradients are then backpropagated to these input matrices.

### ReLU Class

The ReLU class implements the Rectified Linear Unit (ReLU) activation function. It operates on the result of a previous operation.

**Methods in ReLU Class:**

**forward(prevOperation):** The forward method applies the ReLU activation function element-wise to the input prevOperation. It replaces negative values with zeros while passing positive values unchanged.

**backward(value, learning\_rate):** In the backward pass, this method computes gradients and backpropagates them to the previous operation. The gradient is calculated as the element-wise multiplication of a vector of ones (where the input was positive) and the incoming gradient value.

### Regression Loss Class

The Regression Loss class is responsible for calculating the mean squared error loss for regression tasks:

**Methods in RegressionLoss Class :**

**forward(original, predicted):** The forward method takes two variables, original (the ground truth) and predicted, and computes the mean squared error (MSE) loss by taking the mean of the squared differences between these two arrays.

**backward(value, learning\_rate):** In the backward pass, this method computes gradients and backpropagates them to the predicted variable. The gradient is calculated as (predicted - original) \* value \* 2 / original.shape[0], ensuring that the computed gradients match the data dimensions.

### Binary Loss Class

The Binary Loss class is responsible for computing binary cross-entropy loss for binary classification tasks. It contains methods for both forward and backward passes:

**Methods:**

**sigmoidFunction(x):** This method computes the sigmoid function of an input x,transforming logits into probabilities.

**backwardSigmoidFunction(value):** The backward method computes the differentiation of the sigmoid function, which is required during backpropagation.

**forward(original, predicted):** calculates the binary cross-entropy loss.

**backward(value, learning\_rate):** computes gradients and backpropagates them to the predicted variable. The gradient calculation involves the derivatives of the loss with respect to the predicted values and is scaled by the learning rate.

### Add Class

The Add class is used to perform element-wise addition of two matrices

**Methods in Add Class:**

**forward(f\_input1, f\_input2):** The forward method takes two input variables, f\_input1 and f\_input2, and computes their element-wise addition. The result is stored in self.value.

**backward(b\_grad, learning\_rate):** The backward method computes gradients and backpropagates them to f\_input1 and f\_input2. The computed gradient is b\_grad times a matrix of ones that matches the shape of the addition.

### Bias Class

The Bias class is a specialized subclass of the Variable class, designed to represent bias terms in neural networks. It includes a custom backward method for updating bias values and accumulates gradients for later analysis.

### Linear Class

The Linear class represents a fully connected layer in a neural network. It contains weight and bias parameters and an optional ReLU activation function:

**Methods in Linear Class:**

**forward(x):** The forward method computes the forward pass for the layer. It involves matrix multiplication, bias addition, and, if enabled, a ReLU activation function.

**backward(grad, learning\_rate):** In the backward pass, this method computes gradients and backpropagates them through the layer. If a ReLU activation is used, the gradients pass through it.
