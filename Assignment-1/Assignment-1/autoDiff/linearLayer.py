import numpy as np
from .operators import Add, MatrixMul, ReLU


class Linear:
    """
       Fully-Connected Layer.
    """
    def __init__(self, weight, bias, activation_function=False, name="layer"):
        """
            weight and bias should be of type Variable
            activation_function can be set to `True` to use ReLU in layer.
            self.value -> type numpy array final value obtained after performing the forward pass
        """
        self.value = None
        self.finalOperation = None
        self.input = None
        self.weight = weight
        self.bias = bias
        self.addition = Add()
        self.matmul = MatrixMul()
        self.activation_function = activation_function
        self.name = name
        if self.activation_function:
            self.activation = ReLU()

    def forward(self, x):
        """
            Performs a forward pass for a Linear Layer.
            x -> type Variable or Bias
            return -> Object of most recent operation
        """
        self.input = x
        self.value = self.matmul.forward(x, self.weight)
        self.value = self.addition.forward(self.matmul, self.bias)
        self.finalOperation = self.addition
        if self.activation_function:
            self.value = self.activation.forward(self.addition)
            self.finalOperation = self.activation
        return self.finalOperation

    def backward(self, grad, learning_rate):
        """
            Performs a backward pass for a Linear Layer.
            grad -> gradients obtained from the previous layer during backpropogation
            learning_rate -> step size
            Just call the backward method of the last action performing class
        """
        if self.activation_function:
            self.activation.backward(grad, learning_rate)
        else:
            self.addition.backward(grad, learning_rate)

    def __str__(self):
        string = f"For Layer {self.name}"
        string += '\n'
        string += self.weight.__str__()
        string += '\n'
        string += self.bias.__str__()
        return string
