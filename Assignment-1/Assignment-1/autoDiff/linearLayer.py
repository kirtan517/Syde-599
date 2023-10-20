import numpy as np
from .operators import Add, MatrixMul, ReLU


class Linear:
    def __init__(self, weight, bias, activation_function=False, name="layer"):
        # weight and bias should be of type Variable
        # activation_function True means we are using ReLU
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
        self.input = x
        self.value = self.matmul.forward(x, self.weight)
        self.value = self.addition.forward(self.matmul, self.bias)
        self.finalOperation = self.addition
        if self.activation_function:
            self.value = self.activation.forward(self.addition)
            self.finalOperation = self.activation
        return self.finalOperation

    def backward(self, grad, learning_rate):
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
