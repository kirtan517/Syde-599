import numpy as np
from operators import Add, Mult
from nonlinearities import Relu


class FullyConnectedLayer():
    '''
    Fully-Connected Layer.
    '''
    def __init__(self, weights, bias, relu_activation=False):
        '''
        `relu_activation` can be set to `True` to use ReLU in layer.
        '''
        self.weights = weights
        self.bias = bias
        self.add = Add()
        self.mult = Mult()
        self.relu_activation = relu_activation
        if self.relu_activation:
            self.relu = Relu()
    
    def forward(self, input):
        '''
        Performs a forward pass for a Linear Layer.
        '''
        self.input = input
        self.output = self.mult.forward(self.input, self.weights)
        self.output = self.add.forward(self.output, self.bias)
        if self.relu_activation:
            self.output = self.relu.forward(self.output) 
        return self.output
    
    def backward(self, prev_grad, learning_rate):
        '''
        Performs a backward pass for a Linear Layer.\n
        First output gradient corresponds to `input`.\n
        Second output gradient corresponds to `weights`.\n
        Third output gradient corresponds to the `bias`.
        '''
        if self.relu_activation:
            self.grad = self.relu.backward(prev_grad)
            self.grad = self.add.backward(self.grad)        
        else:
            self.grad = self.add.backward(prev_grad)
        self.grad1, self.grad2 = self.mult.backward(self.grad)

        self.weights = self.weights - (learning_rate * self.grad2)
        self.bias = self.bias - (learning_rate * self.add.grad) 

        return self.grad1, self.grad2, self.add.grad