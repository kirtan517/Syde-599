import numpy as np


class Relu():
    '''
    Relu with forward and backward pass
    '''

    def __init__(self):
        pass

    def forward(self, input):
        '''
        Performs a forward pass.
        '''
        self.input = input
        self.output = np.maximum(0, self.input)
        return self.output
    
    def backward(self, prev_grad):
        '''
        Receives the Loss Gradient from previous operator and performs a backward pass.\n
        It uses element-wise multiplication for new gradient.
        '''
        mask = np.zeros_like(self.input)
        mask[self.input > 0] = 1
        self.grad = np.multiply(prev_grad, mask)
        return self.grad