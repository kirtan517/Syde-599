import numpy as np


class Add():
    '''
    Adition with forward and backward pass.
    '''
    def __init__(self):
        pass

    def forward(self, input1, input2):
        '''
        Performs a forward pass.
        '''
        self.input1 = input1
        self.input2 = input2
        self.output = self.input1 + self.input2
        return self.output
    
    def backward(self, prev_grad):
        '''
        Receives the Loss Gradient from previous operator and performs a backward pass. 
        '''
        self.grad = prev_grad
        return self.grad
    
class Mult():
    '''
    Multiplication with forward and backward pass.
    '''
    def __init__(self):
        pass

    def forward(self, input1, input2):
        '''
        Performs a forward pass. It is order sensitive.
        '''
        self.input1 = input1
        self.input2 = input2
        self.output = np.matmul(self.input1, self.input2)
        return self.output
    
    def backward(self, prev_grad):
        '''
        Receives the Loss Gradient from previous operator and performs a backward pass.\n
        First output corresponds to `input1`.\n
        Second output corresponds to `input2`.
        '''
        self.grad1 = np.matmul(prev_grad, self.input2.T)
        self.grad2 = np.matmul(self.input1.T, prev_grad)
        return self.grad1, self.grad2
