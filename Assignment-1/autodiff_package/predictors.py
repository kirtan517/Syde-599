import numpy as np


class Regression:
    '''
    Linear Regression.
    '''
    def __init__(self, expected, actual):
        self.expected = expected
        self.actual = actual

    def loss(self):
        '''
        Returns the sum of the losses for Linear Regression
        '''
        self.output = np.sum(np.power(self.expected - self.actual, 2))/2
        return self.output

    def gradient(self):
        '''
        Returns the difference between the `expected` and `actual` value.
        '''
        self.grad = self.expected - self.actual
        return self.grad
    
class Binary:
    '''
    Binary Clasification.
    '''
    def __init__(self, expected, actual):
        self.expected = expected
        self.actual = 1 / (1 + np.exp(actual * -1))

    def loss(self):
        '''
        Returns the sum of the losses for Binary Classification.
        '''
        self.output = -1 * np.sum(self.actual * np.log(self.expected) + (1 - self.actual) * np.log(1 - self.expected))

    def gradient(self):
        '''
        Returns the difference between the `expected` and `actual` value.
        '''
        self.grad = self.expected - self.actual
        return self.grad