import numpy as np


class Regression:
    '''
    Linear Regression.
    '''
    def __init__(self, target):
        self.target = target

    def loss(self, predicted):
        '''
        Returns the sum of the losses for Linear Regression
        '''
        #self.predicted = predicted
        self.output = np.sum(np.power(self.target - predicted, 2))/2
        return self.output

    def gradient(self, predicted):
        '''
        Returns the difference between the `expected` and `actual` value.
        '''
        self.grad = self.target - predicted
        return self.grad
    
class Binary:
    '''
    Binary Clasification.
    '''
    def __init__(self, target):
        self.target = target

    def loss(self, predicted):
        '''
        Returns the sum of the losses for Binary Classification.
        '''
        self.output = -1 * np.sum(predicted * np.log(self.target) + (1 - predicted) * np.log(1 - self.target))

    def gradient(self, predicted):
        '''
        Returns the difference between the `expected` and `actual` value.
        '''
        predicted = 1 / (1 + np.exp(predicted * -1))
        self.grad = self.target - predicted
        return self.grad