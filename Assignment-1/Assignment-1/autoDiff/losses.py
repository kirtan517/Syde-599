import numpy as np


class RegressionLoss:

    def __init__(self):
        self.dldy = None
        self.value = None
        self.predicted = None
        self.original = None

    def forward(self, original, predicted):
        self.original = original
        self.predicted = predicted
        self.value = np.mean((self.original.value - self.predicted.value) ** 2)
        return self

    def backward(self, value, learning_rate):
        ## Each of this functions should call the backward function of it's calle variables.
        ## scalar value times
        ## [[1]]  1* 1

        self.dldy = (self.predicted.value - self.original.value) * value * 2 / self.original.shape[0]
        self.predicted.backward(self.dldy, learning_rate)


class BinaryLoss():
    def __init__(self):
        self.logits = None
        self.sigmoidGrad = None
        self.value = None
        self.original = None
        self.predicted = None

    def sigmoidFunciton(self, x):
        return 1 / (1 + np.exp(-x))

    def backwardSigmoidFunction(self, value):
        # Computes the differentaition of the sigmoid function
        return self.sigmoidFunciton(value) * (1 - self.sigmoidFunciton(value))

    def forward(self, original, predicted):
        self.original = original
        self.predicted = predicted
        self.logits = self.sigmoidFunciton(self.predicted.value)
        self.value = np.mean(
            self.original.value * np.log(self.logits) * -1 + -1 * np.log(1 - self.logits) * (1 - self.original.value))
        return self

    def __call__(self):
        pass

    def backward(self, value, learning_rate):
        temp = -1 * self.original.value / self.logits + (1 - self.original.value) / (1 - self.logits)
        value = temp * value / self.original.shape[0]
        self.sigmoidGrad = self.backwardSigmoidFunction(self.predicted.value) * value  # this will be a vector
        self.predicted.backward(self.sigmoidGrad, learning_rate)
