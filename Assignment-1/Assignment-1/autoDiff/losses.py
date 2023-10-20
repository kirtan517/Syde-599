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
        return self.value

    def backward(self, value, learning_rate):
        ## Each of this functions should call the backward function of it's calle variables.
        ## scalar value times
        ## [[1]]  1* 1

        self.dldy = (self.predicted.value - self.original.value) * value * 2 / self.original.shape[0]
        self.predicted.backward(self.dldy, learning_rate)


class BinaryLoss:
    def __init__(self):
        self.sigmoidGrad = None
        self.value = None
        self.logits = None
        self.predicted = None
        self.original = None

    def sigmoidFunciton(self, x):
        return np.reciprocal(np.exp(x * -1))

    def backwardSigmoidFunction(self, value):
        return self.sigmoidFunciton(value) * (1 - self.sigmoidFunciton(value))

    def forward(self, original, predicted):
        self.original = original
        self.predicted = predicted
        self.logits = self.sigmoidFunciton(self.predicted.value)
        # TODO:check if its addition or mean
        self.value = np.sum(
            self.original * np.log(self.logits) * -1 + -1 * np.log(1 - self.logits) * (1 - self.original))
        print(self.logits)
        print(self.value)
        return self.value

    def __call__(self):
        pass

    def backward(self, value):
        self.sigmoidGrad = self.backwardSigmoidFunction(self.predicted.value) * value  # this will be a vector
        self.backward(self.sigmoidGrad)
