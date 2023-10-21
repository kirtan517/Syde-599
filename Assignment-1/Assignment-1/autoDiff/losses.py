import numpy as np


class RegressionLoss:
    """
    Regression Loss
    """
    def __init__(self):
        """
            self.dldy -> gradient of loss with respect to inputs
            self.value -> type numpy array final value obtained after performing the forward pass
            self.predicted -> predicted of type Object obtained from pervious operation
            self.original -> original of type Object
        """
        self.dldy = None
        self.value = None
        self.predicted = None
        self.original = None

    def forward(self, original, predicted):
        """
        :param original: original of type Object
        :param predicted: predicted of type Object obtained from pervious operation
        :return: Current Object
        """
        self.original = original
        self.predicted = predicted
        self.value = np.mean((self.original.value - self.predicted.value) ** 2) / 2
        return self

    def backward(self, grad, learning_rate):
        """
        Call the backward of predicted operator
        :param grad: gradients obtained from the previous layer during backpropogation if nothing then [[1]] is passed
        :param learning_rate: step size
        :return: None
        """
        self.dldy = (self.predicted.value - self.original.value) * grad / self.original.value.shape[0]
        self.predicted.backward(self.dldy, learning_rate)


class BinaryLoss:
    """
    Binary Loss
    """
    def __init__(self):
        """
            self.logits -> logits obtained after passing the predited value through the sigmoid function
            self.sigmoidGrad -> gradient of the sigmoid function wrt to inputs
            self.value -> type numpy array final value obtained after performing the forward pass
            self.predicted -> predicted of type Object obtained from pervious operation
            self.original -> original of type Object
        """
        self.logits = None
        self.sigmoidGrad = None
        self.value = None
        self.original = None
        self.predicted = None

    def sigmoidFunciton(self, x):
        """
        :param x: numpy array
        :return: numpy array
        """
        return 1 / (1 + np.exp(-x))

    def backwardSigmoidFunction(self, value):
        """
        Computes the differentaition of the sigmoid function
        :param value: numpy array
        :return: numpy array
        """
        return self.sigmoidFunciton(value) * (1 - self.sigmoidFunciton(value))

    def forward(self, original, predicted):
        """
        :param original: original of type Object
        :param predicted: predicted of type Object obtained from pervious operation
        :return: Current Object
        """
        self.original = original
        self.predicted = predicted
        self.logits = self.sigmoidFunciton(self.predicted.value)
        self.value = np.mean(
            self.original.value * np.log(self.logits) * -1 + -1 * np.log(1 - self.logits) * (1 - self.original.value))
        return self

    def backward(self, grad, learning_rate):
        """
        Call the backward of predicted operator
        :param grad: gradients obtained from the previous layer during backpropogation if nothing then [[1]] is passed
        :param learning_rate:  step size
        :return: None
        """
        temp = -1 * self.original.value / self.logits + (1 - self.original.value) / (1 - self.logits)
        grad = temp * grad / self.original.shape[0]
        self.sigmoidGrad = self.backwardSigmoidFunction(self.predicted.value) * grad  # this will be a vector
        self.predicted.backward(self.sigmoidGrad, learning_rate)
