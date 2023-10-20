import numpy as np


class Variable:
    def __init__(self, inputShapes, data=None, requires_grad=True, name="Variable", store_gradient=False):
        self.shape = inputShapes
        self.requires_grad = requires_grad
        self.data = data.copy()
        self.name = name
        self.store_gradient = store_gradient
        if type(self.data) == type(None):
            self.value = np.random.rand(*self.shape)  # 2d vector
        else:
            self.value = data
        if self.store_gradient:
            self.gradients = []

    def forward(self):
        return self.value

    def backward(self, value, learning_rate):
        if self.requires_grad:
            self.value -= learning_rate * value
        if self.store_gradient:
            # TODO: confirm if they want the gradients to be stored with learning rate multiplied or not
            self.gradients.append(value.tolist())

    def __str__(self):
        string = f"For the Variable {self.name}"
        string += '\n'
        string += self.value.__str__()
        return string


class Bias(Variable):

    def __init__(self, inputShapes, data=None, requires_grad=True, name="Variable", store_gradient=False):
        super().__init__(inputShapes, data, requires_grad, name, store_gradient)
        self.update = None

    def backward(self, value, learning_rate):
        self.update = np.sum(value, axis=0)
        self.value -= self.update * learning_rate

        if (self.store_gradient):
            # TODO: confirm if they want the gradients to be stored with learning rate multiplied or not
            self.gradients.append(self.update.tolist())
