import numpy as np
import copy


class Variable:
    """
    Placeholder for the Variables x,y and weights
    """
    def __init__(self, inputShapes, data=None, requires_grad=True, name="Variable", store_gradient=False):
        """
        self.value -> type numpy array stores the most recent (updated) value after each epochs
        :param inputShapes: tuple mentioniong the shape of the current variable
        :param data: if provied will store the data in the self.value
        :param requires_grad: boolean if true will update the value
        :param name: name given to the variable type str
        :param store_gradient: boolean if true will store the gradient for each update
        """
        self.shape = inputShapes
        self.requires_grad = requires_grad
        self.name = name
        self.data = data
        self.store_gradient = store_gradient
        if type(self.data) == type(None):
            self.value = np.random.rand(*self.shape)  # 2d vector
        else:
            self.value = data
            self.data = data.copy()
        if self.store_gradient:
            self.gradients = []

    def forward(self):
        """
        :return: Current Object
        """
        return self

    def backward(self, grad, learning_rate):
        """
        Update the self.value with incoming gradients
        :param grad: gradients obtained from the previous layer during backpropogation
        :param learning_rate: step size
        :return: None
        """
        if self.requires_grad:
            self.value -= learning_rate * grad
        if self.store_gradient:
            self.gradients = copy.deepcopy(grad).tolist()

    def __str__(self):
        string = f"For the Variable {self.name}"
        string += '\n'
        string += self.value.__str__()
        return string


class Bias(Variable):
    """
    Bias type Variable
    """

    def __init__(self, inputShapes, data=None, requires_grad=True, name="Variable", store_gradient=False):
        """
        self.update -> need to sum up the gradient across dim = 0
        """
        super().__init__(inputShapes, data, requires_grad, name, store_gradient)
        self.update = None

    def backward(self, grad, learning_rate):
        """
        Update the self.value with incoming gradients
        :param grad: gradients obtained from the previous layer during backpropogation
        :param learning_rate: step size
        :return: None
        """
        self.update = np.sum(grad, axis=0)
        self.value -= self.update * learning_rate

        if (self.store_gradient):
            self.gradients = copy.deepcopy(grad).tolist()
