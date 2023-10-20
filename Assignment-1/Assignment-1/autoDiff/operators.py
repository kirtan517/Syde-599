import numpy as np


class MatrixMul:

    def __init__(self):
        """
            As binary operator thus need to perform gradient with respect to each of the variable
            self.value -> type numpy array final value obtained after performing the forward pass
            self.matrixA -> Object obtained from pervious operations
            self.matrixB -> Object obtained from pervious operations
            self.diff_matrixA -> Stores the gradient with respect to matrixA
            self.diff_matrixB -> Stores the gradient with respect to matrixA
        """
        self.diff_matrixA = None
        self.diff_matrixB = None
        self.value = None
        self.matrixA = None
        self.matrixB = None

    def forward(self, matrixA, matrixB):
        """
        :param matrixA: Object obtained from pervious operations
        :param matrixB: Object obtained from pervious operations
        :return: Current Object
        """
        self.matrixA = matrixA
        self.matrixB = matrixB
        self.value = self.matrixA.value @ self.matrixB.value

        return self

    def backward(self, grad, learning_rate):
        """
        Call the backward of matrixA anb matrixB operator
        :param grad: gradients obtained from the previous layer during backpropogation
        :param learning_rate: step size
        :return: None
        """
        self.diff_matrixA = grad @ self.matrixB.value.T  # this will be passed to A
        self.diff_matrixB = self.matrixA.value.T @ grad  # this will be passed to B
        self.matrixA.backward(self.diff_matrixA, learning_rate)
        self.matrixB.backward(self.diff_matrixB, learning_rate)


class Add:
    def __init__(self):
        """
            As binary operator thus need to perform gradient with respect to each of the variable
            self.grad_wrt_inputs-> gradient with respect to inputs just np.ones()
            self.f_input1 -> Object obtained from pervious operation
            self.f_input2 -> Object obtained from pervious operation
            self.value -> type numpy array final value obtained after performing the forward pass
        """
        self.grad_wrt_inputs = None
        self.f_input1 = None
        self.f_input2 = None
        self.value = None

    def forward(self, f_input1, f_input2):
        """
        :param f_input1: Object obtained from pervious operation
        :param f_input2: Object obtained from pervious operation
        :return: Current Object
        """
        self.f_input1 = f_input1
        self.f_input2 = f_input2
        self.value = f_input1.value + f_input2.value
        return self

    def backward(self, grad, learning_rate):
        """
        Call the backward of f_input1 anb f_input2 operator
        :param grad: gradients obtained from the previous layer during backpropogation
        :param learning_rate: step size
        :return: None
        """
        self.grad_wrt_inputs = grad * np.ones(self.value.shape)
        self.f_input2.backward(self.grad_wrt_inputs, learning_rate)
        self.f_input1.backward(self.grad_wrt_inputs, learning_rate)


class ReLU:
    """
        ReLU Operator
    """

    def __init__(self):
        """
            self.prevOperation -> Object of pervious Operation
            self.value -> type numpy array final value obtained after performing the forward pass
        """
        self.result = None
        self.value = None
        self.prevOperation = None

    def forward(self, prevOperation):
        """
        :param prevOperation: Object of pervious Operation
        :return: Current Object
        """
        self.prevOperation = prevOperation
        self.value = np.where(self.prevOperation.value < 0, 0, self.prevOperation.value)
        return self

    def backward(self, grad, learning_rate):
        """
        Call the backward of previous operator
        :param grad: gradients obtained from the previous layer during backpropogation
        :param learning_rate: step size
        :return: None
        """
        temp = np.ones(self.value.shape)
        temp = np.where(self.value <= 0, 0, temp)
        self.result = temp * grad
        self.prevOperation.backward(self.result, learning_rate)
