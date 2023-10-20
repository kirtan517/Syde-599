import numpy as np


class MatrixMul:

    def __init__(self):
        self.result1 = None
        self.result2 = None
        self.value = None
        self.matrixA = None
        self.matrixB = None

    def __call__(self):
        pass

    def forward(self, matrixA, matrixB):
        # matrixA is of Variable type
        # matrixB is of variable type
        # self.value is always numpy array in all the classes
        self.matrixA = matrixA
        self.matrixB = matrixB
        self.value = self.matrixA.value @ self.matrixB.value

        return self.value

    def backward(self, value, learning_rate):
        # value # matrix
        self.result1 = value @ self.matrixB.value.T  # this will be passed to A
        self.result2 = self.matrixA.value.T @ value  # this will be passed to B
        self.matrixA.backward(self.result1, learning_rate)
        self.matrixB.backward(self.result2, learning_rate)


class Add:
    def __init__(self):
        self.grad = None
        self.f_input1 = None
        self.f_input2 = None
        self.value = None

    def forward(self, f_input1, f_input2):
        self.f_input1 = f_input1
        self.f_input2 = f_input2
        self.value = f_input1.value + f_input2.value
        return self.value

    def backward(self, b_grad, learning_rate):
        self.grad = b_grad * np.ones(self.value.shape)

        self.f_input2.backward(self.grad, learning_rate)
        self.f_input1.backward(self.grad, learning_rate)


class ReLU:

    def __init__(self):
        self.result = None
        self.value = None
        self.prevOperation = None

    def forward(self, prevOperation):
        # lastOperation  will be the object of matrix mul
        self.prevOperation = prevOperation
        self.value = np.where(self.prevOperation.value < 0, 0, self.prevOperation.value)
        # self.value values will be positive
        return self.value

    def backward(self, value, learning_rate):
        temp = np.ones(self.value.shape)
        temp = np.where(self.value <= 0, 0, temp)
        self.result = temp * value
        self.prevOperation.backward(self.result, learning_rate)
