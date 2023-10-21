import numpy as np
from autoDiff import Variable,MatrixMul,RegressionLoss,Add,Bias

# Input variables
x = np.array([[1,2],[2,3],[3,4]])
y = np.array([[3.5],[2],[1]])

# Parameters
x_variable = Variable(x.shape,x,requires_grad=False)
y_variable = Variable(y.shape,y,requires_grad=False)
w = Variable((2,1),requires_grad=True,store_gradient=True)
bias = Bias((1,1),requires_grad=True,store_gradient=True)

#Operators
matrixMultiplication = MatrixMul()
add = Add()
loss = RegressionLoss()

# Forward computaiton
y_predict = loss.forward(y_variable,add.forward(matrixMultiplication.forward(x_variable,w),bias))

# Backward computation
y_predict.backward(np.array([[1]]),learning_rate=0.01)

print(w.gradients) # [[-0.9002136477659144], [-1.8244263889590928]]




