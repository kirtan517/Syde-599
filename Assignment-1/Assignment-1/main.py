import numpy as np
import pickle
from autoDiff import Variable, Bias, Linear, RegressionLoss
import matplotlib.pyplot as plt

N_EPOCHS = 10
BATCH_SIZE = 200
LEARNING_RATE = 0.01


def get_batches(x, y, batch_size):
    number_of_steps = x.shape[0] // batch_size

    # TODO: shuffle the dataset
    start = 0
    end = batch_size
    for i in range(number_of_steps):
        x_final = Variable(x[start: end].shape, x[start: end], requires_grad=False)
        y_final = Variable(y[start: end].shape, y[start:end], requires_grad=False)
        start += batch_size
        end += batch_size
        yield x_final, y_final
    if end > x.shape[0] > start:
        x_final = Variable(x[start:].shape, x[start:], requires_grad=False)
        y_final = Variable(y[start:].shape, y[start:], requires_grad=False)
        yield x_final, y_final


def dataloader():
    with open('assignment-one-test-parameters.pkl', 'rb') as f:
        data = pickle.load(f)
    inputs = data["inputs"]
    w1 = data["w1"]
    w2 = data["w2"]
    w3 = data["w3"]
    b1 = data["b1"]
    b2 = data["b2"]
    b3 = data["b3"]
    targets = np.reshape(data["targets"], (data["targets"].shape[0], 1))
    params = {
        "w1": w1.T,
        "w2": w2.T,
        "w3": w3.T,
        "b1": np.reshape(b1, (1, b1.shape[0])),
        "b2": np.reshape(b2, (1, b2.shape[0])),
        "b3": np.reshape(b3, (1, b3.shape[0]))
    }
    trainset = (inputs, targets)

    return trainset, params


def Plot(losses):
    plt.figure(figsize=(8, 6))

    plt.plot(np.arange(len(losses)), losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.title('Loss Plot')
    plt.savefig("LossPlot.jpeg")


def PrintGradients(*args):
    for i in args:
        print(i)


def train(n_epochs, batch_size, learning_rate,printGradientFirstLayer):
    trainset, params = dataloader()
    w1 = Variable(params["w1"].shape, data=params["w1"], name="w1", store_gradient=True)
    w2 = Variable(params["w2"].shape, data=params["w2"], name="w2", store_gradient=True)
    w3 = Variable(params["w3"].shape, data=params["w3"], name="w3", store_gradient=True)
    b1 = Bias(params["b1"].shape, data=params["b1"], name="b1", store_gradient=True)
    b2 = Bias(params["b2"].shape, data=params["b2"], name="b2", store_gradient=True)
    b3 = Bias(params["b3"].shape, data=params["b3"], name="b3", store_gradient=True)

    # training loop
    n_epochs = n_epochs
    batch_size = batch_size
    learning_rate = learning_rate
    # x = Variable(trainset[0].shape,trainset[0],requires_grad=False)
    # y_true = Variable(trainset[1].shape,trainset[1],requires_grad=False)

    # Initialize the netword
    linear1 = Linear(w1, b1, True)
    linear2 = Linear(w2, b2, True)
    linear3 = Linear(w3, b3, False)
    loss = RegressionLoss()

    # bookeeping the losses
    losses = []

    #Print the gradient of 1pair of 1st layer
    printGradientFirstLayer = printGradientFirstLayer

    # eache layer gradient the weight matrix 2 * 10 then average will be also 2 * 10  average over all exaples


    # run through each of the epochs
    for _ in range(n_epochs):
        for x, y_true in get_batches(trainset[0], trainset[1], batch_size=batch_size):
            # forward pass
            y = linear1.forward(x)
            y = linear2.forward(y)
            y = linear3.forward(y)

            loss_value = loss.forward(y_true, y).value

            losses.append(loss_value)

            # backward pass
            loss.backward(np.array([[1]]), learning_rate)

            if(printGradientFirstLayer):
                print("Layer 1 weight Gradients")
                print(linear1.weight.gradients)
                print("Layer 1 bias Gradients ")
                print(linear1.bias.gradients)
                return


    # Printing last loss
    # y = linear1.forward(x)
    # y = linear2.forward(y)
    # y = linear3.forward(y)
    #
    # loss_value = loss.forward(y_true, y)
    # print(loss_value)

    print(losses)
    # PrintGradients(linear1,linear2,linear3)
    # Plot(losses)


if __name__ == "__main__":
    # train(N_EPOCHS, 1, LEARNING_RATE,True)
    train(N_EPOCHS,BATCH_SIZE,LEARNING_RATE,False)
