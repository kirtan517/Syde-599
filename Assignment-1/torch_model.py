from matplotlib import pyplot as plt
import torch.nn as nn
import torch
import pickle

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # Define two hidden layers and an output layer
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

with open('param.pkl', 'rb') as f:
    weights_and_biases = pickle.load(f)

model = SimpleNN()

model.fc1.weight.data = torch.tensor(weights_and_biases['w1']).float()
model.fc1.bias.data = torch.tensor(weights_and_biases['b1']).float()
model.fc2.weight.data = torch.tensor(weights_and_biases['w2']).float()
model.fc2.bias.data = torch.tensor(weights_and_biases['b2']).float()
model.fc3.weight.data = torch.tensor(weights_and_biases['w3']).float()
model.fc3.bias.data = torch.tensor(weights_and_biases['b3']).float()

input_data = torch.tensor(weights_and_biases['inputs']).float()
targets = torch.tensor(weights_and_biases['targets']).float().unsqueeze(1)

predictions = model(input_data)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

n_epochs = 2    # number of epochs to run
batch_size = 200  # size of each batch
batches_per_epoch = len(input_data) // batch_size

loss = loss_fn(predictions, targets)
loss.backward()
print(f"Pre-train loss: {loss/2}")

loss_history = []

for epoch in range(n_epochs):
    for i in range(batches_per_epoch):
        start = i * batch_size
        # take a batch
        Xbatch = input_data[start:start+batch_size]
        ybatch = targets[start:start+batch_size]
        # forward pass
        y_pred = model(Xbatch)
        loss = loss_fn(y_pred, ybatch)
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        # update weights
        optimizer.step()
    loss_history.append(loss)
    
# plt.figure(figsize=(10, 6))
# plt.plot(loss_history, label='Data over Time', color='blue')
# plt.title('Variable over Time')
# plt.xlabel('Time')
# plt.ylabel('Variable Value')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.autoscale(enable=True, axis='y', tight=True)  # Auto-scaling the y-axis

# plt.show()

print(f"After training loss: {loss/2}")
