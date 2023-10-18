import torch.nn as nn
import torch
import pickle
import torch.optim as optim

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
    

with open('assignment-one-test-parameters.pkl', 'rb') as f:
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
# targets = torch.reshape(targets,(200,1))
optimizer = optim.SGD(model.parameters(), lr=0.01)

loss_function = nn.MSELoss(reduction="sum")
print(input_data.shape)
print(targets.shape)


n_epochs = 10
for i in range(n_epochs):
    output = model(input_data)
    loss = loss_function(output,targets)
    loss.backward()
    print(loss)
    optimizer.step()
    optimizer.zero_grad()



predictions = model(input_data)
print(predictions)

#%%
