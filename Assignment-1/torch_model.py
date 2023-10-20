from matplotlib import pyplot as plt
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
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)
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
# targets = torch.tensor(weights_and_biases['targets']).float().unsqueeze(1)
targets = torch.ones((200,1))
# targets = torch.reshape(targets,(200,1))
optimizer = optim.SGD(model.parameters(), lr=0.01)

loss_function = nn.BCELoss(reduction="mean")
print(input_data.shape)
print(targets.shape)

loss_history = []
n_epochs = 10
for i in range(n_epochs):
    output = model(input_data)
    loss = loss_function(output,targets)
    loss.backward()
    print(loss)
    optimizer.step()
    optimizer.zero_grad()
    loss_history.append(loss/2)



# predictions = model(input_data)
# loss_fn = torch.nn.BCELosss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# n_epochs = 200    # number of epochs to run
# batch_size = 10  # size of each batch
# batches_per_epoch = len(input_data) // batch_size
#
# loss = loss_fn(predictions, targets)
# loss.backward()
# print(f"Pre-train loss: {loss/2}")
#
# loss_history = []
#
# for epoch in range(n_epochs):
#     for i in range(batches_per_epoch):
#         start = i * batch_size
#         # take a batch
#         Xbatch = input_data[start:start+batch_size]
#         ybatch = targets[start:start+batch_size]
#         # forward pass
#         y_pred = model(Xbatch)
#         loss = loss_fn(y_pred, ybatch)
#         # backward pass
#         optimizer.zero_grad()
#         loss.backward()
#         # update weights
#         optimizer.step()
#     loss_history.append(loss)
#
# # Create a figure and axis
# plt.figure(figsize=(10,6))
# plt.plot([l.detach().numpy() for l in loss_history])
# #
# # # Add titles and labels
# plt.title('Training Loss Over Time')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.grid(True)
# #
# # # Show the plot
# plt.show()
#
# print(f"After training loss: {loss/2}")
