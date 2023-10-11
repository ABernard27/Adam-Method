# %%
# implementation de la méthode ADAM

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch
# %%

# Instantiate the MNIST dataset
dataset = MNIST(root='./data', train=True, download=True)
dataset.transform = ToTensor()

# Extraire les caractéristiques et les étiquettes
data = DataLoader(dataset, batch_size=128, shuffle=True)

# %%


# Define a simple model
# class SimpleModel(nn.Module):
#     def __init__(self):
#         super(SimpleModel, self).__init__()
#         self.linear = nn.Linear(28*28, 1)
#         self.logit = nn.Sigmoid()

#     def forward(self, x):
#         x = self.linear(x)
#         return self.logit(x)

class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(28*28, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        logits = self.linear(x)
        predictions = self.sigmoid(logits)
        return predictions


# Instantiate the model
model = LogisticRegressionModel()


# Instantiate the Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss = nn.NLLLoss()
stock = []

# Training loop
for epoch in range(10):
    for batch, (X, y) in enumerate(data):
        # Calculate the model predictions
        output = model(X.view(-1, 28*28))

        # Calculate the loss with logistic regression
        res = loss(output.squeeze(), y)

        # Reset the gradients
        optimizer.zero_grad()

        # Backpropagation and weight update
        res.backward()
        optimizer.step()

        # Stockage of the loss
    stock.append(res.item())

# %%

plt.plot(stock)
plt.show()
# %%
