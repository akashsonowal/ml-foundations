import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class RankNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RankNet, self).__init__()
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.hidden_layer(x))
        x = self.output_layer(x)
        return x

def pairwise_ranking_loss(y_pred, y_true):
    diff = y_pred - y_true
    loss = torch.mean(torch.log(1 + torch.exp(-diff)))
    return loss

def generate_pairwise_data(data, labels):
    num_samples, num_features = data.shape
    pairwise_data = []
    pairwise_labels = []

    for i in range(num_samples):
        for j in range(i+1, num_samples):
            if labels[i] != labels[j]:
                # Construct pairs with different labels (indicating which item should be ranked higher)
                pairwise_data.append(torch.abs(data[i] - data[j]))  # Using absolute difference
                pairwise_labels.append(1.0 if labels[i] > labels[j] else -1.0)

    pairwise_data = torch.stack(pairwise_data, dim=0)
    pairwise_labels = torch.tensor(pairwise_labels, dtype=torch.float32)
    return pairwise_data, pairwise_labels

def train_ranknet(model, data, labels, num_epochs, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        pairwise_data, pairwise_labels = generate_pairwise_data(data, labels)
        outputs = model(pairwise_data)
        loss = pairwise_ranking_loss(outputs, pairwise_labels.unsqueeze(1))
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

# Example data
# Let's assume we have 100 samples with 5 features each and pairwise labels
data = torch.tensor(np.random.rand(100, 5), dtype=torch.float32)
labels = torch.tensor(np.random.randint(0, 2, (100,)), dtype=torch.float32)

# Create and train the model
input_dim = 5
hidden_dim = 10
num_epochs = 1000
learning_rate = 0.01

model = RankNet(input_dim, hidden_dim)
train_ranknet(model, data, labels, num_epochs, learning_rate)