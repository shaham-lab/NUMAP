import torch
import torch.optim as optim
from src.numap_model import NUMAP
from src.numap_loss import GraphCrossEntropyLoss
from src.numap_loss import UMAPLoss
from src.graph_utils import get_affinity_matrix


class NumapTrainer:
    def __init__(self, input_dim, hidden_layers, output_dim, learning_rate, W_high):
        self.model = NUMAP(input_dim, hidden_layers, output_dim)
        # self.criterion = GraphCrossEntropyLoss(W_high)
        self.criterion = UMAPLoss(edge_weights=W_high.flatten())
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def fit(self, train_loader, num_epochs):
        self.model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                # print(f'Nans in outputs: {torch.isnan(outputs).sum()}')
                # print(f'Shape of outputs: {outputs.shape}')
                # W_low = get_affinity_matrix(outputs, n_neighbors=5)
                # loss = self.criterion(W_low)
                loss = self.criterion(outputs)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')

    def evaluate(self, test_loader):
        self.model.eval()
        with torch.no_grad():
            running_loss = 0.0
            for inputs, labels in test_loader:
                outputs = self.model(inputs)
                W_low = get_affinity_matrix(outputs, n_neighbors=5)
                loss = self.criterion(W_low)
                running_loss += loss.item()
            print(f'Test Loss: {running_loss/len(test_loader)}')

    def transform(self, inputs):
        self.model.eval()
        with torch.no_grad():
            preds = self.model(inputs)
            return preds.numpy()
