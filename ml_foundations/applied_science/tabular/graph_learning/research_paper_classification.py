from typing import Dict

import os
import numpy as np
import torch
from torch import nn
import urllib.request
import tarfile

from ml_foundations.models.deep_models.layers import GraphAttentionLayer


class CoraDataset:
    """
    content [
              ['31336' '0' '0' ... '0' '0' 'Neural_Networks'],
              ['1061127' '0' '0' ... '0' '0' 'Rule_Learning'],
              ...
            ]
    citations [
                [35    1033]
                [35  103482]
                ...
              ]
    """

    labels: torch.Tensor
    classes: Dict[str, int]
    features: torch.Tensor
    adj_mat: torch.Tensor

    @staticmethod
    def _download():
        data_dir = "./data/"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            url = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
            filename = "cora.tgz"
            urllib.request.urlretrieve(url, os.path.join(data_dir, filename))
            with tarfile.open(os.path.join(data_dir, filename), "r:gz") as tar:
                tar.extractall(data_dir)

    def __init__(self, include_edges: bool = True):
        self.include_edges = include_edges
        self._download()
        content = np.genfromtxt("./data/cora/cora.content", dtype=np.dtype(str))
        citations = np.genfromtxt("./data/cora/cora.cites", dtype=np.int32)
        features = torch.Tensor(np.array(content[:, 1:-1], dtype=np.float32))
        self.features = features / features.sum(dim=1, keepdim=True)
        self.classes = {s: i for i, s in enumerate(set(content[:, -1]))}
        self.labels = torch.tensor(
            [self.classes[i] for i in content[:, -1]], dtype=torch.long
        )
        paper_ids = np.array(content[:, 0], dtype=np.int32)
        ids_to_idx = {id_: i for i, id_ in enumerate(paper_ids)}
        self.adj_mat = torch.eye(len(self.labels), dtype=torch.bool)

        if self.include_edges:
            for e in citations:
                e1, e2 = ids_to_idx[e[0]], ids_to_idx[e[1]]
                self.adj_mat[e1][e2] = True
                self.adj_mat[e2][e1] = True


class GAT(nn.Module):
    def __init__(
        self,
        in_features: int,
        n_hidden: int,
        n_classes: int,
        n_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.layer1 = GraphAttentionLayer(
            in_features, n_hidden, n_heads, is_concat=True, dropout=dropout
        )
        self.activation = nn.ELU()
        self.output = GraphAttentionLayer(
            n_hidden, n_classes, 1, is_concat=True, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_mat):
        x = self.dropout(x)
        x = self.layer1(x, adj_mat)
        x = self.activation(x)
        x = self.dropout(x)
        return self.output(x, adj_mat)


def accuracy(output: torch.Tensor, labels: torch.Tensor):
    return output.argmax(dim=-1).eq(labels).sum().item() / len(labels)


def main():
    device = "gpu" if torch.cuda.is_available() else "cpu"
    dataset = CoraDataset(include_edges=True)
    in_features = dataset.features.shape[1]
    n_hidden = 64
    n_classes = len(dataset.classes)
    n_heads = 8
    dropout = 0.6
    model = GAT(in_features, n_hidden, n_classes, n_heads, dropout).to(device)
    epochs = 1000
    optimizer = torch.optim.Adam
    loss_func = nn.CrossEntropyLoss()

    features = dataset.features.to(device)
    labels = dataset.labels.to(device)
    edges_adj = dataset.adj_mat.to(device)
    edges_adj = edges_adj.unsqueeze(-1)  # add a third dim for heads

    idx_rand = torch.randperm(len(labels))
    idx_train = idx_rand[:500]
    idx_valid = idx_rand[500:]

    for epoch in epochs:
        model.train()
        optimizer.zero_grad()
        output = model(features, edges_adj)
        loss = loss_func(output[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()
        train_accuracy = accuracy(output[idx_train], labels[idx_train])

        model.eval()
        with torch.no_grad():
            output = model(features, edges_adj)
            val_loss = loss_func(output[idx_valid], labels[idx_valid])
            val_accuracy = accuracy(output[idx_valid], labels[idx_valid])

        print(
            f"epoch {epoch} / {epochs}, train loss: {loss}, train accuracy: {train_accuracy}, valid loss: {val_loss}, valid accuracy: {val_accuracy}"
        )


if __name__ == "__main__":
    main()
