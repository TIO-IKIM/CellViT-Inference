# -*- coding: utf-8 -*-
#
# Cell Classification Module
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essenfrom pathlib import Pathfrom pathlib import Path

import torch.nn as nn


class LinearClassifier(nn.Module):
    """Linear Classifier

    Args:
        embed_dim (int): Embedding dimension (input dimension)
        hidden_dim (int, optional): Hidden layer dimension. Defaults to 100.
        num_classes (int, optional): Number of output classes. Defaults to 2.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 100,
        num_classes: int = 2,
        drop_rate: float = 0,
    ):
        super(LinearClassifier, self).__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
