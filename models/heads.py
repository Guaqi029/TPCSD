import torch.nn as nn


class Projector(nn.Module):
    def __init__(self, in_dim, proj_dim=128, hidden_dim=0):
        super().__init__()
        if proj_dim <= 0:
            raise ValueError("proj_dim must be > 0")
        if hidden_dim and hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, proj_dim),
            )
        else:
            self.net = nn.Linear(in_dim, proj_dim)

    def forward(self, x):
        return self.net(x)
