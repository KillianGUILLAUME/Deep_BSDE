import torch
import torch.nn as nn


class FeedForwardNN(nn.Module):
    def __init__(self, hidden=64, depth=3):
        super(FeedForwardNN, self).__init__()
        layers = []
        input_dim, output_dim = 3, hidden

        for _ in range(depth):
            layers += [nn.Linear(input_dim, output_dim), nn.Tanh()]
            input_dim = output_dim

        self.fcn = nn.Sequential(*layers)
        self.y_head = nn.Linear(output_dim, 1)
        self.z_head = nn.Linear(output_dim, 2)

    def forward(self, t, s, v):
        x = torch.stack([t, s, v], dim=-1)
        h = self.fcn(x)
        y = self.y_head(h).squeeze(-1)
        z = self.z_head(h)

        return y, z
