import torch
import torch.nn as nn


class FeedForwardNN(nn.Module):
    def __init__(self, hidden=64, depth=3):
        super(FeedForwardNN, self).__init__()

        assert hidden > 0, "hidden must be positive"
        assert depth > 0, "depth must be positive"
        assert isinstance(hidden, int), "hidden must be an integer"
        assert isinstance(depth, int), "depth must be an integer"

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


class FeedForwardNNBS(nn.Module):
    def __init__(self, d: int = 10, hidden=64, depth=3, s0: float = 1.0):
        super(FeedForwardNNBS, self).__init__()
        self.register_buffer("s0", torch.tensor(s0))

        assert d > 0, "d must be positive"
        assert hidden > 0, "hidden must be positive"
        assert depth > 0, "depth must be positive"
        assert isinstance(d, int), "d must be an integer"
        assert isinstance(hidden, int), "hidden must be an integer"
        assert isinstance(depth, int), "depth must be an integer"

        layers = []
        input_dim, output_dim = 1 + d, hidden

        for _ in range(depth - 1):
            layers += [nn.Linear(input_dim, output_dim), nn.ReLU()]
            input_dim = output_dim
        layers += [nn.Linear(input_dim, output_dim), nn.Tanh()]

        self.fcn = nn.Sequential(*layers)
        self.y_head = nn.Linear(output_dim, 1)
        self.z_head = nn.Linear(output_dim, d)

    def forward(self, t: torch.Tensor, s: torch.Tensor):
        s_norm = torch.log(s / self.s0)
        x = torch.cat([t.unsqueeze(-1), s_norm], dim=-1)
        h = self.fcn(x)
        y = self.y_head(h).squeeze(-1)
        y = y * y
        z = self.z_head(h)
        return y, z
