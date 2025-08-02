import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from bsde.models import BlackScholesMultiSimulator, HestonSimulator
from bsde.nets import FeedForwardNN, FeedForwardNNBS


def set_seed(epoch: int, seed_phase: int = 500, seed: int = 42):
    """
    Set the random seed for reproducibility.
    Args:
        epoch (int): Current epoch number.
        seed_phase (int): Phase of the seed cycle.
        seed (int): Base seed value.
    """

    if epoch <= seed_phase:
        torch.manual_seed(seed)
        np.random.seed(seed)
    else:
        torch.seed()
        np.random.seed()


class BSDESolver:
    def __init__(
        self,
        model_params,
        payoff_fn,
        T,
        n_steps,
        net_cfg=None,
        lr=1e-3,
        device="cpu",
        r=0,
    ):
        self.T = T
        self.n_steps = n_steps
        self.payoff_fn = payoff_fn
        self.device = torch.device(device)
        # Initialize the Heston simulator
        self.simulator = HestonSimulator(**model_params, device=device)

        # Initialize the neural network
        self.net = (
            FeedForwardNN(**net_cfg).to(self.device)
            if net_cfg
            else FeedForwardNN().to(self.device)
        )

        # Define the optimizer
        self.optimizer = Adam(self.net.parameters(), lr=lr)
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=150)

        self.r = r  # Risk-free rate, used for discounting in payoff calculation

        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir="runs/bsde_heston")

    def fwd_one_batch(self, n_paths):
        S, v = self.simulator.sample_paths(n_paths, self.n_steps, self.T)
        dt = self.T / self.n_steps

        t_grid = torch.arrange(self.n_steps + 1, device=self.device) * dt
        t_flattened = t_grid[:, None].repeat(1, n_paths).flatten()
        x_flat = S.reshape(-1)
        v_flat = v.reshape(-1)

        y_flat, z_flat = self.net(t_flattened, x_flat, v_flat)

        y_all = y_flat.view(self.n_steps + 1, n_paths)
        z_all = z_flat.view(self.n_steps + 1, n_paths, 2)
        Y_k = y_all[0]

        for k in range(self.n_steps):

            dW_s = (S[k + 1].log() - S[k].log() - 0.5 * v[k] * dt) / v[k].sqrt()
            dW_v = (
                v[k + 1]
                - v[k]
                - self.simulator.kappa * (self.simulator.theta - v[k]) * dt
            ) / (self.simulator.sigma * v[k].sqrt() + 1e-12)

            z_k = z_all[k]
            Y_k += -self.r * Y_k * dt + z_k[:, 0] * dW_s + z_k[:, 1] * dW_v

        payoff = self.payoff_fn(S) * torch.exp(
            torch.tensor(-self.r * self.T, device=S.device, dtype=S.dtype)
        )

        loss = torch.mean((Y_k - payoff) ** 2)
        return (
            loss,
            Y_k.mean(),
        )  # Y_K.mean() almost Y_0 after training, which is the price at t=0

    def fit(self, n_epochs=2000, batch_size=8192):
        for epoch in range(1, n_epochs + 1):
            self.optimizer.zero_grad()
            loss, y0 = self.fwd_one_batch(batch_size)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.optimizer.step()

            self.lr_scheduler.step(loss)

            self.writer.add_scalar("loss/train", loss.item(), epoch)
            self.writer.add_scalar("price/train", y0.mean().item(), epoch)
            self.writer.add_scalar(
                "learning_rate", self.optimizer.param_groups[0]["lr"], epoch
            )

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch:5d} | Loss: {loss.item():.4e} | price = {y0.item():.4f}"
                )

    @torch.no_grad()
    def predict(self, n_paths=10000):
        _, price = self.fwd_one_batch(n_paths)
        return price.item()


class DeepBSDESolverBS:
    def __init__(
        self,
        model_params,
        payoff_fn,
        T,
        n_steps,
        net_cfg=None,
        lr=1e-3,
        device="cpu",
        dimension=None,
        r=0,
    ):
        self.T = T
        self.n_steps = n_steps
        self.payoff_fn = payoff_fn
        self.device = torch.device(device)

        # Initialize the Black-Scholes multi simulator

        if dimension is not None:
            model_params = {**model_params, "d": dimension}
        self.simulator = BlackScholesMultiSimulator(**model_params, device=device)
        self.d = self.simulator.d

        # Initialize the neural network
        net_cfg = dict(net_cfg or {})
        net_cfg.setdefault("d", self.d)  # to insure d is set in net_cfg

        self.net = (
            FeedForwardNNBS(**net_cfg).to(self.device)
            if net_cfg
            else FeedForwardNNBS().to(self.device)
        )

        # Define the optimizer
        self.optimizer = Adam(self.net.parameters(), lr=lr)
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=100)

        self.r = r

        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir="runs/bsde_blackscholes")

    def fwd_one_batch(self, n_paths: int):
        S = self.simulator.sample_paths(n_paths, self.n_steps, self.T)

        dt = self.T / self.n_steps
        mu, sigma = self.simulator.mu, self.simulator.sigma

        t_grid = torch.arange(self.n_steps + 1, device=self.device) * dt
        t_flattened = t_grid[:, None].repeat(1, n_paths).flatten()
        x_flat = S.reshape(-1, self.d)
        y_flat, z_flat = self.net(t_flattened, x_flat)
        y_all = y_flat.view(self.n_steps + 1, n_paths)
        z_all = z_flat.view(self.n_steps + 1, n_paths, self.d)
        Y_k = y_all[0]

        Y_hist = [y_all[0]]
        Y_k = y_all[0]

        for k in range(self.n_steps):
            dW = (
                S[k + 1].log() - S[k].log() - (mu - 0.5 * sigma**2) * dt
            ) / sigma  # dW
            zk = z_all[k]
            drift_correction = ((mu - self.r) / sigma) * zk.sum(dim=1)
            Y_k += -self.r * Y_k * dt + drift_correction * dt + (zk * dW).sum(dim=1)

            Y_hist.append(Y_k)

        Y_all = torch.stack(Y_hist)

        payoff = self.payoff_fn(S) * torch.exp(
            torch.tensor(-self.r * self.T, device=S.device, dtype=S.dtype)
        )

        mse = torch.mean((Y_k - payoff) ** 2)
        pen_neg = torch.relu(-Y_all).pow(2).mean()  # to avoid negative price
        pen_bias = (Y_all[0].mean() - payoff.mean()).pow(2)  # to avoid bias in Y_0
        lamb = 3  # penalty for negative Y_k

        loss = mse + lamb * (pen_neg + pen_bias)
        return loss, Y_all[0].mean()  # approx Y_0

    def fit(self, n_epochs=2_000, batch_size=8_192):
        for epoch in range(1, n_epochs + 1):

            set_seed(epoch, seed_phase=500, seed=42)  # Set seed to contorl variance

            self.optimizer.zero_grad()
            loss, y0 = self.fwd_one_batch(batch_size)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.optimizer.step()

            self.lr_scheduler.step(loss)
            self.writer.add_scalar("loss/train", loss.item(), epoch)
            self.writer.add_scalar("price/train", y0.mean().item(), epoch)
            self.writer.add_scalar(
                "learning_rate", self.optimizer.param_groups[0]["lr"], epoch
            )

            if epoch % 50 == 0:
                print(
                    f"Epoch {epoch:5d} | Loss: {loss.item():.4e} | price = {y0.item():.4f}",
                    flush=True,
                )

    @torch.no_grad()
    def predict(self, n_paths=10_000):
        _, price = self.fwd_one_batch(n_paths)
        return price.item()
