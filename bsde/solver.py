import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from bsde.models import HestonSimulator
from bsde.nets import FeedForwardNN


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
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=300)

        self.r = r  # Risk-free rate, used for discounting in payoff calculation

        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir="runs/bsde_heston")

    def fwd_one_batch(self, n_paths):
        S, v = self.simulator.sample_paths(n_paths, self.n_steps, self.T)
        dt = self.T / self.n_steps

        t0 = torch.zeros(n_paths, device=self.device)
        y, z = self.net(t0, S[0], v[0])

        Y_k = y

        for k in range(self.n_steps):

            dW_s = (S[k + 1].log() - S[k].log() - 0.5 * v[k] * dt) / v[k].sqrt()
            dW_v = (
                v[k + 1]
                - v[k]
                - self.simulator.kappa * (self.simulator.theta - v[k]) * dt
            ) / (self.simulator.sigma * v[k].sqrt() + 1e-12)

            Y_k = Y_k - self.r * Y_k * dt + z[:, 0] * dW_s + z[:, 1] * dW_v

            t_k1 = torch.full((n_paths,), (k + 1) * dt, device=self.device)
            _, z = self.net(t_k1, S[k + 1], v[k + 1])

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
