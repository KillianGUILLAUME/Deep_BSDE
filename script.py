from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml

from bsde.metrics import delta_hedge
from bsde.payoff import lookback_max
from bsde.solver import BSDESolver

with open("experiments/config.yaml") as f:
    config = yaml.safe_load(f)


def payoff_func(S):
    """Look-back payoff."""
    return lookback_max(S, strike=config["payoff"]["K"])


net_cfg = dict(hidden=128, depth=4)

solver = BSDESolver(
    model_params=config["model"],
    payoff_fn=payoff_func,
    T=config["T"],
    n_steps=config["n_steps"],
    net_cfg=config["net"] if "net" in config else net_cfg,
    lr=config["lr"] if "lr" in config else 1e-4,
)

solver.fit(n_epochs=config["epochs"], batch_size=config["batch"])
price = solver.predict()
print(f"Predicted price: {price:.4f}")

Path("mod").mkdir(exist_ok=True)
torch.save(solver.net.state_dict(), "mod/bsde_lookback.pt")

solver.net.eval()
pnl = delta_hedge(
    simulator=solver.simulator,
    net=solver.net,
    payoff_fn=payoff_func,
    n_paths=config["batch"],
    n_steps=config["n_steps"],
    T=config["T"],
)


plt.hist(pnl, bins=40, density=True)
plt.axvline(pnl.mean(), color="r", label=f"mean={pnl.mean():.4e}")
plt.title("Distribution P&L – delta-hedge lookback")
plt.xlabel("P&L")
plt.legend()
plt.show()


print("mean :", pnl.mean().item(), "stdev :", pnl.std().item())
