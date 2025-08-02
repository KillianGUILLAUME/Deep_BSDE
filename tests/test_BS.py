from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml

from benchmark import MC_sim_BS
from bsde.metrics import delta_hedge_BS
from bsde.payoff import basket_call
from bsde.solver import DeepBSDESolverBS

with open("experiments/config.yaml") as f:
    config = yaml.safe_load(f)


def payoff_func(S):
    """Basket_call payoff."""
    return basket_call(S, strike=config["payoff"]["K"])


net_cfg = dict(hidden=128, depth=5)


solver = DeepBSDESolverBS(
    model_params=config["model"],
    payoff_fn=payoff_func,
    T=config["T"],
    n_steps=config["n_steps"],
    net_cfg=net_cfg,
    lr=0.5e-4,
)

epochs, batch_size = 3_000, 15_000


solver.fit(n_epochs=epochs, batch_size=batch_size)
price = solver.predict()
print(f"Predicted price: {price:.4f}")

Path("mod").mkdir(exist_ok=True)
torch.save(solver.net.state_dict(), "mod/bsde_blackscholes.pt")

solver.net.eval()
pnl = delta_hedge_BS(
    simulator=solver.simulator,
    net=solver.net,
    payoff_fn=payoff_func,
    n_paths=batch_size,
    n_steps=config["n_steps"],
    T=config["T"],
    r=0,
)


plt.hist(pnl, bins=40, density=True)
plt.axvline(pnl.mean(), color="r", label=f"mean={pnl.mean():.4e}")
plt.title("P&L Distribution – basket call")
plt.xlabel("P&L")
plt.legend()
plt.show()


print("mean :", pnl.mean().item(), "stdev :", pnl.std().item())

price_mc_bs, ci95_bs = MC_sim_BS(
    model=solver.simulator,
    payoff_fn=payoff_func,
    n_paths=batch_size,
    n_steps=config["n_steps"],
    T=config["T"],
)

print("MC basket call :", price_mc_bs, " ±", ci95_bs)


"""
~/Desktop/python/.venv/bin/tensorboard --logdir runs   : http://localhost:6006
"""
