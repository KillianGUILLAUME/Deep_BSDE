import torch
import yaml

from bsde.models import HestonSimulator
from bsde.payoff import lookback_max

with open("experiments/config.yaml") as f:
    config = yaml.safe_load(f)


def monte_carlo_sim(model, payoff_fn, n_paths=200000, n_steps=252, T=1, r=0):
    """
    Monte Carlo simulation for option pricing using a given model and payoff function.

    Args:
        model: A callable that simulates paths of the underlying asset.
        payoff_fn: A callable that computes the payoff from the simulated paths.
        n_paths (int): Number of paths to simulate.
        n_steps (int): Number of time steps in each path.
        T (float): Time horizon for the simulation.

    Returns:
        float: Estimated price of the option.
    """
    S, v = model.sample_paths(n_paths, n_steps, T)
    payoff = payoff_fn(S) * torch.exp(
        torch.tensor(-r * T, device=S.device, dtype=S.dtype)
    )
    price = torch.mean(payoff).item()
    se = payoff.std(unbiased=False).item() / (n_paths**0.5)
    return price, 1.96 * se  # 95% confidence interval


T = config["T"]
n_steps = config["n_steps"]
params = config["model"]
n_paths = config["batch"]

simulator = HestonSimulator(**params, device="cpu")


price_mc, ci95 = monte_carlo_sim(
    model=simulator,
    payoff_fn=lambda S: lookback_max(S, strike=config["payoff"]["K"]),
    n_paths=200_000,
    n_steps=config["n_steps"],
    T=config["T"],
)
print("MC look-back :", price_mc, " ±", ci95)
