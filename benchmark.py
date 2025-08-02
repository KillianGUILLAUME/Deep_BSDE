import torch
import yaml

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


def MC_sim_BS(model, payoff_fn, n_paths=200_000, n_steps=252, T=1, r=0.0):
    """
    Monte Carlo simulation for option pricing using a Black-Scholes model.

    Args:
        model: A callable that simulates paths of the underlying asset.
        payoff_fn: A callable that computes the payoff from the simulated paths.
        n_paths (int): Number of paths to simulate.
        n_steps (int): Number of time steps in each path.
        T (float): Time horizon for the simulation.

    Returns:
        float: Estimated price of the option.
    """
    S = model.sample_paths(n_paths, n_steps, T)
    payoff = payoff_fn(S) * torch.exp(
        torch.tensor(-r * T, device=S.device, dtype=S.dtype)
    )
    price = torch.mean(payoff).item()
    se = payoff.std(unbiased=True).item() / (n_paths**0.5)
    return price, 1.96 * se  # 95% confidence interval
