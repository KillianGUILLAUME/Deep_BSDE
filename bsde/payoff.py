import torch


def lookback_max(S_paths, strike=1):
    max_s = S_paths.max(dim=0).values
    return torch.clamp(max_s - strike, min=0.0)


def basket_call(S, K=1.0):
    """
    Basket call option payoff function.

    Args:
        S (torch.Tensor): Tensor of shape (n_paths, n_assets) representing asset prices.
        K (float): Strike price for the basket call option.

    Returns:
        torch.Tensor: Payoff for each path, shape (n_paths,).
    """
    mean_S = S.mean(dim=1)
    return torch.clamp(mean_S - K, min=0.0)
