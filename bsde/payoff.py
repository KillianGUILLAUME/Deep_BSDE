import torch


def lookback_max(S_paths, strike=1):
    max_s = S_paths.max(dim=0).values
    return torch.clamp(max_s - strike, min=0.0)


def basket_call(
    S: torch.Tensor, strike: float = 1.0, weights: torch.Tensor = None
) -> torch.Tensor:
    """
    Basket call option payoff function.

    Args:
        S (torch.Tensor): Asset prices, shape (n_steps, n_paths, n_assets).
        K (float): Strike price.
        weights (torch.Tensor): Weights for each asset, shape (n_assets,).

    Returns:
        torch.Tensor: Payoff values, shape (n_paths,).
    """
    S_T = S[-1]
    d = S_T.shape[-1]
    if weights is None:
        weights = torch.full((d,), 1.0 / d, device=S.device)

    basket_price = (S_T * weights).sum(-1)
    return torch.clamp(basket_price - strike, min=0.0)
