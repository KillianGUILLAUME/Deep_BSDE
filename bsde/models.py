import torch


class HestonSimulator:
    r"""
    Simule (S_t, v_t) sous Heston par schéma d'Euler.
      dS_t = μ S_t dt + √v_t S_t dW^S_t
      dv_t = κ(θ - v_t) dt + σ √v_t dW^v_t,  ⟨dW^S, dW^v⟩_t = ρ dt
    """

    def __init__(
        self,
        mu: float,
        kappa: float,
        theta: float,
        sigma: float,
        rho: float,
        s0: float = 1.0,
        v0: float = 0.04,
        device: str = "cpu",
    ):
        self.mu, self.kappa, self.theta = mu, kappa, theta
        self.sigma, self.rho = sigma, rho
        self.s0, self.v0 = s0, v0
        self.device = torch.device(device)

    def sample_paths(self, n_paths: int, n_steps: int, T: float):
        dt = T / n_steps
        sqrt_dt = dt**0.5

        # Bruit brownien corrélé :
        z = torch.randn(n_steps, n_paths, 2, device=self.device)
        dW_s = z[:, :, 0] * sqrt_dt
        dW_v = (self.rho * z[:, :, 0] + (1 - self.rho**2) ** 0.5 * z[:, :, 1]) * sqrt_dt

        S = torch.empty(n_steps + 1, n_paths, device=self.device)
        v = torch.empty_like(S)
        S[0], v[0] = self.s0, self.v0

        for t in range(n_steps):
            v[t + 1] = torch.clamp(
                v[t]
                + self.kappa * (self.theta - v[t]) * dt
                + self.sigma * v[t].sqrt() * dW_v[t],
                min=1e-8,
            )
            S[t + 1] = S[t] * torch.exp(
                (self.mu - 0.5 * v[t]) * dt + v[t].sqrt() * dW_s[t]
            )

        assert not torch.isnan(S).any(), "S contains NaN values"
        return S, v


class BlackScholesSimulator:
    r"""
    Simule (S_t) sous Black-Scholes par schéma d'Euler.
      dS_t = μ S_t dt + σ S_t dW^S_t
    """

    def __init__(
        self, mu: float, sigma: float, s0: float = 1.0, d=10, device: str = "cpu"
    ):
        self.mu, self.sigma, self.s0, self.d = mu, sigma, s0, d
        self.device = torch.device(device)

    def sample_paths(self, n_paths: int, n_steps: int, T: float):
        dt = T / n_steps
        sqrt_dt = dt**0.5
        drift = -0.5 * self.sigma**2 * dt

        dW_s = torch.randn(n_steps, n_paths, self.d, device=self.device) * sqrt_dt

        S = torch.full((n_steps + 1, n_paths, self.d), self.s0, device=self.device)

        for t in range(n_steps):
            S[t + 1] = S[t] * torch.exp((drift) * dt + self.sigma * dW_s[t])

        assert not torch.isnan(S).any(), "S contains NaN values"
        return S
