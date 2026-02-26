"""
Batched vectorized T1D simulation environment for PyTorch RL training.

This module provides ``BatchedT1DEnv``, a vectorized environment that runs
B independent T1D simulations in parallel on CPU or GPU.  It is designed as
a drop-in replacement for ``gymnasium``'s ``VectorEnv`` pattern but returns
``torch.Tensor`` directly, eliminating costly NumPy↔Tensor conversions inside
the RL training loop.

Key features
------------
* **Fully GPU-native**: observations, rewards, dones – all tensors on device.
* **Auto-reset**: when an episode terminates the environment is automatically
  reset without interrupting the batch.  A ``terminal_obs`` key in the info
  dict carries the final observation before reset (for off-policy algorithms).
* **Differentiable rewards**: the default risk-based reward is computed via
  PyTorch operations, so gradients can flow through the reward if needed
  (useful for model-based RL / differentiable planning).
* **Flexible action space**: accepts basal+bolus as a single insulin rate
  [U/min], matching the existing gymnasium API.

Usage example (RL training loop)
---------------------------------
>>> env = BatchedT1DEnv(
...     patient_names=[f"adult#00{i}" for i in range(1, 9)],
...     device="cuda",
... )
>>> obs = env.reset()                          # (B, 1) tensor on GPU
>>> for step in range(1000):
...     action = policy(obs)                   # (B, 1) tensor
...     obs, reward, done, info = env.step(action)
...     # auto-reset: done envs are already reset; terminal_obs in info
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

from simglucose.patient.t1dpatient_torch import (
    T1DPatientTorch,
    PATIENT_PARA_FILE,
    SAMPLE_TIME,
)

# ---------------------------------------------------------------------------
# Risk reward (differentiable PyTorch reimplementation of analysis/risk.py)
# ---------------------------------------------------------------------------

_MIN_BG = 20.0
_MAX_BG = 600.0


def _risk_scalar(bg: torch.Tensor) -> torch.Tensor:
    """
    Pointwise risk index  r ∈ [0, 100]  (Kovatchev et al., 1997).

    Identical formula to ``simglucose.analysis.risk.risk``, reimplemented
    with PyTorch operations so gradients can propagate.

    Parameters
    ----------
    bg : arbitrary shape  – blood glucose [mg/dL].

    Returns
    -------
    rl, rh, ri : three tensors of the same shape as bg.
    """
    bg = torch.clamp(bg, min=_MIN_BG + 1e-6, max=_MAX_BG - 1e-6)
    U  = 1.509 * (torch.log(bg) ** 1.084 - 5.381)
    ri = 10.0 * U ** 2
    rl = torch.where(U <= 0.0, ri, torch.zeros_like(ri))
    rh = torch.where(U >= 0.0, ri, torch.zeros_like(ri))
    return rl, rh, ri


def _risk_diff_reward(
    cgm_hist: torch.Tensor,
    window: int = 1,
) -> torch.Tensor:
    """
    Reward = risk(prev_CGM) - risk(current_CGM):  positive when BG improves.

    Parameters
    ----------
    cgm_hist : (B, T)  – CGM history, most-recent last.
    window : unused, kept for API compatibility.

    Returns
    -------
    reward : (B,)
    """
    if cgm_hist.shape[1] < 2:
        return torch.zeros(cgm_hist.shape[0], dtype=cgm_hist.dtype,
                           device=cgm_hist.device)
    _, _, ri_curr = _risk_scalar(cgm_hist[:, -1])
    _, _, ri_prev = _risk_scalar(cgm_hist[:, -2])
    return ri_prev - ri_curr


# ---------------------------------------------------------------------------
# CGM sensor noise (vectorised)
# ---------------------------------------------------------------------------

def _cgm_noise(
    bg: torch.Tensor,
    sigma: float = 4.0,
    seed_offset: int = 0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Additive Gaussian CGM noise (σ = 4 mg/dL, Dexcom-equivalent default).

    Clamps output to [39, 400] mg/dL matching sensor hardware limits.
    """
    noise = torch.randn_like(bg, generator=generator) * sigma
    return torch.clamp(bg + noise, min=39.0, max=400.0)


# ---------------------------------------------------------------------------
# BatchedT1DEnv
# ---------------------------------------------------------------------------

class BatchedT1DEnv:
    """
    Vectorized, GPU-native T1D simulation environment.

    Simulates B patients simultaneously.  All I/O uses ``torch.Tensor``.

    Parameters
    ----------
    patient_names : list of str, optional
        Patient identifiers such as "adult#001".  If None, all 30 virtual
        patients are used (repeated if B > 30).
    n_envs : int, optional
        Number of parallel environments.  Inferred from ``patient_names`` if
        not given.  When both are given ``n_envs`` takes priority (names are
        cycled / truncated accordingly).
    device : str or torch.device
        Computation device.
    n_substeps : int
        RK4 substeps per 1-min sample period.  10 (default) gives < 0.1
        mg/dL numerical error vs. scipy dopri5.
    dtype : torch.dtype
        float64 matches scipy reference; float32 is ~2× faster on GPU.
    cgm_noise_sigma : float
        Standard deviation of additive Gaussian CGM noise [mg/dL].
        Set to 0.0 for noise-free (perfect sensor) observations.
    max_bg : float
        Blood glucose ceiling above which an episode is marked done.
    min_bg : float
        Blood glucose floor below which an episode is marked done.
    seed : int, optional
        Global RNG seed for noise and random initialisation.
    random_init_bg : bool
        Randomise initial blood glucose (±10 % noise on BG states).
    reward_fn : callable, optional
        Custom reward function ``(cgm_hist: (B,T) tensor) -> (B,) tensor``.
        Defaults to risk-difference reward.
    """

    OBS_DIM = 1   # subcutaneous CGM [mg/dL]
    ACT_DIM = 1   # insulin basal rate [U/min]

    def __init__(
        self,
        patient_names: Optional[Sequence[str]] = None,
        n_envs: Optional[int] = None,
        device: str | torch.device = "cpu",
        n_substeps: int = 10,
        dtype: torch.dtype = torch.float64,
        cgm_noise_sigma: float = 4.0,
        max_bg: float = 600.0,
        min_bg: float = 10.0,
        seed: Optional[int] = None,
        random_init_bg: bool = False,
        reward_fn=None,
    ):
        self.device = torch.device(device)
        self.dtype  = dtype
        self.cgm_noise_sigma = cgm_noise_sigma
        self.max_bg = max_bg
        self.min_bg = min_bg
        self.random_init_bg = random_init_bg
        self.reward_fn = reward_fn if reward_fn is not None else _risk_diff_reward

        # ------------------------------------------------------------------
        # Resolve patient list
        # ------------------------------------------------------------------
        all_df = pd.read_csv(PATIENT_PARA_FILE)

        if patient_names is None:
            B = n_envs if n_envs is not None else len(all_df)
            indices = np.arange(B) % len(all_df)
            params_df = all_df.iloc[indices].reset_index(drop=True)
        else:
            B = n_envs if n_envs is not None else len(patient_names)
            cycled_names = [patient_names[i % len(patient_names)] for i in range(B)]
            params_df = pd.concat(
                [all_df.loc[all_df["Name"] == n] for n in cycled_names],
                ignore_index=True,
            )

        self.B = B

        # ------------------------------------------------------------------
        # Patient simulator
        # ------------------------------------------------------------------
        self._patient = T1DPatientTorch(
            params_df,
            device=device,
            n_substeps=n_substeps,
            dtype=dtype,
        )

        # ------------------------------------------------------------------
        # RNG for sensor noise
        # ------------------------------------------------------------------
        self._generator: Optional[torch.Generator] = None
        if seed is not None:
            self._generator = torch.Generator(device=self.device)
            self._generator.manual_seed(seed)

        # CGM history buffer for reward computation  (B, window)
        self._cgm_hist: Optional[torch.Tensor] = None
        self._cgm_window = 60   # 60 min of CGM history

        # Initialise state
        self._obs: torch.Tensor
        self.reset()

    # ------------------------------------------------------------------
    # Gymnasium-style API
    # ------------------------------------------------------------------

    def reset(
        self,
        mask: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Reset (all or masked) environments.

        Parameters
        ----------
        mask : (B,) bool tensor, optional
            Reset only environments where mask is True.
        seed : int, optional
            Override RNG seed.

        Returns
        -------
        obs : (B, 1)  subcutaneous CGM [mg/dL].
        """
        if seed is not None and self._generator is not None:
            self._generator.manual_seed(seed)

        bg = self._patient.reset(mask=mask, random_init_bg=self.random_init_bg)
        cgm = self._sense(bg)

        if mask is None or self._cgm_hist is None:
            # Full reset: initialise history with current CGM
            self._cgm_hist = cgm.unsqueeze(1).expand(self.B, self._cgm_window).clone()
        else:
            # Partial reset: overwrite masked rows
            self._cgm_hist = torch.where(
                mask[:, None].expand_as(self._cgm_hist),
                cgm.unsqueeze(1).expand_as(self._cgm_hist),
                self._cgm_hist,
            )

        self._obs = cgm.unsqueeze(1)      # (B, 1)
        return self._obs

    def step(
        self,
        action: torch.Tensor,
        CHO: Optional[torch.Tensor] = None,
        heart_rate: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Step all B environments by one sample period.

        Parameters
        ----------
        action : (B, 1) or (B,) tensor  – insulin basal rate [U/min].
        CHO : (B,) tensor [g], optional
            Carbohydrate meal announcement.  Defaults to zero (no meal).
        heart_rate : (B,) tensor [bpm], optional
            For exercise modulation.

        Returns
        -------
        obs : (B, 1)  subcutaneous CGM [mg/dL].
        reward : (B,)
        done : (B,) bool
        info : dict with keys:
            "bg"          – (B,) plasma glucose [mg/dL]
            "cgm"         – (B,) noisy CGM [mg/dL]
            "terminal_obs"– (B, 1) final obs for auto-reset envs (NaN otherwise)
        """
        insulin = action.to(device=self.device, dtype=self.dtype).squeeze(-1)  # (B,)
        if CHO is None:
            CHO = torch.zeros(self.B, dtype=self.dtype, device=self.device)
        else:
            CHO = CHO.to(device=self.device, dtype=self.dtype)

        # Advance ODE
        bg_sub = self._patient.step(CHO, insulin, heart_rate=heart_rate)  # (B,) sub glucose
        bg_plasma = self._patient.plasma_glucose                            # (B,)

        # Sense
        cgm = self._sense(bg_sub)   # (B,)

        # Update CGM history
        self._cgm_hist = torch.cat(
            [self._cgm_hist[:, 1:], cgm.unsqueeze(1)], dim=1
        )  # (B, window)

        # Done: BG out of survivable range
        done = (bg_plasma < self.min_bg) | (bg_plasma > self.max_bg)

        # Reward
        reward = self.reward_fn(self._cgm_hist)   # (B,)

        # Terminal observation (before auto-reset)
        terminal_obs = torch.full_like(cgm.unsqueeze(1), float("nan"))
        if done.any():
            terminal_obs = torch.where(
                done[:, None], cgm.unsqueeze(1), terminal_obs
            )
            self.reset(mask=done)

        obs = self._obs = self._patient.observation.unsqueeze(1)  # refreshed after reset

        info = {
            "bg":           bg_plasma,
            "cgm":          cgm,
            "terminal_obs": terminal_obs,
        }

        return obs, reward, done, info

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def obs(self) -> torch.Tensor:
        """Current observation (B, 1)."""
        return self._obs

    @property
    def basal_rate(self) -> torch.Tensor:
        """Patient-specific basal insulin rate [U/min]  (B,)."""
        return self._patient.basal_rate

    @property
    def n_envs(self) -> int:
        return self.B

    @property
    def observation_shape(self) -> tuple[int, ...]:
        return (self.OBS_DIM,)

    @property
    def action_shape(self) -> tuple[int, ...]:
        return (self.ACT_DIM,)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sense(self, bg: torch.Tensor) -> torch.Tensor:
        """Apply CGM sensor noise and hardware clamps."""
        if self.cgm_noise_sigma > 0.0:
            return _cgm_noise(bg, sigma=self.cgm_noise_sigma,
                              generator=self._generator)
        return bg.clone()

    def patient_names(self) -> list[str]:
        return self._patient.names

    def state(self) -> torch.Tensor:
        """Full ODE state (B, 16)."""
        return self._patient.state

    def close(self) -> None:
        """No-op (for gymnasium API compatibility)."""
        pass
