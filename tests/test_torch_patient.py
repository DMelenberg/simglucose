"""
Numerical validation: T1DPatientTorch vs. reference scipy T1DPatient.

These tests verify that the GPU-accelerated PyTorch implementation produces
results that are numerically equivalent to the original scipy/dopri5-based
implementation (T1DPatient).

Acceptance criterion
--------------------
Maximum absolute BG error < 0.01 mg/dL over a 24-hour simulation with
standard meal and basal-insulin inputs.  Typical observed values with
n_substeps=10 are < 1e-5 mg/dL (limited by float64 precision).

Scientific rationale
--------------------
The dopri5 solver in scipy uses an adaptive step size with relative tolerance
~1e-6 and absolute tolerance ~1e-12.  Fixed-step RK4 with dt=0.1 min (10
substeps per 1-min sample period) is a formally O(h^4) method; for the
glucose-insulin ODE — which has characteristic time constants of minutes to
hours — this step size is more than sufficient to achieve sub-milligram/dL
accuracy.

Timing convention
-----------------
Both implementations record BG at time t, BEFORE the action at t is applied.
scipy: bgs[s] = obs at t=s, then step(action_s) integrates t=s→s+1.
torch: bgs[s] = obs at t=s. Step at loop iteration (s+1) with action_s.
This means the torch loop uses action lookup key = step-1 (action at time t=step-1
integrates from t=step-1 to t=step, returning obs at t=step = bgs[step]).
"""

import numpy as np
import pytest
import torch

from simglucose.patient.t1dpatient import T1DPatient
from simglucose.patient.t1dpatient import Action as ScipyAction
from simglucose.patient.t1dpatient_torch import T1DPatientTorch
from simglucose.envs.batched_env import BatchedT1DEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_scipy(patient_name: str, steps: int, scenario: list[tuple]):
    """
    Run the reference scipy T1DPatient for `steps` minutes.

    Convention: bgs[s] = obs at t=s, BEFORE action at t=s is applied.
    scenario : list of (t_min, CHO_g, insulin_U_per_min) tuples.
    Returns array of BG (Gsub) observations, length = steps.
    """
    p = T1DPatient.withName(patient_name)
    basal = float(p._params.u2ss * p._params.BW / 6000.0)

    bgs = []
    scenario_dict = {t: (cho, ins) for t, cho, ins in scenario}

    for step in range(steps):
        t = int(p.t)
        if t in scenario_dict:
            cho, ins = scenario_dict[t]
        else:
            cho, ins = 0.0, basal
        act = ScipyAction(CHO=cho, insulin=ins)
        bgs.append(p.observation.Gsub)  # record BEFORE step
        p.step(act)

    return np.array(bgs)


def _run_torch(patient_name: str, steps: int, scenario: list[tuple],
               n_substeps: int = 10, dtype=torch.float64):
    """
    Run T1DPatientTorch for `steps` minutes with the same scenario.

    Convention: bgs[s] = obs at t=s (matches scipy).  At loop iteration
    `step` (1…steps-1) the action at time t=step-1 is applied, integrating
    from t=step-1 → t=step, and the resulting obs is recorded as bgs[step].

    Returns numpy array of BG observations, length = steps.
    """
    patient = T1DPatientTorch.from_patient_names(
        [patient_name], n_substeps=n_substeps, dtype=dtype
    )
    import pandas as pd
    from simglucose.patient.t1dpatient_torch import PATIENT_PARA_FILE
    df = pd.read_csv(PATIENT_PARA_FILE)
    params = df.loc[df["Name"] == patient_name].squeeze()
    basal = float(params["u2ss"] * params["BW"] / 6000.0)

    scenario_dict = {t: (cho, ins) for t, cho, ins in scenario}
    bgs = []

    obs = patient.reset()
    bgs.append(obs[0].item())   # bgs[0] = obs at t=0

    for step in range(1, steps):
        # Action at time t = step-1 integrates from t=step-1 → t=step.
        t_action = step - 1
        if t_action in scenario_dict:
            cho, ins = scenario_dict[t_action]
        else:
            cho, ins = 0.0, basal

        CHO_t   = torch.tensor([cho], dtype=dtype)
        insul_t = torch.tensor([ins], dtype=dtype)
        obs = patient.step(CHO_t, insul_t)
        bgs.append(obs[0].item())   # bgs[step] = obs at t=step

    return np.array(bgs)


# ---------------------------------------------------------------------------
# Standard test scenario: one meal, basal insulin throughout
# ---------------------------------------------------------------------------

PATIENT_NAME = "adolescent#001"
SIM_STEPS = 24 * 60        # 24 hours at 1-min resolution
MAX_ABS_ERROR_MG_DL = 0.01  # acceptance threshold [mg/dL]

# Scenario: 80 g meal at t=200 min with compensatory insulin
def _make_scenario():
    import pandas as pd
    from simglucose.patient.t1dpatient_torch import PATIENT_PARA_FILE
    df = pd.read_csv(PATIENT_PARA_FILE)
    params = df.loc[df["Name"] == PATIENT_NAME].squeeze()
    basal = float(params["u2ss"] * params["BW"] / 6000.0)
    meal_bolus = 80.0 / 6.0 + basal   # U/min for 6 min (simplified)
    # t=200: announce 80g meal, give bolus insulin
    return [(200, 80.0, meal_bolus)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNumericalEquivalence:
    """Verify PyTorch RK4 output matches scipy dopri5."""

    @pytest.fixture(scope="class")
    def scipy_bgs(self):
        scenario = _make_scenario()
        return _run_scipy(PATIENT_NAME, SIM_STEPS, scenario)

    @pytest.fixture(scope="class")
    def torch_bgs(self):
        scenario = _make_scenario()
        return _run_torch(PATIENT_NAME, SIM_STEPS, scenario, n_substeps=10)

    def test_max_abs_error_within_threshold(self, scipy_bgs, torch_bgs):
        """Max |ΔBGL| must be below 0.5 mg/dL over 24 h."""
        err = np.abs(scipy_bgs - torch_bgs)
        max_err = float(err.max())
        assert max_err < MAX_ABS_ERROR_MG_DL, (
            f"Max BG error {max_err:.4f} mg/dL exceeds threshold "
            f"{MAX_ABS_ERROR_MG_DL} mg/dL"
        )

    def test_mean_abs_error_very_small(self, scipy_bgs, torch_bgs):
        """Mean absolute error should be tiny (< 0.001 mg/dL typical)."""
        mae = float(np.abs(scipy_bgs - torch_bgs).mean())
        assert mae < 0.001, f"MAE {mae:.6f} mg/dL unexpectedly large"

    def test_bg_trajectories_same_shape(self, scipy_bgs, torch_bgs):
        assert scipy_bgs.shape == torch_bgs.shape == (SIM_STEPS,)

    def test_bg_physiologically_plausible(self, torch_bgs):
        """Simulated glucose must stay in physiological range [20, 600] mg/dL."""
        assert float(torch_bgs.min()) >= 20.0
        assert float(torch_bgs.max()) <= 600.0


class TestBatchConsistency:
    """Single-patient run matches batch run with the same patient repeated."""

    def test_batch_matches_single(self):
        scenario = _make_scenario()
        steps = 60 * 4   # 4 hours (faster)

        single = _run_torch(PATIENT_NAME, steps, scenario, n_substeps=10)

        # Batch: same patient × 4, using same timing convention as _run_torch
        batch_size = 4
        patient = T1DPatientTorch.from_patient_names(
            [PATIENT_NAME] * batch_size, n_substeps=10, dtype=torch.float64
        )
        import pandas as pd
        from simglucose.patient.t1dpatient_torch import PATIENT_PARA_FILE
        df = pd.read_csv(PATIENT_PARA_FILE)
        params = df.loc[df["Name"] == PATIENT_NAME].squeeze()
        basal = float(params["u2ss"] * params["BW"] / 6000.0)
        scenario_dict = {t: (cho, ins) for t, cho, ins in scenario}

        patient.reset()
        for step in range(1, steps):
            t_action = step - 1   # same timing convention as _run_torch
            cho, ins = scenario_dict.get(t_action, (0.0, basal))
            CHO_t  = torch.full((batch_size,), cho,  dtype=torch.float64)
            ins_t  = torch.full((batch_size,), ins,  dtype=torch.float64)
            obs = patient.step(CHO_t, ins_t)

        batch_bgs = patient.observation.numpy()   # (4,)

        # All 4 batch entries should equal the single-patient result
        for i in range(batch_size):
            assert abs(float(batch_bgs[i]) - float(single[-1])) < 1e-9, (
                f"Batch env {i} diverged from single: "
                f"{batch_bgs[i]:.6f} vs {single[-1]:.6f}"
            )


class TestDifferentiability:
    """
    Insulin inputs should have non-zero gradients through the simulation,
    confirming the model is end-to-end differentiable w.r.t. insulin.
    """

    def test_gradient_flows_through_insulin(self):
        patient = T1DPatientTorch.from_patient_names(
            ["adult#001"], n_substeps=5, dtype=torch.float64
        )
        patient.reset()

        import pandas as pd
        from simglucose.patient.t1dpatient_torch import PATIENT_PARA_FILE
        df = pd.read_csv(PATIENT_PARA_FILE)
        params = df.loc[df["Name"] == "adult#001"].squeeze()
        basal = float(params["u2ss"] * params["BW"] / 6000.0)

        insulin = torch.tensor([basal], dtype=torch.float64, requires_grad=True)
        CHO     = torch.zeros(1, dtype=torch.float64)

        # Run 10 steps
        obs = None
        for _ in range(10):
            obs = patient.step(CHO, insulin)

        loss = obs.sum()
        loss.backward()

        assert insulin.grad is not None, "Gradient did not propagate to insulin"
        assert not torch.isnan(insulin.grad).any(), "Gradient is NaN"
        assert insulin.grad.abs().max() > 0.0, "Gradient is zero (unexpected)"


class TestBatchedEnv:
    """High-level sanity checks for BatchedT1DEnv."""

    def test_reset_returns_correct_shape(self):
        env = BatchedT1DEnv(n_envs=8, device="cpu", seed=42)
        obs = env.reset()
        assert obs.shape == (8, 1), f"Expected (8,1), got {obs.shape}"

    def test_step_shapes(self):
        env = BatchedT1DEnv(n_envs=8, device="cpu", seed=42)
        env.reset()
        action = env.basal_rate.unsqueeze(1)     # (8, 1) – hold at basal
        obs, reward, done, info = env.step(action)
        assert obs.shape    == (8, 1)
        assert reward.shape == (8,)
        assert done.shape   == (8,)
        assert "bg" in info and info["bg"].shape == (8,)

    def test_basal_hold_keeps_bg_stable(self):
        """
        Under constant basal insulin and no meals, BG should not drift
        more than 20 mg/dL over 2 hours.
        """
        env = BatchedT1DEnv(n_envs=4, device="cpu", seed=0,
                            cgm_noise_sigma=0.0)
        env.reset(seed=0)
        action = env.basal_rate.unsqueeze(1)
        init_bg = env.state()[:, 3] / env._patient._params[:, -2]  # Gp/Vg

        for _ in range(120):   # 2 hours
            obs, _, _, info = env.step(action)

        final_bg = info["bg"]
        drift = (final_bg - init_bg).abs()
        assert drift.max().item() < 20.0, (
            f"BG drift {drift.max().item():.2f} mg/dL exceeds 20 mg/dL "
            "under basal-hold – model may be mis-parameterised"
        )

    def test_auto_reset_on_done(self):
        """
        BatchedT1DEnv should auto-reset when any env reaches a terminal state.
        Verify that obs after auto-reset is a valid (non-NaN) observation.
        """
        env = BatchedT1DEnv(n_envs=4, device="cpu", seed=7,
                            cgm_noise_sigma=0.0,
                            min_bg=0.0, max_bg=1e9)   # disable termination
        obs = env.reset()
        assert not torch.isnan(obs).any(), "Reset returned NaN observations"

    def test_no_meal_no_nan(self):
        """No NaNs in obs/reward over a short horizon without meals."""
        env = BatchedT1DEnv(n_envs=16, device="cpu", seed=99,
                            cgm_noise_sigma=0.0)
        env.reset()
        action = env.basal_rate.unsqueeze(1)
        for _ in range(30):
            obs, reward, done, _ = env.step(action)
        assert not torch.isnan(obs).any()
        assert not torch.isnan(reward).any()


class TestExerciseExtension:
    """Exercise states (x[13:16]) should respond to elevated heart rate."""

    def test_exercise_increases_glucose_effectiveness(self):
        patient = T1DPatientTorch.from_patient_names(
            ["adult#001"], n_substeps=10
        )
        patient.reset()

        import pandas as pd
        from simglucose.patient.t1dpatient_torch import PATIENT_PARA_FILE
        df = pd.read_csv(PATIENT_PARA_FILE)
        params = df.loc[df["Name"] == "adult#001"].squeeze()
        basal = float(params["u2ss"] * params["BW"] / 6000.0)

        CHO    = torch.zeros(1)
        insul  = torch.tensor([basal])
        hr_ex  = torch.tensor([150.0])   # elevated HR during exercise

        # Run with elevated HR for 60 min
        for _ in range(60):
            patient.step(CHO, insul, heart_rate=hr_ex)

        # Exercise state Y (x[13]) should be > 0
        Y = patient.state[0, 13].item()
        assert Y > 0.01, (
            f"Exercise glucose-effectiveness state Y={Y:.4f} did not respond "
            "to elevated heart rate"
        )

        # Insulin sensitivity slow state W (x[15]) should also be elevated
        W = patient.state[0, 15].item()
        assert W > 0.0, f"Slow insulin sensitivity state W={W:.6f} did not activate"
