# Modifications to Simglucose (UVA/Padova Model)

This document tracks all modifications made to the base UVA/Padova Type 1 Diabetes simulator in this fork.

## Base Model

**Upstream**: [jxx123/simglucose](https://github.com/jxx123/simglucose) (MIT License)

**UVA/Padova 2008 Model**:
- 13-state ordinary differential equation (ODE) system
- 30 virtual patients (10 adolescent, 10 adult, 10 child)
- Each patient: 62 calibrated parameters from clinical data
- FDA-accepted for preclinical trials (2008)

**Reference**: Dalla Man C, et al. (2014). "The UVA/PADOVA Type 1 Diabetes Simulator." *J Diabetes Sci Technol* 8(1):26-34. DOI: [10.1177/1932296813514502](https://doi.org/10.1177/1932296813514502)

---

## Modification 1: Exercise Model (Breton 2009)

### Extension Summary
Added 3 new ODE states (total: 16 states) to model exercise effects on glucose-insulin dynamics.

### Scientific Basis
Dalla Man C, Breton MD, Cobelli C (2009). "Physical Activity into the Meal Glucose-Insulin Model of Type 1 Diabetes: In Silico Studies." *J Diabetes Sci Technol* 3(1):56-67. DOI: [10.1177/193229680900300107](https://doi.org/10.1177/193229680900300107)

### New State Variables

**x[13]: Y(t)** - Glucose effectiveness modulation
- Equation: `dY/dt = (PVO2 - Y) / tau_GE_on`
- Effect: Increases non-insulin-dependent glucose uptake during exercise
- Time constant: `tau_GE_on` ≈ 15 min (rapid on/off)

**x[14]: Z(t)** - Insulin sensitivity rapid component
- Equation: `dZ/dt = (PVO2 - Z) / tau_SI_on`
- Effect: Rapid increase in insulin action during exercise
- Time constant: `tau_SI_on` ≈ 15 min (rapid on)

**x[15]: W(t)** - Insulin sensitivity slow component
- Equation: `dW/dt = (Z - W) / tau_SI_off`
- Effect: Post-exercise insulin sensitivity persistence (2-48 hours)
- Time constant: `tau_SI_off` ≈ 120 min (slow off)

### Driving Input

**PVO2** - Normalized exercise intensity (0-1)
- Computed from heart rate reserve: `PVO2 = (HR - HRrest) / (HRmax - HRrest)`
- Approximates fraction of VO2max based on heart rate
- Physiological range: 0 (rest) to 1.0 (maximal exertion)

### Modified ODE Terms

**Glucose effectiveness** (line 158-160 in `t1dpatient.py`):
```python
GE_exercise_factor = 1.0 + alpha_GE * Y
Uiit = Fsnc * GE_exercise_factor  # Non-insulin glucose uptake
```
- `alpha_GE` ≈ 0.80 (80% increase during moderate exercise)
- Validated range: 0.66-0.97 (Breton 2009)

**Insulin sensitivity** (line 162-165 in `t1dpatient.py`):
```python
SI_exercise_factor = 1.0 + alpha_SI * W
Vmt = Vm0 + Vmx * x[6] * SI_exercise_factor  # Insulin-dependent uptake
```
- `alpha_SI` ≈ 1.20 (120% increase during/after exercise)
- Validated range: 0.81-1.55 (Breton 2009)

### Parameters Added to Patient Object

These are **NOT** in the base simglucose patient CSV. We add them dynamically:

```python
patient._params.alpha_GE          # Glucose effectiveness multiplier (default: 0.80)
patient._params.alpha_SI          # Insulin sensitivity multiplier (default: 1.20)
patient._params.tau_GE_on         # GE time constant (default: 15 min)
patient._params.tau_SI_on         # SI rapid on (default: 15 min)
patient._params.tau_SI_off        # SI slow off (default: 120 min)
patient._params.resting_heart_rate  # Resting HR (default: 70 bpm)
patient._params.max_heart_rate      # Max HR (default: 185 bpm)
patient._params.current_heart_rate  # Updated each step
```

### Validation

**Clinical validation scope** (from Breton 2009 paper):
- Moderate-intensity aerobic exercise (50-70% VO2max)
- Duration: 15-30 minutes
- Timing: Postprandial (2-3 hours after meals)

**Not validated for**:
- High-intensity interval training (HIIT)
- Resistance training (anaerobic)
- Prolonged exercise (>60 min)
- Fasted exercise (may trigger counter-regulatory hormones not modeled)

### Files Modified

- `simglucose/patient/t1dpatient.py`:
  - Line 119: Extended state vector from 13 to 16 states
  - Lines 156-248: Exercise dynamics implementation
  - Lines 221-248: New ODE equations for x[13], x[14], x[15]

### Impact on Base Model

**Backward compatibility**: If exercise parameters are not set, exercise states remain at zero and the model behaves identically to base UVA/Padova (exercise factors = 1.0).

**Steady state**: Exercise states are initialized to zero. At rest (HR = HRrest), PVO2 = 0, and all exercise effects are off.

---

## Modification 2: Safe Parameter Overrides

### Basal Insulin Rate Override

**Parameter**: `patient._params.u2ss`
**Location**: `src/glucose_rl/ode_patient.py:create_patient_from_profile()`

**What it does**: Overrides the steady-state basal insulin infusion rate.

**Safe because**:
- `u2ss` is a direct input to the insulin delivery equation
- Used in line 122 of `t1dpatient.py`: `basal = params.u2ss * params.BW / 6000`
- Doesn't affect calibrated physiological parameters
- Clinical equivalent: Adjusting pump basal rate setting

**Default behavior**: If not overridden, uses the patient's calibrated `u2ss` from CSV.

**Use case**: Testing basal-bolus strategies, simulating pump setting changes.

---

### Initial Glucose Override

**Parameter**: `patient.state[3]` (Gp - plasma glucose)
**Location**: `src/glucose_rl/env.py:reset()`

**What it does**: Sets the starting glucose level at episode start.

**Safe because**:
- Only affects initial condition, not dynamics
- ODE solver evolves state naturally from this starting point
- Clinical equivalent: Patient's actual glucose when starting CGM

**Default behavior**: If not overridden, uses randomized initial state from simglucose (±10% around basal).

**Use case**: Testing controller recovery from hypo/hyper, varying initial conditions for robustness.

---

### CGM Sensor Noise

**Location**: `src/glucose_rl/env.py:_get_observation()`
**Applied to**: Observation returned to agent (not ODE state)

**What it does**: Adds Gaussian noise to CGM readings to simulate real sensor error.

**Safe because**:
- Observation-level only (doesn't affect patient state)
- ODE simulation runs with perfect glucose values
- Only the agent's observation is noisy

**Realism**: Real CGM sensors have MARD (Mean Absolute Relative Difference) ~9-15%:
- Dexcom G6: ~9% MARD
- Abbott Libre 2: ~9.3% MARD
- Medtronic Guardian 3: ~8.7% MARD

**Default**: No noise (perfect sensor). User can configure via `glucose_noise_std` parameter.

**Use case**: Testing policy robustness to sensor error, offline RL dataset generation with realistic noise.

---

## Modification 3: Temporary Insulin Resistance (Stress/Illness)

**Location**: `src/glucose_rl/ode_patient.py` + scenario generators
**Status**: Planned (Phase 1B)

**What it does**: Temporarily modulates insulin sensitivity to simulate illness, stress, or hormonal effects.

**Implementation approach**:
```python
# In patient.model() ODE:
stress_factor = get_stress_factor(day_of_episode)  # Returns 0.6-1.0
SI_effective = SI_base * stress_factor
```

**Safe because**:
- Time-varying modulation of existing parameters (not breaking calibration)
- Clinical reality: Insulin needs increase 20-50% during illness
- Returns to baseline after stress period

**Physiological basis**:
- Illness (flu, infection): Cytokines → insulin resistance
- Menstrual cycle (luteal phase): Progesterone → 10-30% increased needs
- Stress: Cortisol → hepatic glucose production, reduced peripheral uptake

**Use case**: Testing RL adaptation to non-stationary dynamics, robustness to distribution shift.

---

## Modification 4: PhysioState Interface

**Location**: `simglucose/patient/t1dpatient.py` (the `model()` ODE function, `phys` keyword argument) + `simglucose/patient/t1dpatient_torch.py` (GPU-batched ODE counterpart).

**Status**: Implemented (added 2026-05-08 via simglucose PR #7 / commit `fac004a` — see T-003 in AID's `TASKS.md`).

**What it does**: A unified value-object carrying dimensionless multipliers for circadian, menstrual, daily-SI-drift, and exercise modulations of the calibrated UVA/Padova ODE. The ODE's `model()` accepts an optional `phys` keyword: when `phys is None`, the ODE reduces bit-for-bit to the calibrated UVA/Padova baseline (regression-tested on AID side via `tests/test_physio_exercise_refactor.py::test_default_path_bit_identical` with `atol=1e-6`); when `phys` is supplied, each multiplier scales its target ODE term. The 62 calibrated parameters in `params/vpatient_params.csv` are NEVER modified — the multipliers act as time-varying gains on existing terms.

**Fields**: The following dimensionless multipliers (or minute-valued time constants) are carried on `PhysioState`:

- `circadian_SI` (dimensionless): Insulin sensitivity multiplier for the dawn phenomenon; less than 1.0 during early morning hours when cortisol-driven insulin resistance peaks, approximately 1.0 at all other times.
- `circadian_EGP` (dimensionless): Endogenous glucose production multiplier for cortisol-driven EGP rise during dawn; greater than 1.0 at dawn, approximately 1.0 elsewhere.
- `menstrual_SI` (dimensionless): Insulin sensitivity multiplier for the menstrual cycle phase; less than 1.0 during the luteal phase (progesterone-driven insulin resistance) and greater than 1.0 during the follicular phase.
- `menstrual_kabs` (dimensionless): Gastric absorption rate multiplier for luteal-phase progesterone-driven gastric emptying slowdown; less than 1.0 during the luteal phase, 1.0 elsewhere.
- `daily_SI` (dimensionless): Day-to-day insulin sensitivity drift multiplier drawn from a LogNormal distribution around 1.0; captures inter-day variability in insulin requirements.
- `exercise_GE` (dimensionless): Glucose effectiveness multiplier during exercise, computed from the Breton 2009 Y state variable (non-insulin-dependent glucose uptake increase).
- `exercise_SI` (dimensionless): Exercise-driven insulin sensitivity multiplier computed from the Breton 2009 W state variable (post-exercise insulin sensitivity persistence).
- `exercise_PVO2` (dimensionless, range 0–1): Normalized exercise intensity (percent of VO2max), computed from heart-rate reserve by the AID layer before each ODE call.
- `exercise_tau_GE` (minutes): Glucose effectiveness time constant during exercise; governs the rate at which the Y state variable responds to changes in exercise intensity.
- `exercise_tau_SI_on` (minutes): Rapid insulin sensitivity activation time constant during exercise; governs the rise of the Z state variable.
- `exercise_tau_SI_off` (minutes): Slow insulin sensitivity decay time constant after exercise cessation; governs the decline of the W state variable.

The ODE also exposes two computed properties that aggregate the above fields: `composite_SI` (product of `circadian_SI`, `menstrual_SI`, `daily_SI`, and `exercise_SI`), applied to the insulin-dependent glucose uptake term `Vm0 + Vmx * x[6]`; and `composite_kabs` (currently equal to `menstrual_kabs`), applied to the gastric absorption rate.

**Citations**:

- **Circadian / dawn SI reduction**: Carroll MF, Schade DS (2005). "The dawn phenomenon revisited: implications for diabetes therapy." *Endocr Pract* 11(1):55-64. DOI: [10.4158/EP.11.1.55](https://doi.org/10.4158/EP.11.1.55)
- **Menstrual-cycle SI swing**: Brown SA, et al. (2015). "Fluctuations of hyperglycemia and insulin sensitivity are linked to menstrual cycle phases in women with T1D." *J Diabetes Sci Technol* 9(6):1192-1199. DOI: [10.1177/1932296815608400](https://doi.org/10.1177/1932296815608400)
- **Day-to-day SI drift (LogNormal model)**: Kovatchev BP, Breton M, Dalla Man C, Cobelli C (2009). "In silico preclinical trials: a proof of concept in closed-loop control of type 1 diabetes." *J Diabetes Sci Technol* 3(1):44-55. DOI: [10.1177/193229680900300106](https://doi.org/10.1177/193229680900300106)
- **Exercise dynamics (PVO2 + Y/Z/W ODE extension)**: See Modification 1 above (Dalla Man C, Breton MD, Cobelli C (2009). DOI: [10.1177/193229680900300107](https://doi.org/10.1177/193229680900300107)).

**Safe because**: When `phys=None`, the ODE is byte-for-byte equivalent to the pre-PhysioState UVA/Padova baseline — no code path inside `model()` changes when `phys` is absent. All multipliers are bounded: in worst-case extremes tested by `tests/test_joint_physio_sanity.py` on the AID side, the composite SI factor lies within `[0.1, 2.5]`; the 62 calibrated parameters in `params/vpatient_params.csv` are preserved unchanged. The PhysioState pattern is the only scientifically valid extension mechanism: it adds time-varying gains on existing ODE terms without altering any calibrated constant.

**Use case**: On the AID side, `src/glucose_rl/physio_modulations.py` builds a `PhysioState` instance at each one-minute ODE inner step using the current simulation clock, menstrual-cycle day, per-day SI multiplier, and heart-rate sensor reading. `src/glucose_rl/env.py` calls `compute_phys_state()` once per ODE inner step and passes the result to `T1DPatient.step()`, which forwards it to `model()`. `BatchedBloodGlucoseEnv` performs the equivalent GPU-side computation, constructing exercise and heart-rate modulation tensors and passing them through the PyTorch RK4 ODE solver in `simglucose/patient/t1dpatient_torch.py`.

**Related**: The exercise multipliers in `PhysioState` (`exercise_PVO2`, `exercise_tau_GE`, `exercise_tau_SI_on`, `exercise_tau_SI_off`, `exercise_GE`, `exercise_SI`) feed the Breton 2009 exercise extension documented in Modification 1 above; `PhysioState` generalises the 2009 paper's HR-driven mechanism to also carry circadian, menstrual, and daily-SI factors in a single unified value-object.

---

## Non-Modifications (What We Don't Change)

### Base 13-State ODE Equations

The core UVA/Padova equations remain **unchanged**:
- Gastric emptying (states 0-2)
- Glucose kinetics (states 3-4)
- Insulin kinetics (states 5-11)
- Subcutaneous glucose (state 12)

**Why**: These equations are clinically validated. Modifying them would invalidate the FDA acceptance and require re-validation.

### 62-Parameter Patient Calibrations

Each virtual patient's 62 parameters (Vg, Vi, kabs, Vmx, etc.) remain as calibrated in `vpatient_params.csv`.

**Why**: Parameters are interdependent. Changing one breaks steady-state balance and clinical realism. For example:
- Insulin sensitivity emerges from: Vmx, Km0, kp2, kp3, m1-m4 (10+ params)
- Carb absorption emerges from: kabs, kmax, kmin, b, d, f (6+ params)

Changing individual parameters without re-fitting the entire system produces unrealistic dynamics.

---

## Future Modifications (Planned)

### Circadian Rhythm (Phase 2)

**Scientific basis**: Dawn phenomenon - early morning (4-8 AM) insulin resistance due to cortisol, growth hormone.

**Implementation**: Time-varying modulation of insulin sensitivity:
```python
hour = (t / 60) % 24
if 4 <= hour < 8:
    circadian_factor = 0.7  # 30% more insulin resistant
else:
    circadian_factor = 1.0

SI_total = SI_base * circadian_factor * exercise_factor
```

**Safe because**: Modulation, not parameter replacement. Clinically observed phenomenon.

### Menstrual Cycle (Phase 3)

**Scientific basis**: Luteal phase (days 14-28) shows 10-30% increased insulin needs.

**Implementation**: Cycle-day dependent modulation similar to circadian.

### Dual-Hormone (Glucagon) Support (Phase 3+)

**Status**: Research-level extension (6+ months)

**Requires**:
- 3-5 new ODE states (glucagon kinetics)
- Glucagon-EGP dynamics
- Insulin-glucagon antagonism
- New action space
- Clinical validation

**Not a simple modification** - essentially a new simulator.

---

## Testing & Validation

All modifications are validated via:

1. **Unit tests** (`tests/test_ode_integration.py`):
   - Exercise states integrate correctly
   - Parameter overrides apply correctly
   - Steady state is maintained at rest

2. **Comparison against published figures**:
   - Breton 2009 Fig 4: Exercise-induced glucose drop
   - Breton 2009 Fig 5: Post-exercise insulin sensitivity

3. **Sanity checks**:
   - Glucose remains in physiological range (10-700 mg/dL)
   - Insulin on board is non-negative
   - ODE solver doesn't fail

---

## How to Verify Modifications

To check if you're using modified vs base simglucose:

```python
import numpy as np
from simglucose.patient.t1dpatient import T1DPatient

patient = T1DPatient.withID(1)

# Check state vector length
print(len(patient.state))
# Base simglucose: 13
# This fork: 16 (includes exercise states)

# Check for exercise parameters
if hasattr(patient._params, 'alpha_GE'):
    print("Exercise model available")
else:
    print("Base model only")
```

---



## Changelog

- **2024-Q4**: Added Breton 2009 exercise model (3 new states)
- **2025-02-15**: Documented all modifications, added safe parameter overrides
