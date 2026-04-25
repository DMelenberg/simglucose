# ROADMAP — simglucose (DMelenberg fork)

> Living document. Append a new dated section per session. The latest section is authoritative.

This is the **simulator-side** roadmap. The companion repo `DMelenberg/aid_simulation_suite` consumes this fork as a git submodule and contains the RL training stack. Decisions here propagate to that repo via a submodule SHA bump.

**Project north star.** Provide a physiologically defensible UVA/Padova ODE backend on which an RL policy can be trained for a fully closed-loop insulin pump. Every modification to upstream simglucose must be traceable to a peer-reviewed paper and must preserve the calibrated 62-parameter steady state.

---

## Session 2026-04-25 — Synthesis & Direction

### 1. Branch / PR landscape (snapshot)

| Branch | SHA | Purpose | Status |
|--------|-----|---------|--------|
| `master` | `f29142f` | Stable backend for AID_Simulation_Suite | Current submodule pin in PRs #21/#22/#23 of consumer repo |
| `claude/epic-clarke-3P183` | `f29142f` | This session's working branch | Active |
| `claude/epic-clarke-YZT7L` | `f29142f` | Prior agent session | Same SHA — likely abandoned, candidate for cleanup |
| `exercise-model` | `9e0404e` | Historical exercise-extension branch | Already merged content; safe to delete |
| `copilot/document-simglucose-project` | `4e30082` | Documentation seed | Stale, not progressed since merge of `MODIFICATIONS.md` |
| `copilot/sub-pr-3` | `01f3143` | GPU port sub-PR (already merged) | Stale pointer, safe to delete |

**Open PRs:** none on this fork. All cross-cutting work happens via the consumer repo's submodule pin and the corresponding feature branch here.

### 2. What is actually merged on `master`

Verified by reading source, not by trusting docs:

- ✅ Base UVA/Padova 2008 ODE (13 states), 30 virtual patients, FDA-accepted preclinical model — preserved.
- ✅ Breton 2009 exercise extension: 3 added states (Y, Z, W), driven by PVO2 from heart-rate reserve.
- ✅ `PhysioState` interface: `t1dpatient.py:step(action, phys=None)` accepts a struct of multiplicative modulators (`exercise_GE`, `composite_SI`, `composite_kabs`, `circadian_EGP`).
- ✅ Dual code path: `if phys is not None` uses the new interface; `else` falls back to legacy `hasattr(params, "alpha_GE")` exercise injection.
- ✅ `t1dpatient_torch.py`: PyTorch-batched RK4 ODE for GPU training, claimed < 0.1 mg/dL drift over 24 h vs. dopri5 at 10 substeps/min.
- ✅ Hatchling build, PEP 660 editable install, gymnasium (not gym).
- ❌ **`PhysioState` does not yet carry exercise HR / time-constant fields.** Exercise still needs `params.alpha_GE`, `params.tau_*` to be injected by the caller. The caller (`aid_simulation_suite/src/glucose_rl/ode_patient.py`) does this via `patient._params.alpha_GE = …` mutation. This is the Priority #1 debt.
- ❌ **No regression test of the PyTorch RK4 path against the SciPy reference.** The < 0.1 mg/dL claim is undocumented.
- ❌ **No CSV / patient-data tests for vpatient_params** — schema drift risk if upstream rebases.

### 3. Socratic & scientific review

Severity in {CRITICAL, MAJOR, MINOR}.

#### CRITICAL

- **C1. Dual-code-path correctness.** When `phys is not None`, the legacy `hasattr(params, 'alpha_GE')` branch is skipped. But the consumer can still set `params.alpha_GE` via `_params` mutation. **Are exercise effects double-counted in any path where both `phys.exercise_GE` is populated and `params.alpha_GE` is set?** Need a unit test that runs both paths with the same input and asserts identical glucose trajectories.

#### MAJOR

- **M1. RK4 vs. dopri5 numerical drift unverified.** `t1dpatient_torch.py` claims a tolerance but no test asserts it. **Add a test:** for patient `adult#001`, run a 24 h rollout in both backends with the same meal/insulin schedule and assert max |Δ glucose| < 0.1 mg/dL.
- **M2. Composite multiplier range.** `phys.composite_SI = circadian * menstrual * daily * exercise` is computed in the consumer, not here. Joint range can be ~0.27 → ~5.3 with current defaults. **Question:** does the ODE remain numerically stable at the extremes? Add a sanity test that injects synthetic `phys` with extreme multipliers and asserts no NaN over 24 h.
- **M3. Exercise validation envelope.** Breton 2009 validated for moderate, postprandial, 15–30 min sessions. The fork accepts arbitrary HR profiles for arbitrary durations. **Question:** should the patient model log a warning (or refuse) when the input HR profile would extrapolate outside Breton's domain? At minimum, document the envelope in `MODIFICATIONS.md` with a clear "use at your own risk outside this envelope" notice.
- **M4. `random_init_bg=True` distribution unspecified.** The constructor accepts a flag but the upstream upstream-simglucose semantics are "±10% around basal". **Question:** is that documented and tested in this fork?

#### MINOR

- **m1.** `t1dpatient.py` uses a 16-element `dxdt` whether or not `phys` is provided. If exercise is unused, three states are wasted but harmless. Consider gating on length-13 vs. length-16 init states for downstream consumers that want the lean model.
- **m2.** `Action` and `Observation` namedtuples are defined per-module; they are also defined in upstream. Drift risk if upstream renames fields.

### 4. Strategic direction

Ordered.

1. **Test the RK4-vs-dopri5 invariant (M1).** This is a short, contained PR that protects every downstream RL claim about determinism between backends.
2. **Extend `PhysioState` to carry exercise parameters (closes Priority #1 of consumer repo).** Add fields `alpha_GE`, `alpha_SI`, `tau_GE_on`, `tau_SI_on`, `tau_SI_off`, `current_heart_rate`, `resting_heart_rate`, `max_heart_rate`. Update `t1dpatient.py:model()` to source them from `phys` first, with the legacy `params` path retained for backward compatibility but flagged as deprecated. Land this as a single PR; bump submodule pin in the consumer repo afterwards.
3. **Joint-multiplier sanity test (M2).** Synthetic `PhysioState` with extreme multipliers; assert no NaN, no overflow, glucose stays in [10, 500] mg/dL.
4. **Document Breton envelope (M3).** Add a section to `MODIFICATIONS.md` and a runtime warning in `t1dpatient.py` when the heart-rate input is sustained > 30 min above the moderate-intensity band.
5. **Branch hygiene.** Delete `exercise-model`, `copilot/sub-pr-3`, `claude/epic-clarke-YZT7L` after confirming with the owner.

### 5. Long-horizon questions

- **Vendoring vs. re-fork from upstream:** upstream `jxx123/simglucose` is single-maintainer and not actively versioned. Should this fork eventually carry its own version stamp and decouple, or maintain merge-from-upstream cadence?
- **GPU backend parity:** `t1dpatient_torch.py` should remain bit-equivalent in glucose output to `t1dpatient.py` under fixed seeds. Is that an explicit invariant or aspirational?
- **Glucagon dual-hormone:** explicitly out of scope (per `MODIFICATIONS.md`). Would require new states, new action space, new validation. Defer indefinitely.

---

## How to read / extend this file

- New session: append `## Session YYYY-MM-DD — <title>` above prior sessions, never edit prior sessions.
- Resolutions of CRITICAL / MAJOR items must reference the question ID (e.g. `closes M1`) in the commit message.
- Cross-repo coordination: any change that requires the consumer repo to bump submodule SHA must be flagged in both ROADMAPs.
