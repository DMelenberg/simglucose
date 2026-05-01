# CLAUDE.md — simglucose (DMelenberg fork)

This file provides durable guidance to Claude / Copilot / SWE-agents working in this repository.

> **When in doubt, do less. When confident, still do less.** Modifying a clinically validated ODE without evidence is worse than not modifying it.

## What this repository is

A **fork** of [`jxx123/simglucose`](https://github.com/jxx123/simglucose) (MIT). It is the simulator backend for [`DMelenberg/aid_simulation_suite`](https://github.com/DMelenberg/aid_simulation_suite) — an RL training stack for fully-closed-loop insulin pumps.

This repo provides:
- The FDA-accepted UVA/Padova 2008 ODE (13 base states, 30 virtual patients).
- A Breton 2009 exercise extension (3 added states: Y, Z, W).
- A `PhysioState` interface for time-varying multiplicative modulation (circadian, menstrual, daily SI drift, exercise).
- A PyTorch-batched ODE (`t1dpatient_torch.py`) for GPU-accelerated RL training.

The consumer repo embeds this as a git submodule pinned to a SHA on `master`.

## End goal

A policy trained against this simulator transfers to a real fully-closed-loop AID system — a pump deciding insulin dosing every ~5 minutes without patient interaction. Every modification here must be defensible against: *"would this break, mislead, or invalidate the policy when it sees a real human?"*

This project is **NOT**: a SaaS product, a clinical device, a glucose forecasting service, or a wearable data ingestion pipeline.

## Build & Run

```bash
# Editable install (uv-friendly, hatchling backend)
pip install -e .
# or
uv pip install -e .

# Run tests
pytest tests/ -v

# Smoke a single-patient simulation
python examples/run_gymnasium.py
```

## Architecture (relevant files only)

```
simglucose/
├── patient/
│   ├── base.py                # Abstract Patient class
│   ├── t1dpatient.py          # UVA/Padova ODE — THE core file. 16 states (13 base + 3 exercise). Dual code path at lines 166-180 (PhysioState vs. legacy hasattr).
│   └── t1dpatient_torch.py    # GPU-batched RK4 ODE for RL training (709 lines). Must remain glucose-equivalent to t1dpatient.py.
├── params/vpatient_params.csv # 30 virtual patients × 62 calibrated parameters. NEVER edit values.
├── controller/                # Basal-bolus baseline.
├── envs/simglucose_gym_env.py # Gymnasium-API wrapper (the consumer repo provides its own richer env).
├── simulation/                # Scenario/sensor/pump abstractions.
└── analysis/risk.py           # Kovatchev LBGI/HBGI risk index.

docs/
└── ROADMAP.md                 # LIVING. Read this first.
```

`MODIFICATIONS.md` is the authoritative log of every deviation from upstream.

## Key Design Decisions (DO NOT CHANGE without owner approval)

1. **Base 13-state ODE is frozen.** The equations in `t1dpatient.py` for states 0–12 (gastric emptying, glucose kinetics, insulin kinetics, subcutaneous glucose) match published UVA/Padova 2008. Modifying them invalidates the FDA acceptance lineage.
2. **62 calibrated parameters per patient are frozen.** Parameters in `vpatient_params.csv` are interdependent. Changing one breaks the steady state.
3. **All physiological extensions are multiplicative modulators.** Never replace a calibrated parameter; only multiply through `PhysioState`.
4. **`PhysioState` interface is the path forward.** Legacy `params._params.alpha_GE = …` mutation is retained for backward compatibility only and will be deprecated.
5. **`t1dpatient_torch.py` must remain glucose-output-equivalent to `t1dpatient.py`** under matched inputs and seeds, within the documented numerical tolerance (currently target: max |Δ| < 0.1 mg/dL over 24 h). Any divergence is a regression. **No test currently asserts this** — see ROADMAP M1.

## Scientific Integrity Rules

Every modification must satisfy ALL of:

1. **Citation.** A peer-reviewed paper with DOI in `MODIFICATIONS.md`.
2. **Validation envelope.** The cited paper's clinical scope (intensity, duration, fed/fasted state). State this explicitly.
3. **Test that asserts behaviour, not internals.** "Glucose drops with exercise during postprandial moderate aerobic" is a behaviour test. "x[14] == 0.42" is brittle.
4. **Backward compatibility.** When `phys=None` and exercise parameters are absent, the model must reduce exactly to base UVA/Padova.

## What you may NOT do without explicit owner approval

- Add or remove ODE states.
- Modify any of the 62 patient parameters in `vpatient_params.csv`.
- Change the `Action` or `Observation` namedtuple shapes.
- Replace `scipy.integrate.ode` with a different solver in `t1dpatient.py` (the consumer's GPU path uses RK4 separately).
- Introduce a new ODE term that is not a multiplicative modulator on existing terms.
- Merge upstream `jxx123/simglucose` without doing an explicit diff review of the calibrated parameters.

## Cross-repo coordination

This repository is consumed as a git submodule by `DMelenberg/aid_simulation_suite`. Workflow:

1. **Land changes here first.** Open a branch (matching the agent-branch convention), implement, add tests, merge to `master`. **Always push the branch publicly** before any consumer-side work depends on it.
2. Note the new `master` SHA.
3. **In the consumer repo**, file a separate PR that bumps `lib/simglucose` submodule pointer to the new SHA and adjusts dependent code.
4. Do not bundle simulator-internals changes with consumer-side changes in a single PR.
5. **Do not bump the consumer's submodule pointer to a SHA that is not yet on a public branch here.** Consumer PR #21 is the cautionary example: it pins `5e0290d`, a SHA that is not on this remote, so the PR cannot build. See ROADMAP C2.

## Test policy

- All new ODE behaviour gets a behavioural test in `tests/`.
- The `phys=None` legacy path and the `phys=PhysioState(...)` path must both be exercised by tests.
- Numerical regression: keep at least one rollout test that compares `t1dpatient.py` against `t1dpatient_torch.py` for a single patient over 24 h. (ROADMAP M1: not yet implemented.)
- Schema regression: assert `vpatient_params.csv` columns and dtypes; assert one steady-state sentinel for `adult#001`. (ROADMAP M5: not yet implemented.)

---

## Agent Instructions (read this first if you are an AI session)

### Session start protocol

1. **Read `docs/ROADMAP.md` first.** The latest dated section is the source of truth for current direction.
2. **Read `MODIFICATIONS.md`.** Every fork modification is documented there with citations.
3. **Read this file end-to-end.** The "DO NOT CHANGE" list is enforceable.
4. **Confirm your branch.** Develop on the branch you were told to use. Never push to `master`.
5. **Run `pytest tests/ -q` to establish a baseline** before changing anything. If the baseline is red, *that* is your task.

### Synthesis duty

If your task is exploratory or planning ("review", "what next", "plan"), append a new dated section to `docs/ROADMAP.md`. Snapshot the branch landscape, verify the merged state by reading source (don't trust prior docs), and raise at least 2 Socratic / scientific questions ranked CRITICAL / MAJOR / MINOR. Do not delete prior sessions. Use stable IDs (C1, M1, m1, ...).

### Cross-repo SHA contract (HARD RULE)

If your work changes simulator internals and the consumer repo will need a submodule bump:

```text
1. Push the simglucose-side branch to DMelenberg/simglucose first.
   The branch and its commit SHA must be reachable via
       git ls-remote https://github.com/DMelenberg/simglucose
   before any consumer-side PR is opened.
2. The consumer-side PR must reference the simglucose-side branch by name
   and SHA in its body. The two PRs must be linked from each other.
3. Never have a consumer PR pinning a SHA that is not on this remote.
   Consumer PR #21 is the open cautionary example.
```

### Scientific gatekeeping (this is medical-device-adjacent)

Before adding or modifying any ODE term, multiplier, or parameter handling, you must answer:

1. **Citation?** DOI of the supporting paper.
2. **Validation envelope?** Does your usage stay inside it?
3. **Backward compatibility?** Does `phys=None` still reduce to base UVA/Padova exactly?
4. **Test asserting behaviour?** Not a hard-coded internal value.

If you cannot answer all four, stop. Either narrow the change or escalate.

### Safe scope for an autonomous session

- Documentation updates (`MODIFICATIONS.md`, `docs/ROADMAP.md`, this file).
- Adding behavioural tests, especially for the dual-code-path correctness invariant (legacy vs. `PhysioState`).
- Adding the RK4-vs-dopri5 numerical regression test (ROADMAP M1).
- Adding the vpatient_params schema test (ROADMAP M5).
- Extending `PhysioState` to carry exercise parameters (closes Priority #1 in the consumer repo, ROADMAP step 1).
- Branch hygiene (with confirmation before deletion; never delete a branch with an open PR).

### Hand-off etiquette

- Commit messages should reference the ROADMAP question ID being closed (e.g. `test: assert RK4 vs dopri5 drift < 0.1 mg/dL (closes M1)`).
- After landing a change here, file a paired submodule-bump PR in the consumer repo with a clear summary linking back to this side's PR.
- Always end with `pytest tests/ -q` green.

### When to escalate to the human owner

- Any change touching states 0–12, the 62 calibrated parameters, or the `Action`/`Observation` namedtuples.
- Any merge-from-upstream that touches calibration data.
- Any test you had to skip to make pass.
- Any divergence > 0.1 mg/dL between `t1dpatient.py` and `t1dpatient_torch.py`.

## References

- Dalla Man C, et al. (2014). The UVA/PADOVA Type 1 Diabetes Simulator. *J Diabetes Sci Technol* 8(1):26–34. DOI: 10.1177/1932296813514502
- Dalla Man C, Breton MD, Cobelli C (2009). Physical Activity into the Meal Glucose-Insulin Model of Type 1 Diabetes. *J Diabetes Sci Technol* 3(1):56–67. DOI: 10.1177/193229680900300107
- Kovatchev BP, et al. (1998). Symmetrization of the blood glucose measurement scale. *Diabetes Care* 21(11):1655–1658.
- Brown SA, et al. (2015). Menstrual cycle SI variation. (citation in consumer repo's `docs/SCIENTIFIC_REFERENCES.md`)
