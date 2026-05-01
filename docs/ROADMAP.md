# ROADMAP — simglucose (DMelenberg fork)

> Living document. Append a new dated section per session at the **top** of the dated-sections block. The latest section is authoritative.

This is the **simulator-side** roadmap. The companion repo `DMelenberg/aid_simulation_suite` consumes this fork as a git submodule and contains the RL training stack. Decisions here propagate to that repo via a submodule SHA bump.

**Project north star.** Provide a physiologically defensible UVA/Padova ODE backend on which an RL policy can be trained for a fully closed-loop insulin pump. Every modification to upstream simglucose must be traceable to a peer-reviewed paper and must preserve the calibrated 62-parameter steady state.

---

## Session 2026-05-01 — Cross-repo coordination, RK4 invariant gap, branch hygiene

### 1. Branch / PR landscape (snapshot)

| Branch | SHA | Purpose | Status |
|--------|-----|---------|--------|
| `master` | `f29142f` | Stable backend. Current submodule pin in PRs #22-#30 of consumer repo. | Live. |
| `claude/epic-clarke-sC7Ix` | `f29142f` | This session's working branch (parity with master). | Active. |
| `claude/epic-clarke-3P183` | `f29142f` | Prior session's working branch. Same SHA as master — only adds `CLAUDE.md` + `docs/ROADMAP.md` (paired with consumer PR #5, draft). | Active draft, not merged. |
| `claude/epic-clarke-Kz2Lx`, `claude/epic-clarke-tAps2` | `f29142f` | Prior agent runs that produced no PR. | Cleanup candidates. |
| `claude/epic-clarke-YZT7L` | `f29142f` | Prior agent run. | Cleanup candidate. |
| `exercise-model` | `9e0404e` | Historical exercise-extension branch; content already merged. | Safe to delete after owner confirmation. |
| `copilot/document-simglucose-project` | `4e30082` | Documentation seed; stale since `MODIFICATIONS.md` landed. | Cleanup candidate. |
| `copilot/sub-pr-3` | `01f3143` | GPU port sub-PR; already merged. | Cleanup candidate. |

**Open PRs:** PR #5 (paired with consumer's #24, prior session's roadmap draft). All cross-cutting work happens via the consumer repo's submodule pin and the corresponding feature branch here.

### 2. Cross-repo state — the missing SHA

The consumer repo's PR #21 (`feat: Priority #1 + #2 — PhysioState exercise fix, ...`) bumps the submodule pointer to:

```
5e0290dccf4f7e86c904cc3f85b6cd3e5453a656
```

That SHA is **not** on any branch listed by `git ls-remote DMelenberg/simglucose`. Either the corresponding simglucose-side commit (extending `PhysioState` with `alpha_GE`, `tau_GE_on`, etc., and routing exercise dynamics through the new dataclass instead of `params._params.alpha_GE`) lives only in a private clone, or the SHA was synthesised by the consumer-side patch hunk and never produced.

Until that branch is published on `DMelenberg/simglucose`, PR #21 cannot build. **This session's recommended action (mirrored in `aid_simulation_suite/docs/ROADMAP.md` M8):**

1. Reproduce the simglucose-side change of PR #21 on a real branch here, e.g. `claude/<id>-physio-exercise-fields`.
2. Confirm the diff matches what `physio_modulations.py` in PR #21 expects (`hr_current`, `hr_resting`, `hr_max`, `tau_GE_on`, `tau_SI_on`, `tau_SI_off` fields on `PhysioState`).
3. Push, get the new SHA, then re-target PR #21 to that SHA.

### 3. What is actually merged on `master`

Verified by reading source against `f29142f`, not by trusting docs:

- ✅ Base UVA/Padova 2008 ODE (13 states), 30 virtual patients, FDA-accepted preclinical model — preserved.
- ✅ Breton 2009 exercise extension: 3 added states (Y, Z, W), driven by PVO2 from heart-rate reserve.
- ✅ `PhysioState` interface in `simglucose/patient/t1dpatient.py:77-180`. `step(action, phys=None)` and `model(..., phys=None)` accept a struct of multiplicative modulators (`exercise_GE`, `composite_SI`, `composite_kabs`, `circadian_EGP`).
- ✅ Dual code path at `t1dpatient.py:166-180`: `if phys is not None` uses the new struct; `else` falls back to `hasattr(params, 'alpha_GE')` exercise injection. Backward-compatible.
- ✅ `t1dpatient_torch.py`: PyTorch-batched RK4 ODE (709 lines), 10 substeps/min, claimed < 0.1 mg/dL drift over 24 h vs. dopri5.
- ✅ Hatchling build, PEP 660 editable install, gymnasium (not gym).
- ❌ **`PhysioState` does not yet carry exercise dynamics fields (`alpha_GE`, `tau_*`, HR).** Exercise still needs the consumer to set `params.alpha_GE` or to mutate `_params`. PR #21 of the consumer repo extends `PhysioState` to fix this; the change requires the missing simglucose-side commit (§2).
- ❌ **No regression test of the PyTorch RK4 path against the SciPy reference.** The < 0.1 mg/dL claim is undocumented.
- ❌ **No CSV / patient-data tests for `vpatient_params`.** Schema drift risk if upstream rebases.
- ❌ **No `MODIFICATIONS.md` section documenting the Breton 2009 validation envelope.** The exercise model is validated for moderate, postprandial, 15–30 min sessions; consumers can drive arbitrary HR profiles for arbitrary durations with no warning.

### 4. Socratic / scientific review

#### CRITICAL

- **C1 (carried). Dual-code-path correctness.** When `phys is not None`, the legacy `hasattr(params, 'alpha_GE')` branch is skipped. But the consumer can still set `params.alpha_GE` via `_params` mutation. **Are exercise effects double-counted in any path where both `phys.exercise_GE` is populated and `params.alpha_GE` is set?** A unit test must run both paths with the same input and assert identical glucose trajectories.
- **C2 (NEW). The unpublished SHA.** Consumer PR #21 references a simglucose commit that is not reachable from any branch of this repo. The substantive change (PhysioState extension) is sound; the *publication* is missing. Severity is CRITICAL because any downstream merge of PR #21 will silently produce a non-buildable consumer repo. Fix as in §2.

#### MAJOR

- **M1 (carried). RK4 vs. dopri5 numerical drift unverified.** `t1dpatient_torch.py` claims a tolerance but no test asserts it. **Add a test:** for patient `adult#001`, run a 24 h rollout in both backends with the same meal/insulin schedule and assert max |Δ glucose| < 0.1 mg/dL. This must run in CI and gate every PR that touches either patient file.
- **M2 (carried). Composite multiplier range.** `phys.composite_SI = circadian * menstrual * daily * exercise` is computed in the consumer, not here. Joint range can be ~0.27 → ~5.3 with current defaults. **Question:** does the ODE remain numerically stable at the extremes? Add a sanity test that injects synthetic `phys` with extreme multipliers and asserts no NaN over 24 h.
- **M3 (carried). Exercise validation envelope.** Breton 2009 validated for moderate, postprandial, 15–30 min sessions. The fork accepts arbitrary HR profiles for arbitrary durations. **Action:** at minimum, document the envelope in `MODIFICATIONS.md` with a clear "use at your own risk outside this envelope" notice. Optionally: emit a runtime warning when an HR profile sustains > 30 min above the moderate band.
- **M4 (carried). `random_init_bg=True` distribution unspecified.** Constructor accepts a flag; upstream semantics are "±10% around basal". **Question:** is that documented and tested in this fork?
- **M5 (NEW). vpatient_params schema unsanity-checked.** `params/vpatient_params.csv` is the calibrated 62-parameter table for 30 patients. There is no test that asserts the column set, dtypes, or a sentinel patient (e.g. `adult#001` Vmt at steady state). **Add a schema test.** Without it, an upstream rebase or a Pandas DataFrame edit can silently corrupt the steady state.

#### MINOR

- **m1 (carried).** `t1dpatient.py` uses a 16-element `dxdt` whether or not `phys` is provided. Three states are wasted-but-harmless when exercise is unused. Consider gating on length-13 vs. length-16 init for downstream consumers that want the lean model. Low priority.
- **m2 (carried).** `Action` and `Observation` namedtuples are defined per-module here and in upstream. Drift risk if upstream renames fields. Pin upstream version in `requirements.txt` if upstream is ever vendored back as a dependency.
- **m3 (NEW).** `MODIFICATIONS.md` is the authoritative log of fork deviations. It currently has no entries for the PhysioState interface added in `3229ecda`. Add a short entry citing the multiplicative-modulator pattern's lineage from Breton 2009.

### 5. Strategic direction (next 4–6 sessions, ordered)

1. **Land the missing simglucose-side commit for PR #21 (closes C2).** Branch from `master`, extend `PhysioState` dataclass with `hr_current`, `hr_resting`, `hr_max`, `tau_GE_on`, `tau_SI_on`, `tau_SI_off`. Update `t1dpatient.py:model()` to source those from `phys` when present and fall through to the legacy `params` path otherwise. Push the branch; tell the consumer repo to retarget PR #21's submodule pointer.
2. **Test the RK4-vs-dopri5 invariant (closes M1).** A short, contained PR that protects every downstream RL claim about determinism between backends. Run a 24 h `adult#001` rollout in both backends with matched insulin/meal schedule; assert max |Δ glucose| < 0.1 mg/dL. Make this CI-gating.
3. **Joint-multiplier sanity test (closes M2).** Synthetic `PhysioState` with extreme multipliers; assert no NaN, no overflow, glucose stays in [10, 500] mg/dL over a 24 h roll.
4. **Document Breton envelope (closes M3).** Add a section to `MODIFICATIONS.md`. Optionally: a runtime warning in `t1dpatient.py` when the heart-rate input is sustained > 30 min above the moderate-intensity band.
5. **vpatient_params schema test (closes M5).** Assert column names, dtypes, and one steady-state sentinel value for `adult#001`. Run on every PR.
6. **Branch hygiene.** Delete `exercise-model`, `copilot/sub-pr-3`, `copilot/document-simglucose-project`, and the dead `claude/epic-clarke-*` branches after owner confirmation. **Never delete a branch that has an open PR pointing at it.**
7. **Dual-code-path correctness test (closes C1).** Run the legacy `params._params.alpha_GE = ...` path and the new `PhysioState`-with-exercise path with identical inputs; assert identical trajectories. Run only after step 1 lands.

### 6. Long-horizon questions (no decision yet)

- **Vendoring vs. re-fork from upstream.** Upstream `jxx123/simglucose` is single-maintainer and not actively versioned. Should this fork eventually carry its own version stamp and decouple, or maintain merge-from-upstream cadence? The deeper question is whether the consumer should depend on a published simglucose package or always carry the submodule. Tied to consumer's M9.
- **GPU backend parity.** `t1dpatient_torch.py` should remain bit-equivalent in glucose output to `t1dpatient.py` under fixed seeds. Is that an explicit invariant or aspirational? Step 2 above forces a decision.
- **Glucagon dual-hormone.** Explicitly out of scope (per `MODIFICATIONS.md`). Would require new states, new action space, new validation. Defer indefinitely.

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
- IDs (C1, M1, m1, ...) are stable across sessions. New questions get the next free integer in their severity tier; never re-use an ID.
- Cross-repo coordination: any change that requires the consumer repo to bump submodule SHA must be flagged in both ROADMAPs and the simglucose branch must be public **before** the consumer PR is opened.
