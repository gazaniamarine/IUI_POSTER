# belief_exp — parameter experimentation harness for NavDP's belief system

Plays with `navdp.extensions.SubgoalBeliefBank`'s mean/covariance dynamics and its
direct downstream consumers (`RouteManager`, a sigma-driven scan/slow-down gate) to
find which parameter values track a goal best and produce honest uncertainty
estimates. **Nothing under `navdp/` is imported-and-copied or modified** — every belief
number here comes straight out of the real `SubgoalBeliefBank`/`RouteManager` classes.

## Why "mean" has no sweep range

`SubgoalBeliefBank.update()` either snaps `mu` to a fresh measurement or deterministically
dead-reckons it by the reported `odom_delta` — no constructor parameter touches `mu`. So
there's nothing to "sweep" for the mean; what's actually being tested is how *accurately*
`mu` tracks the true goal under different measurement/odometry noise (a property of the
**scenario**, randomized in `sweep.py`), not a knob on the bank. **Covariance** is the real
tunable surface (`sigma_init`, `sigma_visible`, `odom_noise`, `decay_factor`,
`large_uncertainty`).

## Environment

Everything here is numpy-only in principle, but `navdp.extensions.__init__` transitively
imports modules (`foresight_gate.py`, `safe_diffusion.py`) that `import torch` at module
load time, even though this harness never uses a GPU. Run under a conda env that has torch:

```bash
conda run -n sam2 python belief_exp/inspect_one.py
conda run -n sam2 python belief_exp/sweep.py --configs-n 200 --episodes-per-config 60
```

(`sam2` or `sam3` both work; plain `/usr/bin/python3` does not have torch installed.)

## Files

| file | purpose |
|---|---|
| `common.py` | imports the real navdp belief classes; defines a tiny bearing-following P-controller and a noise-free SE(2) ground-truth integrator (generic control math, not belief logic) |
| `scenario.py` | closed-loop episode simulator: drives the real `SubgoalBeliefBank` + `RouteManager` through a randomized occlusion/noise scenario. Also defines `InterventionConfig`, an optional switch from belief-mean steering to human-bearing steering once `sigma_ale` crosses `gate_cfg.sigma_ale_threshold` (see "Human intervention vs. belief-only" below) |
| `metrics.py` | scores a batch of episodes for one config: calibration + task-performance |
| `sweep.py` | CLI: paired random search over the param space → leaderboard CSV |
| `inspect_one.py` | CLI: run one config, print a step-by-step trace |
| `intervention_study.py` | CLI: reachability study comparing belief-only vs. human-intervention steering across a severity sweep of irreducible occlusion/odometry noise → CSV + optional illustrative trace |

## Quick start

```bash
# eyeball one config's behavior
conda run -n sam2 python belief_exp/inspect_one.py

# see the effect of an overconfident Sigma
conda run -n sam2 python belief_exp/inspect_one.py --sigma-visible 1e-4 --env-obs-noise 0.3

# small smoke-test sweep
conda run -n sam2 python belief_exp/sweep.py --configs-n 20 --episodes-per-config 20

# full sweep
conda run -n sam2 python belief_exp/sweep.py --configs-n 200 --episodes-per-config 60 \
    --out belief_exp/results/sweep_001.csv
```

## Parameter glossary (what's being swept)

**`SubgoalBeliefBank` (the belief bank itself — covariance/confidence only):**
- `sigma_init` — initial `Sigma` (uninitialized-but-seen state; rarely hit in practice).
- `sigma_visible` — `Sigma` snapped to this (times identity) every time the goal is freshly
  observed. Smaller = "I trust a fresh sighting completely."
- `odom_noise` — how much `Sigma` grows per dead-reckoned step while occluded. This is the
  bank's own *belief* about how noisy its odometry is — swept independently of
  `env_odom_noise_std` (the scenario's *actual* odometry noise) so the calibration metric
  can tell you whether that belief is well-calibrated to reality.
- `decay_factor` — per-step multiplicative decay on `confidence` while occluded.
- `large_uncertainty` — `Sigma` value for a goal that's never been seen at all.

**Others:**
- `RouteManager.success_radius` — the belief-mean distance under which the route pointer
  advances ("we've arrived").
- `sigma_ale_threshold` — the harness's own scan/slow-down gate: pause and rotate in place
  when `sigma_ale = sqrt(max(Sigma_xx, Sigma_yy))` exceeds this. This threshold (and the
  pause behavior) is harness-level policy, not navdp code — navdp's `EpistemicGate` needs a
  `RelationalBelief`-refined 13-column tensor to do anything (its `sigma_epi` reads `0.0`
  otherwise), and `RelationalBelief` has no trained checkpoint anywhere in this repo, so
  it's out of scope here. The uncertainty *number* being thresholded is still 100%
  navdp-sourced (`Sigma` from the real bank); only the "what to do about it" cutoff is new.

**Scenario (ground truth, not belief params — randomized per episode in `sweep.py`):**
- `env_obs_noise_std` — real measurement noise when the goal is visible.
- `env_odom_noise_std` — real noise corrupting the *reported* odometry (this is what
  actually makes `mu` drift during occlusion — separate from the bank's `odom_noise`,
  which only inflates `Sigma`).
- occlusion regime — independent per-step Bernoulli or bursty Markov streaks.

## Human intervention vs. belief-only: reachability under irreducible noise

`intervention_study.py` tests a specific claim: past some severity of occlusion +
odometry noise, `mu`'s drift during a long blind dead-reckoning streak is a property
of the *scenario*, not something any `BankConfig` can fix (this is the same fact
the "why mean has no sweep range" section above and `sigma_min_sweep.py` already
establish for the calibration side — here it's applied to closed-loop steering). A
controller that blindly steers off `mu` therefore drifts off the true goal, and can
even have `RouteManager` declare "arrived" while the robot is meters away (a false
advance). `scenario.InterventionConfig` models a human teleoperator who, watching
the raw feed, can still supply a roughly-correct bearing (own small, per-step,
*non-accumulating* noise) without having integrated that drift — and switches
steering to that bearing only while `sigma_ale = sqrt(max(Sigma_xx, Sigma_yy))`
(the real, navdp-sourced uncertainty) exceeds `gate_cfg.sigma_ale_threshold`. Both
conditions share one `BankConfig`/`RouteConfig`/`GateConfig` and the same paired
scenario seeds per severity level — `InterventionConfig.enabled` is the only thing
that differs, isolating "who supplies the steering bearing" as the single variable.

```bash
conda run -n sam2 python belief_exp/intervention_study.py \
    --out belief_exp/results/intervention_study_001.csv --trace
```

`--trace` searches a few seeds at the most severe level for one where `belief_only`
fails (or false-advances) and `human_intervention`, replayed on the *same* seed,
reaches the true goal, and prints both step traces side by side.

Severity is a 2D interpolation from mild to severe between `--mild/--severe-mean-streak-len`
(markov occlusion burst length) and `--mild/--severe-env-odom-noise` (real odometry
noise corrupting dead reckoning while occluded) — the two scenario knobs that drive
irreducible `mu` drift (see `EnvConfig`'s docstring in `scenario.py`).

New metrics (in `metrics.py`, apply to every study/sweep, not just this one):
- `true_success_rate` — fraction of episodes where the ROBOT's true final distance
  to the goal is `<= success_radius`, regardless of whether `RouteManager` ever
  declared `advanced`. This is the ground-truth reachability number the belief-based
  `advance_rate`/`false_advance_rate` don't directly give you — a false advance ends
  the episode "successfully" from the belief's point of view while this metric still
  correctly scores it a failure if the robot wasn't actually there.
- `human_active_frac` — mean per-episode fraction of steps steered by the human
  bearing instead of `mu`; `0.0` whenever `InterventionConfig.enabled=False`.

Result so far (`results/intervention_study_001.csv`, 100 paired episodes/level,
default severity range): `true_success_rate` delta (`human_intervention -
belief_only`) is essentially noise at severity 0 (`-0.03`, intervention barely
triggers) and turns solidly positive from severity 0.4 upward (`+0.10` to `+0.21`),
while `belief_only`'s `false_advance_rate` climbs from `0.0` to `0.16` and its
`mean_final_dist` roughly doubles (`0.40` → `0.71`–`0.88`) — belief-only drifts off
and increasingly fails (including falsely declaring arrival) exactly as severity
rises, while human-intervention's `mean_final_dist` stays flat around `0.40`–`0.47`.

## Metric glossary

- `mean_err_visible` / `mean_err_occluded` — `||mu - true_goal||`, split by visibility.
  Diagnostic; not directly controlled by any bank covariance parameter (see above).
- `calibration_nll` — mean Gaussian negative log-likelihood of the true error under
  `N(mu, diag(Sigma))`. Lower = `Sigma` better explains the actual error.
- `coverage_1sigma` / `coverage_2sigma` — empirical fraction of steps where the true
  per-axis error falls inside 1σ / 2σ. A well-calibrated `Sigma` gives ~68.3% / ~95.4%.
  `coverage_deviation` is the absolute distance from those nominal values (lower = better
  calibrated, not just "smaller Sigma" — an overconfident *or* underconfident Sigma both
  score badly here).
- `mean_final_dist` — closed-loop distance-to-goal at episode end (task performance).
- `advance_rate` — fraction of episodes where `RouteManager` ever advanced (found the goal).
- `mean_steps_to_advance` — among episodes that advanced, how fast.
- `false_advance_rate` — among episodes that advanced, the fraction where the true distance
  was still `> 2 * success_radius` at that moment — i.e. `RouteManager` was fooled by a
  noisy/miscalibrated `mu` into declaring arrival prematurely.
- `calibration_score` / `task_score` / `combined_score` — only computed in `sweep.py`
  (z-normalized *across the leaderboard*, so they're only meaningful relative to the other
  rows in the same sweep run — the raw columns above are the numbers to trust in isolation).
