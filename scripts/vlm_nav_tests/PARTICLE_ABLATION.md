# Particle-only NavDP/S2Diff ablation

## Controlled setup

All variants use the frozen NavDP actor, the same HLC trajectory energy, four
candidate modes, the same controller, rover radius, static world-coordinate
obstacle meshes, goal mesh, layouts, and matched random seeds. The planner mode
is always s2diff; gradient guidance is never used.

The reference is particle_full with eight particles, standard deviation 0.28,
temperature 0.25, a nominal anchor, Gibbs energy reweighting, collision
masking, diffusion-scaled proposal noise, and progressive guidance.

## Core ablations

| Variant | Only change | Question answered |
|---|---|---|
| particle_p1 | One anchored particle | Are local alternative trajectories needed at all? |
| particle_sigma0 | Eight identical particles | Is particle diversity-not merely particle count-responsible for improvement? |
| particle_uniform_weights | Uniform rather than Gibbs weights | Does trajectory energy actually select better local samples? |
| particle_no_collision_mask | Colliding particles retain posterior weight | Is hard safety gating needed during denoising? Final candidate rejection remains active. |
| particle_no_anchor | No exact NavDP clean estimate among particles | Does preserving the learned proposal stabilize local search? |
| particle_fixed_noise_scale | Constant proposal width at all DDPM steps | Is matching exploration to diffusion uncertainty useful? |
| particle_constant_guidance | Full guidance at every DDPM step | Does gradually increasing guidance prevent early noisy overcorrection? |
| particle_no_feedback | Guidance strength zero | Does feeding the particle posterior into reverse diffusion help beyond final HLC candidate ranking? |

These experiments establish necessity only if the full method improves the
relevant metrics consistently across matched layouts and seeds with uncertainty
intervals supporting the claimed effect. A tie means the tested data do not
show that component is necessary.

## Optional parameter sweeps

Set RUN_SWEEPS=1 to additionally evaluate:

- particles per candidate: 2, 4, 8, and 16;
- particle standard deviation: 0, 0.10, 0.28, and 0.40;
- Gibbs temperature: 0.10, 0.25, and 0.75.

Particle count measures coverage versus latency. Standard deviation measures
exploration radius. Temperature controls weight selectivity.

## Metrics

Primary metrics are success, geometric collision rate, minimum rover-surface
clearance, deadlock rate, and path efficiency. Median and p95 latency measure
computational cost.

Two mechanism diagnostics are also stored:

- guide_rms: magnitude of particle-induced noise correction;
- ESS: effective sample size of final particle weights.

ESS near one means one particle dominates. ESS near the particle count means
weights are nearly uniform. ESS is diagnostic rather than an objective:
neither extreme is automatically better.

## Running

Quick head-on check:

    SEEDS=7 LAYOUTS=head_on SAVE_MEDIA=1 \
    ./run_navdp_particle_ablation_mesh.sh

Core multi-layout study:

    ./run_navdp_particle_ablation_mesh.sh

Sweeps:

    RUN_SWEEPS=1 ./run_navdp_particle_ablation_mesh.sh

For paper results:

    SEEDS="$(seq -s ' ' 1 30)" RUN_SWEEPS=1 \
    ./run_navdp_particle_ablation_mesh.sh

If scene files are elsewhere, set SCENE and TERRAIN_OBJ explicitly. Completed
episodes are skipped by default, so an interrupted study can resume.
