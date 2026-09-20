# Reach pilot protocol v0 — declared before training

Question: can compact, visually trained action-conditioned models predict short G1 arm motions well enough for shared image-goal CEM/MPC to improve reaching over hold/random controls?

This is development validation, not the sealed final compositional benchmark. No manipulation success is inferred from reaching. Both models receive the same canonical corpus and a current RGB image; both explicitly ignore proprioception internally in this first visual baseline.

- Corpus: 100 independently seeded reset episodes, 60 control steps each; seed 123. Right-arm target tracking plus Gaussian action noise. Four procedural proxy pairs: cube→target, banana→bowl, apple→bowl, cube→plate. Apple→plate is excluded entirely. Rejected commands never become fabricated transitions; retain short/failing episode accounting.
- Split: whole reset/session, deterministic 80/10/10 via seed123. Fit statistics only on training episodes. Validation used for model selection; test and Apple→Plate remain unused by tuning.
- Initial training: each backend seed0, 500 updates, batch16, horizon4, at most600s, CPU four threads. Checkpoint selection on fixed validation windows. Record all attempted runs, update counts and timings. A failed/negative pilot triggers documented diagnosis, not a positive-result claim.
- Planning: shared CEM horizon4,64 candidates,3 iterations,8 elites, seed matched per reset; only right xyz deltas vary within ±0.5, both hands stay open and remaining components fixed. These bounds are common action configuration, not backend-specific logic.
- Development evaluation: seeds10000–10004, max100 control steps, same reachable goal images produced by an oracle in a separate reset of the same scene. Goal/scoring uses a frozen target position; learned planning receives only RGB goal and current canonical sensors.
- Controls: hold (explicitly open hands and zero pose delta), random within the same action bounds, scripted oracle validating each goal. The normalized zero vector is not called a hold because its grasp fields mean half closed.
- Reach: right EE within0.025m of the frozen target for0.15 simulated seconds. Record initial/final distance, reach rate and sample count, planner latency and any IK/deadline failure. Five episodes establish a smoke result only.
- Model controls: within-backend prediction errors at horizons1/4/8, action shuffling, normalized-zero action predictions, persistence, latent variance/collapse and effective rank where implemented. Raw latent errors are not compared between backends.

Local data and weights stay ignored; version concise reports and artifact hashes. The final manipulation cohort/training seeds/budget are frozen separately only after this development gate.
