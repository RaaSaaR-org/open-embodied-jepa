# Reach pilot v1 — selector correction declared before rerun

This protocol changes checkpoint selection after the v0 validation diagnostics
showed that native JEPA's lowest raw prediction MSE checkpoint was collapsed.
The v0 experiment, selected checkpoint, reports, and subsequent exploratory
control results are retained. The correction is motivated exclusively by
validation collapse evidence; no test or held-out manipulation samples are used.

Data, architecture, initialization, optimizer, and budget remain the same as v0:
`data/reach-pilot-v0`, manifest SHA-256
`9a5256cc2070d2bc8c44f44b741830e428a33682c5baf0719c7973c62231804f`;
80/10/10 frozen session split; seed 0; 500 updates per backend, batch 16,
training horizon 4, CPU four threads, maximum 600 seconds each. Validation uses
the same fixed cohorts of four batches of 16 windows, at step zero and every
50 updates, with horizons 1/4/8 when supported. Both backends use their unchanged
v0 configurations and exactly the same sequence of sampled training windows.

The explicit selector is `--selection noncollapsed_relative`. On the **horizon-4
validation cohort**, a checkpoint is eligible only when both:

- `collapsed_fraction <= 0.05` (fraction of latent dimensions with std < 0.01), and
- `latent_std_mean >= 0.1`.

Among eligible checkpoints minimize
`prediction_mse / max(persistence_mse, 1e-12)`, keeping the earlier checkpoint on
an exact tie. Both numerator and denominator are measured within the same model
and cohort. Log eligibility, rejection reasons, and score for every validation.
No eligible checkpoint means selection failure and no best checkpoint; retain
the latest checkpoint and all diagnostics. A score below one indicates beating
latent persistence on this cohort, not physical success or generalization.

Write new artifacts to `checkpoints/reach-pilot-v1/{native_jepa,leworldmodel}.pt`
and corresponding reports/curves/latest files. Preserve v0 outputs. Compare final
training state tensors and training curves against v0 to verify that only selection
and metadata changed. Report any numerical differences rather than assuming
bitwise replay.

The original `raw_mse` selector remains the default so v0 CLI commands retain
their historical meaning. This v1 is a development experiment, not an independent
confirmation of a hypothesis selected using v0 validation. Downstream reaching
uses the same shared CEM/MPC budgets, five development seeds, goal construction,
and controls from [v0](reach_pilot.md); report failures and make no claim of
Apple→Plate success from reaching outcomes.
