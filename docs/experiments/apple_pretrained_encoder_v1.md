# Apple→Plate pretrained encoder v1: does a frozen, externally pretrained encoder expose the post-look apple beyond its random init? (TASK-063)

**Status: preregistration. No readout has been fitted on any feature this study introduces.** The
only numbers here are TASK-061's and TASK-062's published results and a pre-freeze calibration
(`scripts/calibrate_pretrained_encoder.py`, committed with this document). The calibration fits
**no readout**, and **no target or label** (apple position, expert command, privileged label)
enters any of its statistics. It renders frames, using each root's reset coordinates only to
render them, and it measures encoder activations. **The test split is never decoded. Cohort D
(45000–45007, 45100–45107) and cohort C (45300–45339) are never simulated. `exemption_spent`
(`benchmarks/manifests/apple-policy-diagnostics-v1.json`) stays `false`.**

This is a **perception/representation task**. It tests whether a frozen image encoder makes the
apple **readable**; it tests nothing about control. It trains nothing, collects no corpus and
preregisters no control formulation: CEM over the world-model cost was abandoned at TASK-054 and
the behaviour-cloning line at TASK-057 (`docs/DECISIONS.md`), and both clauses still hold.
**Learned Apple→Plate is still 0 successes.** Nothing here is a control result.

Predecessors:
- [`apple_encoder_study_v1.md`](apple_encoder_study_v1.md) /
  [`_results.md`](apple_encoder_study_v1_results.md) (TASK-062, outcome **O-ENC-ARCH**; its
  abandonment clause fired);
- [`apple_observation_reprobe_v1.md`](apple_observation_reprobe_v1.md) /
  [`_results.md`](apple_observation_reprobe_v1_results.md) (TASK-061, outcome **O-LOOK-RAW**).

Manifest: `benchmarks/manifests/apple-pretrained-encoder-v1.json`.

---

## 1. The question

TASK-062 closed the line "train a LeWM-family encoder on `apple-wide-v1` so that its frozen
features expose the post-look apple". Its O-ENC-ARCH row recommended exactly one next task: **one
preregistered test of an externally pretrained frozen encoder against its own random-init floor,
on this probe, before any corpus.** This is that test.

**Question:** does a frozen encoder that was pretrained elsewhere, and never saw this corpus,
expose the apple in TASK-061's post-look 112 px frame to TASK-061's readouts, under the unchanged
TASK-059 bars, **and** beat a random-init encoder of exactly the same architecture read at the
same point?

**Claim scope, fixed now.** A pass would say: *this* pinned encoder's frozen features, read at
*this* point, make the post-look apple position and the expert's first reset-dependent command
readable by kernel ridge on 190 roots, beyond what its architecture alone gives. It would **not**
say that a world model built on it predicts the apple, that any controller works, or anything
about closed-loop behaviour.

## 2. What is reused unchanged

The probe is TASK-061's L arm, as TASK-062 ran it. `scripts/probe_observation_reprobe.py`,
`src/embodied_jepa/observation_reprobe.py`, `scripts/probe_info_ceiling.py`,
`src/embodied_jepa/info_ceiling.py`, and TASK-062's `src/embodied_jepa/encoder_study.py`,
`scripts/probe_encoder_study.py` and `scripts/calibrate_encoder_study.py` are loaded as they are, and their bytes are pinned (G-hash,
§10).

- **Frames.** TASK-061's `measure` renders them: the same 190 train + val roots, the **renderer
  warm-up** (every renderer renders once and discards the result, because of a first-render
  artifact), the 8-command look and the apple-hidden ablation render. TASK-061's G-render,
  G-expert, G-look and G-prior run unchanged. Only the post-look 112 px frame (`post_112`) and its
  apple-hidden twin are used.
- **Targets.** T1 is the reset apple xy. T2 and T3 are the sign and value of the expert's
  `right_dx` at the post-look state (TASK-061 §4).
- **Readouts, folds and statistics.** Linear and RBF kernel ridge with nested inner CV
  (`info_ceiling`, λ grid 1e-4…1e4). Grouped 10-fold CV with `default_rng(59)` and TASK-061's fold
  hash `1dcc9086…c9b172a`. The same 10 000 paired bootstrap resamples (seed 5901), Wilson
  intervals, and baselines B-occ, B-maj and B-const on the post-look targets. The out-of-fold
  procedure is `probe_info_ceiling.evaluate_source`, exactly as TASK-061 ran L-raw and L-E0.
- **Bars** (TASK-059 §9), on all 190 roots:
  - T1: median ≤ 1.5 cm **and** upper 95 % bound of the ratio to B-occ ≤ 0.6;
  - T2: accuracy ≥ 0.85 **and** Wilson lower bound > B-maj (0.621);
  - T3: upper bound of the MAE ratio to B-const ≤ 0.6.
- **Floor rule** (TASK-059 §9, `info_ceiling.beats_random_floor`, unchanged): the paired bootstrap
  95 % CI of the median T1 error difference (arm − floor) lies below 0, **and** the arm's T2
  accuracy is higher than the floor's.
- **p-values** (TASK-062 §7, `encoder_study.arm_pvalues`, unchanged): p_arm = max(p_T1, p_T2,
  p_T3, p_F); p_arm = 1 if a point condition fails (T1 median > 1.5 cm, T2 accuracy < 0.85, or T2
  accuracy not higher than the floor's).
- **Spurious check** (TASK-061 §7.4, `observation_reprobe.spurious_verdict`): the same
  out-of-fold readouts, applied to the apple-hidden frames featurised by the same frozen encoder.
  If those still beat the prior on all roots, the cue is not the apple. The ablation renderer must
  reproduce the un-ablated frame on 190/190 roots.

**No cross-fitting.** TASK-062 needed two-way cross-fitting because its encoders were trained on
this corpus. The pretrained encoder never saw it (§4.4: released in 2023, from web images; the
corpus was simulated in 2026), and the floor is untrained. Every root is therefore held out from
both encoders by construction, and the single-encoder procedure of TASK-061 applies.

**What changes:** the encoder (§4), its read-out points (§5), the Holm family (§7). The bars, the
floor rule, the readouts and the frames do not change.

## 3. Context (reported, not decisional)

### 3.1 The facts this study starts from

From TASK-062's results (all on the same 190 roots and post-look frames):

| source | what | T1 median (cm) | T2 | meets bars |
|---|---|---|---|---|
| L-raw | raw post-look pixels (ceiling reference) | 0.469 [0.419, 0.528] | 180/190 | yes |
| A-tok | E0 recipe, retrained cross-fitted, final patch tokens | 0.463 [0.422, 0.536] | 178/190 | yes, but not its floor |
| **F-tok** | **seed-0 random init of E0's ViT, final patch tokens** | **0.601 [0.547, 0.673]** | **179/190** | **yes** |
| R0-cls | E0 recipe, retrained cross-fitted, pooled 128-d feature | 1.223 [1.051, 1.402] | 162/190 | no |
| L-E0 | frozen E0, pooled 128-d feature | 1.273 [1.005, 1.462] | 147/190 | no |

**Reported-only context, given no row:** token features of a ViT, **even random-init ones**,
expose the post-look apple on this probe (F-tok). The information is lost at the pooled 128-d
latent. This is why the floor of this study is read at the **same point** as the arm, and why the
token arm's floor is expected to be strong (§6).

### 3.2 Why a pretrained encoder, and what it can and cannot show

TASK-062's in-corpus encoders did not beat their random init. Three things could explain that,
and TASK-062 left them untested: the corpus (170 resets of the expert's own trajectories, the
shared E-distribution confound), capacity (128-d), and input handling (patch 14 at 112 px). An
externally pretrained encoder changes all three at once: it was trained on about 142 M web
images, it is wider (384-d), and it is run at its standard 224 px input. **So a pass would not say
which of the three mattered.** It would say that a frozen encoder with this provenance makes the
apple readable beyond its architecture. That is the owner's precondition for any look-prefix
corpus (TASK-062 §6), and it is all this study tests.

### 3.3 Pre-freeze calibration (label-free; fits nothing)

`scripts/calibrate_pretrained_encoder.py`, artifact
`outputs/task063-pretrained-encoder/calibration-v1.json` (sha256 in the manifest). The encoder,
the input handling, the read-out points and the floor (§4–§6) were fixed in
`src/embodied_jepa/pretrained_encoder.py` **before** the calibration ran, and the module's bytes
did not change afterwards (its hash is pinned). The calibration informs only the budget (§12).
It reports TASK-062's label-free statistics for both encoders; none of them measures
readability (TASK-062 §3.3, caveat).

Facts (from the artifact):
- 190 roots; the post-look frames hash to `bff0fb60…1831`, **identical to TASK-062's
  calibration and run**; the ablation renderer reproduces the post-look frame on 190/190 roots.
- Both weight digests as pinned (§4.2); a second forward pass is bit-identical for both encoders.
- The converted weights equal the original FAIR checkpoint tensor for tensor (§4.2).
- CPU forward pass over 190 frames: about 3 s per encoder per frame set (torch 2.14.0, 6 threads).
  The whole calibration, rendering included, took 68 s.

Label-free statistics (S_apple: the apple's own effect relative to the typical difference between
two roots; S_plate: the same for the plate; effective rank over the 190 post-look frames):

| representation | S_apple | S_plate | effective rank |
|---|---|---|---|
| raw pixels | 0.245 | 1.547 | 18.0 |
| pretrained, patch-embedding tokens | 0.207 | 0.867 | 34.2 |
| pretrained, tokens after block 6 | 0.476 | 1.949 | 20.3 |
| **pretrained, final tokens (P-tok)** | **0.760** | 1.903 | 15.8 |
| **pretrained, CLS (P-cls)** | **0.883** | 2.353 | 11.0 |
| pretrained, mean of final tokens (P-mean) | 0.914 | 2.121 | 10.2 |
| random init, final tokens (R-tok) | 0.205 | 1.695 | 14.8 |
| random init, CLS (R-cls) | 0.258 | 4.298 | 17.5 |
| random init, mean of final tokens (R-mean) | 0.247 | 4.630 | 18.6 |
| *TASK-062, for comparison:* E0 probe feature / E0 final tokens / F-tok | 0.151 / 0.338 / 0.221 | 4.140 / 3.055 / 1.808 | 3.2 / 14.7 / 16.4 |

**Reading (interpretation, not a result, and not a prediction of the run):** in the pretrained
encoder the apple's relative effect grows block by block (0.207 → 0.760 in the tokens) and stays
large in the pooled features, unlike E0, where it shrinks along the head (0.338 → 0.151). The
random init stays flat at raw-pixel level. S_apple is a size of response, not readability: E0's
tokens had S_apple 0.338 and read well, F-tok 0.221 and read almost as well. Only the probe decides.

## 4. The encoder, its provenance and its licence

### 4.1 Choice, and why

**DINOv2 ViT-S/14** (Oquab et al. 2023, arXiv 2304.07193), without registers, in the Hugging Face
transformers conversion `facebook/dinov2-small`. One encoder, one input size, no sweep.

- **Externally pretrained, never on this corpus.** Self-supervised on LVD-142M web images (§4.4).
  No robot or simulator data is disclosed in its training set, and it was released three years
  before `apple-wide-v1` was simulated. It is held out by construction.
- **The closest match to E0's architecture among permissively licensed pretrained encoders.** A
  ViT with a CLS token, patch size 14 (E0's), 12 blocks, width 384 (E0: 4 blocks, width 128). The
  floor is therefore a meaningful comparison with TASK-062's F-tok.
- **Permissive licence for code and weights** (Apache-2.0, §4.3). DINOv3 (custom licence), I-JEPA,
  MAE and V-JEPA checkpoints under CC BY-NC terms, and DINOv2's XRay/Cell variants (non-commercial)
  were not chosen, because AGENTS.md keeps restrictive integrations out of the core path.
- **No new package.** `transformers` 4.57.6 (`Dinov2Model`) and `safetensors` 0.8.0 are already
  locked (the `lewm` extra). A new optional extra `pretrained` names them explicitly (§4.5).
- **Small and fast.** 21 M parameters; the CPU forward on all frames takes seconds (§3.3), so no
  accelerator and no training budget is involved.
- **Without registers.** The original release, and the one the HF `dinov2-small` repository holds.
  The register variant is a different checkpoint; choosing it would have been a second encoder.

### 4.2 Pinned files (G-weights)

`scripts/fetch_dinov2.py` downloads into the git-ignored `third_party/dinov2-small/` and checks
every sha256 before use. The runner reads only these local files.

| file | source | bytes | sha256 |
|---|---|---|---|
| `model.safetensors` | `huggingface.co/facebook/dinov2-small`, revision `ed25f3a31f01632728cabb09d1542f84ab7b0056` | 88 249 960 | `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1` |
| `config.json` | same revision | 547 | `1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d` |
| `README.md` (HF model card) | same revision | 3 033 | `4c20dca454a8e5c670e8de5c7e6040f512aeca5438516f7623eedc4e3b00599c` |
| `official/dinov2_vits14_pretrain.pth` (provenance check only; never loaded by the runner) | `dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth` | 88 283 115 | `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9` |

- The safetensors sha256 equals the Git LFS object id the Hugging Face API reports for that file
  at that revision. The model repository's last commit is dated 2023-09-06.
- FAIR publishes no hash for the original `.pth`; its sha256 is pinned from the first download
  (trust on first use), disclosed.
- **The conversion equals the original.** `fetch_dinov2.py --verify-official` maps the original
  checkpoint's keys onto the HF names (transformers' own conversion mapping; the fused qkv
  projection is split into query, key and value) and compares every tensor exactly: **all 223
  converted tensors equal their source tensors bit for bit, and all 175 original tensors are
  used.** The calibration artifact records the same check.
- **Loaded digest** (the `weights_digest` rule of TASK-062: sha256 over the state dict in key
  order): pretrained `3a697b87…2af27`; seed-0 random init `3d305f9c…7c9db` (full values in the
  manifest).

### 4.3 Licence (checked 2026-09-26)

- **Upstream code and weights: Apache-2.0.**
  - The upstream `README.md`, "License" section, at `facebookresearch/dinov2` commit
    `7764ea0f912e53c92e82eb78a2a1631e92725fc8`: "DINOv2 code and model weights are released under
    the Apache License 2.0."
  - `MODEL_CARD.md` at the same commit: "License: Apache License 2.0".
  - The repository was relicensed from CC-BY-NC to Apache-2.0 on 2023-08-31 (commit
    `81b2b6419385a321287de91e00282ef7cbd26f94`, "Update code and models license from CC-BY-NC to
    Apache 2.0"). The HF model repository's pinned revision is dated after that, 2023-09-06.
- **The HF conversion:** the model card at the pinned revision declares `license: apache-2.0` and
  states that the Hugging Face team, not the DINOv2 authors, wrote the card. The weights are shown
  to be the original ones (§4.2), so the upstream licence statement covers them.
- **Not covered, not used:** the XRay-DINO weights (FAIR Noncommercial Research License) and the
  Cell-DINO models, which have their own licence files in the same repository; DINOv3.
- **Obligations.** Apache-2.0 §4: keep the licence and any NOTICE when redistributing. Upstream
  has no NOTICE file. **This project does not redistribute the weights:** they are fetched into the
  git-ignored `third_party/` and never committed or published. No upstream source is copied;
  `transformers`' own `Dinov2Model` (Apache-2.0) builds the architecture.

### 4.4 Training data, as far as it is disclosed

- **LVD-142M** (MODEL_CARD.md, "Training data: LVD-142M (see paper)"; paper §3 and appendix A).
  About 142 M images, assembled by retrieving, from a pool of about 1.2 B unique images crawled from
  "a publicly available repository of crawled web data", images close to those of several curated
  datasets (the paper names ImageNet-22k, the ImageNet-1k train split and Google Landmarks among
  them). The pool was filtered: unsafe or restricted URLs discarded, PCA-hash deduplication, NSFW
  filtering, identifiable faces blurred, and copy-detection deduplication against evaluation sets.
- **LVD-142M is not released.** Its exact contents cannot be audited here.
- **ViT-S/14 is distilled** from the DINOv2 ViT-g/14 with the standard DINOv2 objective (DINO and
  iBOT losses, KoLeo), the ViT-g frozen as teacher (MODEL_CARD.md).
- **Disclosed limitation** (MODEL_CARD.md): biases toward rich households from Western countries.
  Nothing in this study depends on people or households.
- **No robot, MuJoCo or G1 imagery is disclosed**, and the corpus postdates the weights, so no
  frame of `apple-wide-v1` can be in the training data.

### 4.5 Dependencies

- New optional extra `pretrained = ["torch>=2.7,<3", "transformers>=4.50,<5", "safetensors>=0.4,<1"]`
  in `pyproject.toml`. It adds **no package** to `uv.lock`: all three are already locked (torch
  2.14.0, transformers 4.57.6, safetensors 0.8.0), through `learning` and `lewm`.
- `src/embodied_jepa/pretrained_encoder.py` imports torch, transformers and safetensors only
  inside functions; `import embodied_jepa` stays torch-free (the core CI check).
- `docs/DEPENDENCIES.md` gains the entry for the weights.

## 5. Arms, floors and reported-only sources

### 5.1 Input handling (both encoders, identical)

The uint8 post-look 112 px frame, scaled to [0, 1], resized to **224 × 224** by bicubic
interpolation (`torch.nn.functional.interpolate`, `align_corners=False`, `antialias=False`, **no
crop**), then normalised with the ImageNet mean and std. 224 px is DINOv2's standard evaluation
resolution; patch 14 gives a 16 × 16 grid, and the position embeddings are interpolated from the
37 × 37 training grid by transformers' own bicubic rule. The upsampling adds no information to the
frame: a median apple of about 12 px at 112 px covers about 1.7 patches at 224 px. The HF image
processor is not used, because its centre crop would cut the frame's borders.

CPU, float32, eager attention, `eval()`, `no_grad`, batches of 16. The features are cast to
float64 for the readouts.

### 5.2 The two decisional arms (Holm order)

| arm | encoder | read-out point | dim | floor |
|---|---|---|---|---|
| **P-cls** | pretrained, frozen | the **CLS token after the final LayerNorm** (`pooler_output`), the encoder's pooled feature | 384 | **R-cls** |
| **P-tok** | pretrained, frozen | the **256 final patch tokens** after the final LayerNorm (`last_hidden_state[:, 1:]`), flattened, no pooling | 98 304 | **R-tok** |

- **P-cls first.** A pooled feature is the drop-in for a world model's latent; that is what LeWM
  consumes today. The Holm tie-break follows this order.
- **The floors.** R-cls and R-tok are the **same architecture** (the same pinned `config.json`)
  at transformers' own initialisation under `torch.manual_seed(0)`, read at the same point with
  the same input handling. Their digest is pinned (§4.2). The floor is never trained.

### 5.3 Reported-only sources (none of them decides anything)

| source | what | why |
|---|---|---|
| **L-raw, L-E0, L-random** | TASK-061's anchors, recomputed through TASK-061's code | G-anchor (§9); L-raw is the ceiling reference |
| **F-tok, E0-tok** | TASK-062's random-init and E0 patch tokens, recomputed through TASK-062's stage code | G-anchor (§9); F-tok is the context fact of §3.1 |
| **P-mean, R-mean** | the mean of the 256 final patch tokens (384-d), pretrained and floor | a second pooling, so that a P-cls failure can be told apart from a failure of pooling as such. **It cannot change a row.** |

**Paired comparisons reported** (median T1 difference with its paired CI; McNemar on T2): each arm
with its floor, with L-raw and with F-tok; P-cls with L-E0; P-tok with E0-tok; R-tok with F-tok;
P-mean with P-cls and with R-mean.

## 6. The floors and the ceiling

- **An arm must beat its own floor** (`beats_random_floor`, all 190 roots), not merely the prior.
- **R-tok is expected to be strong.** TASK-062's F-tok (a random ViT of E0's architecture, read at
  its tokens) met every bar at 0.601 cm and 179/190. R-tok is a wider, deeper random ViT at a
  larger input. **Beating it may be hard, and that is the point:** P-tok passes only if pretraining
  adds something beyond the architecture.
- **The T2 condition is strict at the ceiling** (TASK-062 §8 caveat). If R-tok reaches k/190, P-tok
  needs at least k + 1. L-raw's own count is 180/190. Stated in advance; not re-litigated after.
- **Ceiling.** L-raw (0.469 cm, 180/190) is the reference. Being below it is not a failure. It is
  reported for every arm as the paired difference arm − L-raw.

## 7. Pass rule and multiplicity

**Arm states.**
- **Not evaluated (mechanical failure):** the arm's or its floor's featurisation plus readouts
  exceed the per-arm cap (§12), or produce a non-finite feature. It cannot pass. This says nothing
  about the encoder.
- **Evaluated:** otherwise. (No collapse gate: nothing is trained. `_statistics` of every read-out
  on the 190 post-look frames is reported.)

An arm **passes** only if all of the following hold:
1. **Evaluated.**
2. **Succeeds** on all 190 roots under the unchanged bars (`info_ceiling.evaluate`).
3. **Beats its floor** (`beats_random_floor`, all 190 roots).
4. **Holm rejection** at a family-wise one-sided α = 0.025 over the **two** arms. p_arm as in §2.
   The thresholds are 0.0125 and 0.025; ties go to the listed order P-cls, P-tok. A
   not-evaluated arm has p_arm = 1. Conditions 2 and 3 are kept alongside Holm.
5. **Not spurious** (§2), with the renderer reproducing 190/190.

**Qualifiers, not rows** (TASK-062's `encoder_study.qualifier`): "unadjusted only; not a pass",
"meets the bars, not the floor", "partial information".

## 8. Diagnostics (reported only)

- `_statistics` (std mean, collapsed fraction, effective rank) of every read-out on the 190
  post-look frames, both encoders.
- The label-free table of §3.3, recomputed in the run on the same frames.
- The sha256 of every source's post-look feature matrix, and a second forward pass of the
  pretrained encoder on the post-look frames, compared bit for bit (a check, §10).

## 9. Reproduction anchors (G-anchor)

Before any new feature is read out, the run recomputes:
- **L-raw, L-E0 and L-random** through TASK-061's code, compared with TASK-061 run-1's report
  (`outputs/task061-observation-reprobe/run-1/report.json`, sha256 `289470f4…fa88`);
- **F-tok and E0-tok** through TASK-062's stage code, compared with TASK-062 run-1's report
  (`outputs/task062-encoder-study/run-1/report.json`, sha256 `fc9d5e52…962f`).

Compared are every field of `results[name]` that `evaluate_source` and the stratum masks produce
(the primary estimate on every stratum, the secondary estimate, the per-fold selections) and
every root's out-of-fold xy, dx and dy.

**Handling, fixed now (TASK-062 §9, unchanged):**
- **Bit-exact** (max |Δ| = 0): the anchors hold.
- **Any T2 count, pass/fail flag, `succeeds`/`beats_prior` reading or selection differs, or any
  numeric max |Δ| > 1e-6:** G-anchor fails and the run is **V**. The owner is told before any
  repeat.
- **0 < max |Δ| ≤ 1e-6 with every count, flag and selection identical:** the run continues, the
  difference is a reproduction caveat, it is escalated to the owner before the results document is
  written, and no row changes.

The run uses the machine TASK-061 and TASK-062 ran on (kernel-ridge values are not
bit-reproducible across BLAS builds).

## 10. Guards: any failure voids the run

| guard | condition |
|---|---|
| **G-hash** | Every file in the manifest's `hashes` matches: TASK-059's and TASK-061's runners and modules; TASK-062's module, runner (`scripts/probe_encoder_study.py`, for its E0 loader and stage code) and calibration script; this study's module, fetch and calibration scripts and calibration artifact; TASK-061's manifest and run-1 report; TASK-062's manifest and run-1 report; the E0 checkpoint and the task056 a1/a2 checkpoints; the configs and the dataset manifest; `models/base.py`, `models/lewm.py`, `models/readout.py`, `policy.py`. Episode files are checked against the dataset manifest before decoding (TASK-061's reader). The tree is clean. |
| **G-weights** | The three pinned DINOv2 files match (§4.2). The pretrained state dict loads `strict`, and its digest and the seed-0 floor's digest equal the pinned ones. No network access is made: the files are local. |
| **G-split** | TASK-061's: exactly the 190 train + val roots and the fold hash. Episodes are decoded only by TASK-061's reader, for the roots' stored reset frames (G-render); nothing is trained. |
| **G-render, G-expert, G-look, G-prior** | TASK-061's, unchanged, run by TASK-061's `measure` and checks. |
| **G-frames** | The post-look frames hash to TASK-062's value (`bff0fb60…`, the calibration's too). The ablation renderer reproduces 190/190 (else the spurious check is unavailable and every arm counts as spurious, TASK-061's rule). |
| **G-repro** | A second forward pass of the pretrained encoder on the post-look frames is bit-identical to the first. |
| **G-anchor** | §9. |

**Not guards.** A per-arm cap or a non-finite feature makes the arm **not evaluated**; it never
voids the run. Any other exception is a crash, and a crash is V.

**PR 2's tests** must exercise, in both directions: every guard (including G-anchor's three
branches for both anchor sets, and G-weights on a wrong file, a wrong digest and a missing file);
the Holm step-down over two arms; the floor condition and p_F through `arm_pvalues`; the spurious
check; both arm states and every outcome row; a crash that still writes `report.json` with outcome
V and a `void_reason`; non-finite values written as `null` and listed in `non_finite_fields`.

## 11. Pre-declared outcomes (first matching row)

| row | condition | reading | next task implied (a recommendation; the owner chooses) |
|---|---|---|---|
| **V** | a guard fails (§10), or the run stops before writing a complete report (crash, global wall cap) | nothing is read | one repeat (§13) |
| **O-PT-POOLED** | P-cls passes | The pinned pretrained encoder's frozen **pooled** feature exposes the post-look apple, beyond its random init, on this probe. | The owner's corpus precondition (TASK-062 §6) is met. **Recommended: preregister the look-prefix 112 px corpus with this frozen encoder as the candidate image encoder.** Any world-model or control use of it needs its own preregistration; none is made here. |
| **O-PT-TOKENS** | P-tok passes, P-cls does not | The frozen **tokens** carry the apple beyond random init; the pooled feature does not, on this probe. | As above, with the token grid (or a pooling that keeps it) as the latent, separately preregistered. The pooled feature is not used for the apple. |
| **O-PT-INCOMPLETE** | no arm passes, and at least one arm was **not evaluated** (§7) | The unevaluated arm is untested. | **The abandonment clause does not fire.** The owner decides whether a disclosed amendment re-runs it. |
| **O-PT-FLOOR** | no arm passes, both were evaluated, and at least one arm **succeeds** (bars) and is not spurious | The pretrained features expose the apple, but not beyond the architecture's random init (or not under Holm). | **The abandonment clause fires** (below). |
| **O-PT-NONE** | no arm passes, both were evaluated, and no arm succeeds without being spurious | Neither read-out of the pretrained encoder exposes the apple under the bars. | **The abandonment clause fires** (below). |

**INCONCLUSIVE (task-level, not a row of a run).** A second V closes TASK-063 as INCONCLUSIVE
(§13). Nothing is read; the abandonment clause does not fire; the owner decides what follows.

**Abandonment clause (for O-PT-FLOOR and O-PT-NONE).**
- **What closes.** The frozen-encoder route to the post-look apple: *"find a frozen image encoder
  whose features expose the post-look apple in the 112 px onboard frame beyond its random-init
  floor"*. No further pretrained encoder, model size, input resolution, layer or read-out point is
  preregistered on this probe without new evidence of a different kind. Together with TASK-062
  §10, **no look-prefix corpus is collected.**
- **What does not close:** the LeWM backend, the encoder as a component, and the product goal.
- **Context, recorded here and not preregistered as a follow-up:** if this test fails, the
  remaining route on record is the observation change TASK-061 already validated, the fixed
  `overview` camera (O-raw: 0.183 cm [0.168, 0.207], 185/190, reported in TASK-061). That is a
  **hardware/workspace change on the robot**, not a model change. Whether to take it is the owner's
  decision, and it would need its own preregistration.

**Always reported, whatever the row:** every arm's numbers, p-values, Holm step, floor comparison,
spurious check and state; every reported-only source on every stratum and its secondary
estimate; the paired comparisons of §5.3; the rows that also match further down.

**In every outcome:** nothing is refitted, re-thresholded or re-chosen after the numbers are seen.
Learned Apple→Plate stays at 0 successes, and `exemption_spent` stays `false`. The executing agent
recommends a next task and does not choose it.

**Stated in advance.**
- **O-PT-FLOOR and O-PT-NONE are live possibilities.** R-tok may succeed on its own, as F-tok did.
- A pass would not identify whether pretraining data, width or input size mattered (§3.2).
- There is one frozen encoder and one floor seed. Floor-seed variance is not measured.

## 12. Budget, device, seeds, recording

- **Device: CPU for everything.** Nothing is trained, so **MPS is not used**: the frozen forward
  pass runs on CPU in float32 so that features are bit-reproducible (G-repro, G-anchor), as
  TASK-061's and TASK-062's features were. The runner refuses any other device.
- **Caps:**
  - **Per arm: 1800 s** for the arm's and its floor's featurisation (post-look and apple-hidden
    frames) plus their nested-CV readouts. Exceeding it makes the arm not evaluated.
  - **Global wall cap: 7200 s** for the whole run. Exceeding it is V.
  - Expected (§3.3 and TASK-062's run): a few minutes for frames and anchors, seconds for each
    forward pass, well under 30 min in total.
- **Seeds:** floor initialisation `torch.manual_seed(0)` in a forked RNG; folds 59; inner
  5900 + k; bootstrap 5901. Nothing else is random.
- **Recorded:** the code revision (clean tree required) and every hash; the weight files' sha256
  and both digests; the environment (torch, transformers, threads); per-root out-of-fold
  predictions for every source and selections per fold; p-values, Holm steps, floor comparisons
  and spurious checks; the anchor comparisons; per-arm and total elapsed time; peak RSS.
- **Output:** `outputs/task063-pretrained-encoder/run-1/report.json`. The runner refuses to
  overwrite. Nothing under `data/` or `checkpoints/` is written or modified.

## 13. Void rule

- A run that stops early for any reason (a guard, a crash, the global wall cap) is **V**, with a
  `void_reason`. Nothing in it is read.
- **Exactly one from-scratch repeat** is allowed, into `run-2`: the same seeds, caps and device,
  and a clean tree.
- **A second void closes TASK-063 as INCONCLUSIVE.**
- A fix between the runs is limited to runner mechanics. It goes through a reviewed PR and a fresh
  pre-run GO, and it is disclosed in the results.
- Any owner ruling is recorded with a UTC timestamp before the run it affects.

## 14. Pre-freeze calibration and fetch (fit nothing)

```
uv run --no-sync python scripts/fetch_dinov2.py --verify-official
uv run --no-sync python scripts/calibrate_pretrained_encoder.py \
    --output outputs/task063-pretrained-encoder/calibration-v1.json
```

- **What the calibration reads:** the 190 roots of the frozen plan, checked against the dataset's
  splits; their post-look frames and apple-/plate-hidden twins, rendered through TASK-062's
  calibration renderer (warm-up included); the pinned weights.
- **What it does not read:** no target, no expert command and no label sidecar; no episode is
  decoded; it fits nothing.

## 15. Not done (declared)

- No encoder is trained or fine-tuned; no controller is trained or evaluated; no closed-loop
  attempt is run; no corpus is collected and no dataset is written.
- No cohort is opened. The test split is not decoded, and cohorts C and D are never simulated.
  `DatasetStore.verify`, reached through the E0 loader, reads episode bytes only to check their
  sha256, as in TASK-059, TASK-061 and TASK-062.
- No change is made to `models/`, `policy.py`, `simulation.py`, `info_ceiling.py`,
  `observation_reprobe.py`, `encoder_study.py`, TASK-061's or TASK-062's runners, or any
  checkpoint.
- Only one pretrained encoder is tested. Other families (CLIP, SigLIP, MAE, DINOv3, V-JEPA),
  sizes, the register variant, other resolutions and intermediate layers are not tested; the
  abandonment clause covers them.
- The overview camera is not re-tested; it is context (§11).

## 16. Process

1. **PR 1:** this document, the manifest, the encoder module, the fetch and calibration scripts,
   the dependency entry, the optional extra and the task card. It merges on an independent
   reviewer's **reported** APPROVE, which explicitly covers the licence and provenance review of
   §4, and green CI.
2. **PR 2:** the runner, the decision code and the tests of §10. It merges on a reported APPROVE
   and green CI.
3. **The gated run** starts only on the pre-run reviewer's **reported** verdict, delivered as a
   message, never on a review file read from disk.
4. **PR 3:** the results document, the results manifest, a summarize script and the recommended
   next task; a reviewer checks every restated number against `report.json` with a script. The
   card goes to done.
5. **One task, one agent.** A protocol defect found after the freeze is escalated to the task owner
   and fixed only through a disclosed amendment.
