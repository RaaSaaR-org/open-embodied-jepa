# TASK041 prospective goal-alignment screen

Prospective protocol; no fitting, learned inference or physics has been performed for this specification. Freeze this protocol and its implementation before either candidate sees VAL images. Planning-time metadata/state/access inspections are disclosed in `apple_goal_alignment_data_audit.md`. This replaces the earlier suggestion that every encoder must beat nearest-neighbor retrieval: retrieval is itself a valid image-only encoder candidate.

## Question and scope

Can an image-only encoder infer useful right-arm goal configuration on held-out parent sessions, and can the frozen world model use that information to rank action futures better than its pixel cost? This is an arm-alignment feasibility screen, not a complete apple/plate goal representation or a physical-control acceptance test. Goal inference receives RGB only. Measured goal joints are offline labels, never supplied to the encoder, dynamics or a robot planner. No goal velocity, phase label, simulator pose, scorer state, TEST payload or final-cohort observation is a model input.

Frozen corpus: `apple-branches-v1`, manifest SHA `6e9a5bcb38a42a27ce1118e102865db985e8e490f37d10de0075c437676f0331`. Frozen sensor checkpoint SHA `0192b99a60abf1d426d127505680570f238401b5a80f6270dd4859d6e64aa5a5`. Its existing TRAIN-only RGB and state normalization is reused without fitting new scales on VAL. Freeze source/configuration/schema/protocol/checkpoint/corpus hashes before launch and verify afterward.

Seven labels, resolved by exact schema names and asserted to be positions in radians: `right_shoulder_pitch_joint.position`, `right_shoulder_roll_joint.position`, `right_shoulder_yaw_joint.position`, `right_elbow_joint.position`, `right_wrist_roll_joint.position`, `right_wrist_pitch_joint.position`, `right_wrist_yaw_joint.position`. Current schema indices are 22–28; do not silently rely on positions if names disagree. Let y be these measured positions normalized with the frozen checkpoint's corresponding mean/scale. Predicted y comes directly from the same seven positions of the frozen recursive sensor prediction.

## Deterministic samples and weights

Preserve exact phase names: orient, descend, close, lift, transfer, release_high. For each of 26 TRAIN and 3 VAL original episodes, choose four pre-action observation indices per phase using `floor(linspace(0,n-1,4))` over sorted indices bearing that phase label. This gives 624 TRAIN/72 VAL original samples. Do not success-filter: failed TRAIN episode 42003 stays included. Its additional lower_open phase is outside the declared six-phase screen, remains in the corpus, and is explicitly reported as unsampled.

Add observations at H8 and H16 of every corresponding branch episode:384 TRAIN branches give 768 samples;144 VAL branches give 288. Expected totals: 1,392 TRAIN and 360 VAL records. A sample key is `(episode_id, frame_index)`; retain the declared records, hashes, parent/session, source group, phase and horizon. Do not silently replace missing/invalid samples or deduplicate by image similarity. Root construction must verify all sibling starting RGB/state/mask/time and relative timestamps match, and actions are actual applied labels. All seven state masks must be valid.

Weights: give each original parent equal mass; within a parent give available source groups (original versus branch) equal mass; within each group give its selected records equal mass. Every declared phase contributes four original frames or sixteen branch endpoint frames where that source exists, so the fixed sample ledger is also phase-balanced. Do not alter weights or fill gaps if sample validation fails. Use these same weights for TRAIN sampling/constant baseline and VAL encoder-error reporting. No frame from a sibling family crosses its original parent/session split. Report expected/actual coverage, including omissions; an incomplete declared dataset is not a silently altered experiment.

## Two encoders and one constant baseline

Features x are the frozen normalized 24×24 RGB grid (1,728 values); no proprio input.

- **NN:** exact 1-nearest TRAIN-bank x in squared Euclidean feature distance; return that record's y. Resolve exact ties lexicographically by sample key. No VAL neighbor, interpolation tuning or per-phase retrieval restriction. The index uses all 1,392 declared TRAIN records, and is frozen before VAL decoding.
- **Neural:** MLP 1728→128→64→14, SiLU hidden activations; outputs seven means and seven log variances clipped to [-6,3]. Gaussian diagonal NLL averaged over seven fields: `0.5 * mean(exp(-logvar)*(mean-y)^2 + logvar)`. AdamW lr 1e-3, weight_decay 1e-4, seed 0, batch 128 sampled with replacement using the declared weights, exactly 2,000 updates. Use final checkpoint only; no VAL selection or architecture sweep.
- **Constant:** the weighted TRAIN mean of y. This baseline is used for encoder error, not presented as a viable world-model controller.

Only encoder means enter candidate costs. Uncertainty must not downweight costs or exclude difficult cases in this screen. Keep NN available even if the neural attempt is incomplete or fails; failure to replace NN is not failure of the representation route.

## Encoder error and ambiguity

For each VAL record compute normalized seven-position MSE, then aggregate using the declared parent/phase/source weights: E_f for each encoder and E_mean for the constant. Report each parent's error, each joint's error in radians, and `R_f = 1 - E_f/E_mean`. Zero constant error makes this ratio unavailable, not infinity or a pass. Compare neural versus NN descriptively and under the replacement rule below.

For each same-root H16 sibling pair report normalized arm RMS separation and raw resized-RGB RMS separation on [0,1]. Label an ambiguity example when arm RMS≥0.1 normalized units but RGB RMS<1/255; never remove these examples from the ranking cohort merely because they are visually difficult. Also report the complementary limitation: similar arm configuration can accompany different object/hand outcomes. Stored scoring labels may illustrate that limitation offline but must never train or select either arm encoder. One verified VAL release_high pair already has maximum seven-arm difference 0.007933 rad yet an apple-height difference 0.1486 m; this screen cannot certify object state.

Neural uncertainty: report per-joint empirical coverage of mean±1.644854*std (nominal 90% marginal interval), widths and parent breakdown. NN uncertainty: report normalized-y RMS spread among the eight nearest TRAIN records as an uncalibrated confidence heuristic. For both, report parent-balanced error at 100%, 75%, 50% retained coverage ordered by uncertainty, with lexicographic ties. These are diagnostics only; all gates use 100% declared VAL coverage. Do not call NN spread a calibrated probability.

## Primary goal-ranking test: independent goals, not just own-future matching

For each of 18 VAL roots, make ONE frozen H16 prediction for each of its eight recorded applied-action sequences from the common measured starting observation. Reuse those predictions, including H8 for secondary reporting. No optimization, refitting or physics occurs.

For a root in phase p and parent a, use the four sampled ORIGINAL goal images in phase p from EACH of the other two VAL parents: eight independent goals per root, 144 root-goal cases. They are held-out image queries; corresponding measured y_g is an evaluation label only. This cross-parent goal set avoids equating success with identifying a branch's own endpoint image.

For candidate branch i, define:

- label distance `D_i = mean((y_i_measured - y_g_measured)^2)`;
- measured-state encoder cost `C_i_meas(f) = mean((y_i_measured - f(goal_RGB))^2)`;
- predicted-state encoder cost `C_i_pred(f) = mean((y_i_predicted - f(goal_RGB))^2)`;
- frozen pixel comparator: original sensor-model image distance between predicted visual endpoint and goal image; also report its measured-image counterpart.

For every unordered candidate pair i,j, include a ranking comparison iff measured normalized arm RMS separation≥0.1 and `abs(D_i-D_j) > 1e-6 + 1e-4*max(1,D_i,D_j)`. Excluded ties/low-separation pairs remain counted with reasons. The correct order is the lower D. Score each method 1 for correct order, 0 for reverse, 0.5 for a predicted tie using tolerance `1e-10 + 1e-6*max(abs(C_i),abs(C_j))`. Pixel and encoder methods use exactly the same eligible comparisons. No pixel-visibility filter is allowed for the primary test.

Aggregate eligible pairs equally within each goal, goals equally within each root, roots equally within each original parent, then the three parents equally. Report measured and predicted ranking separately, plus `Delta_f = A_pred(f)-A_pred(pixel)`. H16 is primary; H8 and matched sibling-endpoint assignment are secondary and cannot rescue a failed primary gate. Report every phase/root/goal/pair coverage count. Require all three parents with at least two informative roots each before issuing a primary pass/fail; otherwise classify the screen as insufficient coverage. Do not claim coverage of a phase with no eligible comparison.

Use 2,000 fixed-seed 0 parent-cluster bootstrap resamples of the three parent summaries for 95% percentile intervals of R_f, A_pred and paired Delta_f; resample the same parent indices for method differences. Publish all three individual-parent values because three clusters give limited uncertainty resolution. No pair-wise independence assumption.

## Compact prospective decision rules

Apply independently to NN and neural:

1. Encoder error reduction versus TRAIN mean `R_f >= 0.20`.
2. H16 predicted goal-ranking improvement `Delta_f >= 0.05` versus frozen pixel cost, with strictly positive improvement on EACH of the three VAL parents. Report absolute ranking and bootstrap intervals; they are not additional gates. This screening threshold deliberately requires a nontrivial aggregate improvement without letting one parent offset deterioration on another, but it is not a manipulation-quality guarantee.

Both require complete declared sample handling and the stated informative-root coverage. A candidate failing its numerical screen is not promoted; insufficient coverage or an incomplete compute attempt is explicitly unverified. Measured-state ranking is an attribution diagnostic: a measured/predicted gap implicates using the frozen dynamics outputs, whereas poor measured ranking implicates goal inference/cost alignment. Neither identifies a unique causal mechanism.

NN passing is a viable retrieval-based goal representation even if neural fails. Neural is a preferred **replacement for NN** only when it passes its own route screen AND has strictly lower full-cohort encoder error AND strictly higher primary predicted ranking than NN. Otherwise retain NN or describe the tradeoff; do not automatically reject the whole route. If neither candidate passes, stop this route before control. No result here authorizes arm-only control deployment, threshold changes or a physical-MVP claim.

## Resource and artifact contract

One registered run with three separately supervised stages, CPU4, no dependency changes. Preparation ≤120 true-wall seconds includes imports, input verification, selected-frame decoding, feature/bank construction and serialization. Freeze and hash the TRAIN bank before decoding any VAL payload; store TRAIN and VAL caches separately. Neural fitting ≤300 true-wall seconds includes imports, TRAIN-cache loading, exactly 2,000 updates, hashes and serialization. Fitting reads only the TRAIN cache. Offline evaluation ≤60 true-wall seconds includes imports, frozen prediction, NN queries, metrics and reporting. Reserve the final 5 seconds of each stage. Retain partial logs and bank on timeout; only a completed 2,000-update head is eligible, while a complete NN bank remains independently assessable. The total authorized stage allocation is 480 wall seconds; report each stage and overall supervisor time explicitly. No retries/extensions, TEST decoding or simulator execution. The parent supervises elapsed `max(wall,monotonic)` and preserves all planned candidates/roots and missing outcomes. Coordinator may tighten CPU/memory limits before freeze; do not alter these wall limits after results.

Artifacts: registration/config/source/package/device/thread hashes; exact TRAIN/VAL sample ledger and weights; NN bank/features/labels digest; final neural checkpoint plus completion status; unchanged world-model checkpoint identity; per-record errors/uncertainties; per-root/per-goal/pair costs, eligibility and rankings; parent summaries/bootstrap seed; all missing/tied/ambiguous counts; resource usage and post-run input verification. No hidden goal-joint array may cross the encoder input boundary. This protocol requires explicit reviewed implementation and coordinator authorization before execution.
