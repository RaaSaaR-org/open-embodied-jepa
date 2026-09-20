# Matched Apple→Plate action branches v1

TASK035 is a prospective data experiment prompted by weak action dependence in
`apple_sensor_diagnostics_results_v1.md` and failed exploratory learned control.
It does not change physics, task thresholds, model architecture, or earlier data.
No collection is authorized before protocol/code review and the coordinator's
source-freeze commit. The proposed resource cap is 600 true-wall seconds including
startup and sealing; stop starting commands at 540 seconds. Kill the worker at
595 seconds, reserving five seconds for supervisor finalization. No extensions/retries.

## Cohort and split inheritance

Use sealed `data/apple-task-v1`, manifest SHA256
`d70edd9daaeec58e80a4e6a743980c049335a03f2b32b7d42bbac5ec5bc579df`.
Select the lexicographically first eight TRAIN episodes whose recorded variant is
nominal, and all three original VAL episodes. VAL contains no nominal parents;
its recorded perturbed continuations remain labeled as such. Do not select parents
by new outcomes. All six phases are used: orient, descend, close, lift, transfer,
release_high. Choose the middle start among starts that keep 16 transitions inside
the recorded phase. If a phase is too short, record an unavailable root without
replacement. This gives 66 planned roots and 528 planned branches.

All siblings and their complete source demonstration retain the original parent's
session and partition. The new immutable corpus also retains the complete original
32 demonstrations with unchanged TRAIN/VAL/TEST membership. TEST payloads are
copied as encoded files without decoding any test images, selecting roots, or
computing test metrics. Historical held-out corpora remain untouched. The new
corpus is explicitly task-specific development, `heldout_combinations=()`.
TRAIN-only normalization is refitted after sealing. Successful TRAIN image goals
remain traceable to their original source episode/frame/image hash.

## Actions and restoration

Each root has eight branches of 16 commands (0.8 seconds at 20 Hz); H8 endpoints
at 0.4 seconds are also retained. Start with the source's recorded requested
command at the matching phase time. Preserve left-arm/grasp channels. The fixed
right-arm alternatives are:

1. Recorded continuation, including any recorded perturbation.
2. Zero six-dimensional arm motion on both arms; retain root's applied grasps.
3. Half the recorded right translation and rotation.
4. Reverse recorded right translation, retain rotation/grasp.
5. Reverse recorded right rotation, retain translation/grasp.
6. Zero right rotation and XY motion; sustained normalized right Z = +0.25.
7. Recorded right motion with fully open requested right grasp (-1).
8. Recorded right motion with fully closed requested right grasp (+1).

Clip requests to the existing normalized contract. Every command passes the
unchanged embodiment; store actual applied actions as canonical labels. Store
requested actions, rejection reasons and phase/root identifiers as metadata.
A rejected command has no invented transition; retain its prior accepted prefix,
including prefixes shorter than H8. Rejected/incomplete branches are denominators,
not successful dynamics examples.

Obtain roots by reset plus deterministic replay of recorded requests, checking
actual actions and sensor state against the source. Clone MuJoCo's complete data,
controller targets, grasp state, and ordered scorer state at each root. Restore
this snapshot for every sibling. Require exactly identical first RGB, proprio,
mask, and timestamp across siblings. Verify a repeated restored continuation has
identical first-step sensors and actual action; otherwise mark root invalid and
skip all its branches. No qpos-only restore, object teleport, weld, or force assist.
Simulator truth is permitted only for reset and unchanged scoring, never for
branch selection, command construction, canonical observations, or model inputs.

## Artifact and failure requirements

Freeze protocol, script, Python source, action conversion, asset manifest, and
parent dataset hashes in a plan before physics. Run from a copied runtime or
reject source drift; recheck inputs after collection. Preserve all 528 planned
branch records, including unavailable roots, replay failures, rejections,
interrupted active branches and unstarted branches. Record actual CPU/true-wall
elapsed time and completion status. Partial artifacts remain unsuitable for
training until explicitly sealed and reviewed; process timeout is nonzero.

Each canonical branch is an ordinary Episode with T+1 measured RGB/proprio/mask
frames and T actual action/raw-action labels. Metadata records parent episode,
parent session, root frame/phase, sibling group, branch definition, replay errors,
requests and ordered stage scores. A separate branch index exposes the grouping;
no model is given privileged simulator fields. Full demonstrations provide goals
and long-horizon trajectory coverage; branches add local causal contrasts.

## Prospective assessment

Report completion/rejection counts per split, phase and branch, replay/restore
agreement, actual action contrast, H8/H16 RGB and measured-proprio divergence,
and unchanged task-stage outcomes. Evaluate model predictions only after a
separate frozen training protocol. The relevant test is within-root action
assignment: compare each observed branch future with predictions using its own
actual action sequence versus siblings' actual sequences, restricting comparisons
to equal complete horizons. Report ties and branch visibility instead of inferring
causality from whole-frame prediction loss. Near-identical observed endpoints
provide little ranking evidence even when requested actions differ.

This collection may expose frequent safety rejection or visually indistinguishable
branches. Preserve those negative outcomes. More epochs or architecture changes
are separate hypotheses, not automatic consequences of collecting these data.

Per-command journals retain an initial sensor frame and each acknowledged applied
command with its resulting sensors, using atomic file replacement. A pending
command is written before execution. A crash may leave uncertain execution beyond
the durable prefix; the supervisor reports that uncertainty and never trains from
an incomplete run. Ordinary branch errors preserve captured prefixes as canonical
episodes where storage remains functional; journals retain prefixes after a
storage failure. No interrupted command is silently converted into a zero action.

Before training, report actual matched-future distances at H8/H16. The declared
visual distinction threshold is whole-frame RGB RMS >= 1/255; also report measured
proprio and applied-action RMS separately. This is an operational screen, not an
object-visibility claim. At least 75% of available roots in EACH split must have
four complete H16 branches and two visually distinct branch pairs; otherwise the
corpus does not pass the action-contrast gate. A future model ranking gate must
compare own-action and wrong-sibling-action predictions at the same initial frame,
report root-grouped uncertainty and ties, and require a positive validation
advantage whose root-bootstrap 95% interval excludes zero. Training budgets and
exact ranking metric must be frozen separately before training; collecting these
data alone never establishes that gate passed.

`training_ready` requires complete collection, successful integrity checks, and
the action-contrast gate in both splits. Low contrast preserves completed attempt
records and yields `completed_not_training_ready` with a nonzero exit status.
