"""Frozen image/pose goal-cost composition; child dynamics and planner stay unchanged."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from embodied_jepa.contracts import EE_DELTA_GRASP_V0, Capabilities, ContractError, StateSchema
from embodied_jepa.models.sensor import SensorWorldModel

FIELDS = tuple(
    f"{name}_joint.position"
    for name in (
        "right_shoulder_pitch",
        "right_shoulder_roll",
        "right_shoulder_yaw",
        "right_elbow",
        "right_wrist_roll",
        "right_wrist_pitch",
        "right_wrist_yaw",
    )
)
RECIPE = "train_parent_median_stride28_v1"
WEIGHTS = {"visual": 0.5, "pose": 0.5}
PROVENANCE_ROLES = (
    "dataset_manifest",
    "goal_registration",
    "goal_fit_report",
    "goal_fit_seal",
    "goal_train_samples",
    "goal_source",
    "calibration_ledger",
)
ROLES = ("sensor", "goal_head", "calibration", *PROVENANCE_ROLES)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def field_indices(schema):
    if any(schema.names.count(name) != 1 for name in FIELDS):
        raise ContractError("aligned goal requires seven unique named right-arm positions")
    indices = tuple(schema.names.index(name) for name in FIELDS)
    if any(schema.units[i] != "rad" for i in indices):
        raise ContractError("aligned right-arm positions must have radian units")
    return indices


def hybrid_cost(visual_mse, pose_mse, scales):
    """Same fixed formula for model-owned cost and saved-array evaluation."""
    if set(scales) != {"visual", "pose"} or any(
        not np.isfinite(v) or v <= 0 for v in scales.values()
    ):
        raise ContractError("both calibrated scales must be finite and positive")
    visual = np.asarray(visual_mse, dtype=np.float64)
    pose = np.asarray(pose_mse, dtype=np.float64)
    if visual.shape != pose.shape or any(
        not np.isfinite(x).all() or (x < 0).any() for x in (visual, pose)
    ):
        raise ContractError("hybrid component costs must have identical finite nonnegative shapes")
    result = (0.5 * visual / scales["visual"] + 0.5 * pose / scales["pose"]).astype(np.float32)
    if not np.isfinite(result).all():
        raise ContractError("nonfinite calibrated cost")
    return result


def make_head(seed=0):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return nn.Sequential(
            nn.Linear(1728, 128), nn.SiLU(), nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 14)
        )


@dataclass(frozen=True)
class AlignedLatent:
    child: object
    owner: object


@dataclass(frozen=True)
class AlignedGoal:
    visual: object
    pose: torch.Tensor
    owner: object


class AlignedSensorWorldModel:
    backend = "aligned_sensor_wm_v1"

    def __init__(self, state_schema, device="cpu", seed=0, config=None, metadata=None):
        if not isinstance(state_schema, StateSchema):
            raise ContractError("named state schema required")
        if device != "cpu":
            raise ContractError("aligned composition is validated on CPU only")
        if type(seed) is not int or seed < 0:
            raise ContractError("nonnegative integer seed required")
        self.indices = field_indices(state_schema)
        self.state_schema = state_schema
        self.device_name = device
        self.seed = seed
        config = copy.deepcopy(config or {})
        if set(config) - {"sensor_config", "weights", "version"}:
            raise ContractError("unknown aligned config")
        self.config = {"sensor_config": {}, "weights": WEIGHTS.copy(), "version": 1} | config
        if (
            self.config["weights"] != WEIGHTS
            or self.config["version"] != 1
            or not isinstance(self.config["sensor_config"], dict)
        ):
            raise ContractError("aligned version1 fixes equal visual/pose weights")
        self.metadata = copy.deepcopy(metadata or {})
        json.dumps(self.metadata, allow_nan=False)
        self._owner = object()
        self._sensor = None
        self._head = None
        self._envelope = None
        self._members = None

    @property
    def implementation_sha256(self):
        return sha256(__file__)

    @property
    def capabilities(self):
        return Capabilities(
            EE_DELTA_GRASP_V0,
            self.state_schema,
            self.config["sensor_config"].get(
                "max_horizon", SensorWorldModel.defaults["max_horizon"]
            ),
            supported_devices=("cpu",),
        )

    def _ready(self):
        if self._sensor is None:
            raise ContractError("load a verified aligned bundle before inference")

    def _state(self, value, rank):
        self._ready()
        if not isinstance(value, AlignedLatent) or value.owner is not self._owner:
            raise ContractError("latent belongs to another aligned instance or checkpoint")
        return self._sensor.check_latent(value.child, rank)

    def _goal(self, value):
        self._ready()
        if not isinstance(value, AlignedGoal) or value.owner is not self._owner:
            raise ContractError("goal belongs to another aligned instance or checkpoint")
        x = self._sensor.check_latent(value.visual, 2)
        if value.pose.shape != (len(x), 7) or not torch.isfinite(value.pose).all():
            raise ContractError("invalid aligned image-goal pose estimate")
        return x

    @torch.no_grad()
    def encode(self, observation, robot_state):
        self._ready()
        return AlignedLatent(self._sensor.encode(observation, robot_state), self._owner)

    @torch.no_grad()
    def encode_goal(self, goal):
        self._ready()
        visual = self._sensor.encode_goal(goal)
        pose = self._head(visual.values)[:, :7]
        if not torch.isfinite(pose).all():
            raise ContractError("nonfinite image-only goal estimate")
        return AlignedGoal(visual, pose, self._owner)

    @torch.no_grad()
    def predict(self, z, actions):
        self._state(z, 2)
        return AlignedLatent(self._sensor.predict(z.child, actions), self._owner)

    @torch.no_grad()
    def distance(self, predicted_z, goal_z):
        values = self._state(predicted_z, 4)
        self._goal(goal_z)
        visual = self._sensor.distance(predicted_z.child, goal_z.visual)
        indices = np.asarray(self.indices) + self._sensor.visual_dimension
        if values.shape[0] != goal_z.pose.shape[0]:
            raise ContractError("aligned goal batch differs")
        pose = (values[..., indices] - goal_z.pose[:, None, None]).square().mean(-1).cpu().numpy()
        return hybrid_cost(visual, pose, self.scales)

    @torch.no_grad()
    def observed_distance(self, observation_images, goal_images):
        current = self.encode_goal(observation_images)
        goal = self.encode_goal(goal_images)
        if current.pose.shape != goal.pose.shape:
            raise ContractError("observed image batch differs")
        visual = (current.visual.values - goal.visual.values).square().mean(-1).cpu().numpy()
        pose = (current.pose - goal.pose).square().mean(-1).cpu().numpy()
        return hybrid_cost(visual, pose, self.scales)

    def train_step(self, batch):
        raise ContractError("aligned_sensor_wm_v1 is a frozen composition; training is forbidden")

    def load(self, path):
        path = Path(path)
        envelope = torch.load(path, map_location="cpu", weights_only=True)
        expected = dict(
            format_version=1,
            backend=self.backend,
            config=self.config,
            state_schema=asdict(self.state_schema),
            action_schema=asdict(EE_DELTA_GRASP_V0),
            implementation_sha256=self.implementation_sha256,
        )
        for key, value in expected.items():
            if envelope.get(key) != value:
                raise ContractError(f"aligned checkpoint {key} is incompatible")
        for key, value in self.metadata.items():
            if envelope.get("metadata", {}).get(key) != value:
                raise ContractError(f"aligned provenance mismatch: {key}")
        members = envelope.get("members", {})
        if set(members) != set(ROLES):
            raise ContractError("aligned bundle member roles differ")
        files = {}
        for role, record in members.items():
            name = record.get("path")
            relative = Path(name) if isinstance(name, str) else Path("..")
            if (
                not name
                or relative.is_absolute()
                or len(relative.parts) != 1
                or name in (".", "..")
            ):
                raise ContractError("bundle members must be safe relative filenames")
            member = path.parent / relative
            if (
                member.is_symlink()
                or not member.is_file()
                or sha256(member) != record.get("sha256")
            ):
                raise ContractError(f"bundle member hash/path mismatch: {role}")
            files[role] = member
        sensor = SensorWorldModel(
            self.state_schema,
            device="cpu",
            seed=envelope["seed"],
            config=self.config["sensor_config"],
            metadata={
                k: envelope["metadata"][k] for k in ("dataset_hash", "split_hash", "action_hash")
            },
        )
        sensor.load(files["sensor"])
        if sensor.visual_dimension != 1728 or not bool(sensor.normalization_fitted):
            raise ContractError("aligned composition requires fitted 24x24 sensor features")
        dataset = json.loads(files["dataset_manifest"].read_text())
        metadata = envelope["metadata"]
        checks = dict(
            dataset_hash=sha256(files["dataset_manifest"]),
            split_hash=json_hash({"splits": dataset["splits"], "policy": dataset["split_policy"]}),
            action_hash=json_hash(dataset["action_manifest"]),
        )
        if any(metadata.get(k) != v for k, v in checks.items()):
            raise ContractError("bundle dataset/split/action identity mismatch")
        if dataset["state_schema"] != json.loads(json.dumps(asdict(self.state_schema))):
            raise ContractError("bundle dataset state schema differs")
        training = sorted(dataset["splits"]["train"])
        if (
            sorted(sensor.metadata["normalization"]["episode_ids"]) != training
            or metadata.get("normalization") != sensor.metadata["normalization"]
        ):
            raise ContractError("normalization must match complete frozen TRAIN membership")
        head = torch.load(files["goal_head"], map_location="cpu", weights_only=True)
        registration = json.loads(files["goal_registration"].read_text())
        report = json.loads(files["goal_fit_report"].read_text())
        seal = json.loads(files["goal_fit_seal"].read_text())
        samples = json.loads(files["goal_train_samples"].read_text())
        for filename, role in [
            ("head.pt", "goal_head"),
            ("registration.json", "goal_registration"),
            ("report.json", "goal_fit_report"),
        ]:
            if seal["artifacts"].get(filename) != sha256(files[role]):
                raise ContractError("head producer seal differs")
        identity = registration["identities"]
        if identity.get("script") != sha256(files["goal_source"]):
            raise ContractError("goal head training source hash differs")
        if (
            identity.get("checkpoint") != sha256(files["sensor"])
            or identity.get("dataset") != checks["dataset_hash"]
            or identity.get("prepare/train_samples.json") != sha256(files["goal_train_samples"])
        ):
            raise ContractError("head source model/data/TRAIN sample identity mismatch")
        if (
            report.get("status") != "completed"
            or not report.get("integrity_verified_after")
            or report.get("updates") != 2000
        ):
            raise ContractError("head producer did not complete verified fixed training")
        if not samples or any(
            r.get("split") != "train"
            or r.get("episode_id") not in training
            or r.get("parent") not in training
            for r in samples
        ):
            raise ContractError("goal head sample ledger must contain TRAIN-only family members")
        fixed_head_config = dict(
            seed=0,
            batch_size=128,
            updates=2000,
            lr=1e-3,
            weight_decay=1e-4,
            gradient_clip=10,
            logvar_bounds=[-6, 3],
        )
        if (
            head.get("config") != fixed_head_config
            or head.get("updates") != 2000
            or head.get("architecture") != [1728, 128, 64, 14]
            or tuple(head.get("fields", ())) != FIELDS
            or head.get("sample_sha256") != json_hash(samples)
        ):
            raise ContractError("goal head config/fields/sample identity mismatch")
        module = make_head()
        module.load_state_dict(head["head"], strict=True)
        if any(not torch.isfinite(p).all() for p in module.parameters()):
            raise ContractError("nonfinite goal head parameters")
        calibration = json.loads(files["calibration"].read_text())
        ledger = json.loads(files["calibration_ledger"].read_text())
        provenance = dict(
            packaging=calibration.get("packaging"),
            goal_source_sha256=sha256(files["goal_source"]),
            sensor_checkpoint_sha256=sha256(files["sensor"]),
            goal_head_sha256=sha256(files["goal_head"]),
            calibration_sha256=sha256(files["calibration"]),
        )
        if metadata.get("composition_provenance") != provenance:
            raise ContractError("composition packaging provenance differs")
        self._validate_calibration(calibration, ledger, dataset, files)
        # Commit only after every role/config/provenance check has succeeded.
        sensor.eval()
        sensor.requires_grad_(False)
        module.eval()
        module.requires_grad_(False)
        self._sensor = sensor
        self._head = module
        self.scales = copy.deepcopy(calibration["scales"])
        self.metadata = copy.deepcopy(metadata)
        self.seed = envelope["seed"]
        self._envelope = copy.deepcopy(envelope)
        self._members = files
        self._owner = object()
        return self

    @staticmethod
    def _validate_calibration(calibration, ledger, dataset, files):
        expected = dict(
            format_version=1,
            recipe=RECIPE,
            stride=28,
            weights=WEIGHTS,
            fields=list(FIELDS),
            dataset_hash=sha256(files["dataset_manifest"]),
            sensor_checkpoint_sha256=sha256(files["sensor"]),
            goal_head_sha256=sha256(files["goal_head"]),
            ledger_sha256=sha256(files["calibration_ledger"]),
        )
        if any(calibration.get(k) != v for k, v in expected.items()):
            raise ContractError("calibration recipe/provenance differs")
        packaging = calibration.get("packaging", {})
        for key in ("source_sha256", "protocol_sha256"):
            value = packaging.get(key)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ContractError("packaging source/protocol identity is required")
        if (
            not isinstance(packaging.get("source_revision"), str)
            or not packaging["source_revision"]
        ):
            raise ContractError("packaging source revision is required")
        rows = {r["episode_id"]: r for r in dataset["episodes"]}
        parents = sorted(
            p
            for p in dataset["splits"]["train"]
            if not rows[p]["metadata"].get("parent_episode_id")
        )
        if sorted(calibration.get("parent_ids", [])) != parents:
            raise ContractError("calibration requires every original TRAIN parent")
        expected_pairs = {
            (p, t, t + 28) for p in parents for t in range(0, rows[p]["length"] - 28, 28)
        }
        pairs = ledger.get("pairs", [])
        actual = [(r["parent_id"], r["frame0"], r["frame1"]) for r in pairs]
        if len(set(actual)) != len(actual) or set(actual) != expected_pairs:
            raise ContractError("calibration stride28 pair coverage differs")
        scales = {}
        for component in ("visual", "pose"):
            medians = []
            for parent in parents:
                values = [r[component + "_mse"] for r in pairs if r["parent_id"] == parent]
                if not values or any(not np.isfinite(v) or v < 0 for v in values):
                    raise ContractError("invalid calibration pair distances")
                medians.append(statistics.median(values))
            scales[component] = statistics.median(medians)
        hybrid_cost(np.zeros(1), np.zeros(1), calibration["scales"])
        if any(
            not np.isclose(calibration["scales"][k], v, rtol=1e-12, atol=0)
            for k, v in scales.items()
        ):
            raise ContractError("calibration scales differ from TRAIN parent medians")

    def save(self, path):
        self._ready()
        return _copy_bundle(path, self._envelope, self._members)


def _copy_bundle(path, envelope, files):
    path = Path(path)
    if path.exists():
        raise FileExistsError("refusing to overwrite aligned checkpoint")
    path.parent.mkdir(parents=True, exist_ok=True)
    members = {}
    destinations = {role: path.parent / ("bundle_" + role + files[role].suffix) for role in ROLES}
    if any(p.exists() for p in destinations.values()):
        raise FileExistsError("aligned bundle members already exist")
    for role in ROLES:
        source = files[role]
        if sha256(source) != envelope["members"][role]["sha256"]:
            raise ContractError("source bundle changed before save")
    for role, target in destinations.items():
        shutil.copyfile(files[role], target)
        members[role] = dict(path=target.name, sha256=sha256(target))
    state = copy.deepcopy(envelope)
    state["members"] = members
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)
    return path


def package_bundle(
    destination_checkpoint,
    *,
    sensor_checkpoint,
    head_checkpoint,
    calibration_json,
    provenance_files,
):
    if set(provenance_files) != set(PROVENANCE_ROLES):
        raise ContractError("bundle provenance roles differ")
    files = {
        "sensor": Path(sensor_checkpoint),
        "goal_head": Path(head_checkpoint),
        "calibration": Path(calibration_json),
    } | {k: Path(v) for k, v in provenance_files.items()}
    child = torch.load(files["sensor"], map_location="cpu", weights_only=True)
    schema = StateSchema(**child["state_schema"])
    config = dict(sensor_config=child["config"], weights=WEIGHTS.copy(), version=1)
    model = AlignedSensorWorldModel(schema, config=config, seed=child["seed"])
    envelope = dict(
        format_version=1,
        backend=model.backend,
        config=config,
        state_schema=asdict(schema),
        action_schema=asdict(EE_DELTA_GRASP_V0),
        metadata=copy.deepcopy(child["metadata"]),
        seed=child["seed"],
        implementation_sha256=model.implementation_sha256,
        members={role: dict(path=path.name, sha256=sha256(path)) for role, path in files.items()},
    )
    calibration = json.loads(files["calibration"].read_text())
    envelope["metadata"]["composition_provenance"] = dict(
        packaging=calibration.get("packaging"),
        goal_source_sha256=sha256(files["goal_source"]),
        sensor_checkpoint_sha256=sha256(files["sensor"]),
        goal_head_sha256=sha256(files["goal_head"]),
        calibration_sha256=sha256(files["calibration"]),
    )
    path = _copy_bundle(destination_checkpoint, envelope, files)
    model.load(path)
    return path
