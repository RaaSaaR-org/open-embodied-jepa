"""Registered normalized action schemas, looked up by version (SO-101 S0, TASK-096).

``contracts.py`` defines the frozen 14-D ``ee_delta_grasp_v0`` and is byte-pinned by the frozen
G1 protocols (TASK-091–TASK-094), so new schemas are registered here, beside it, rather than
there. A schema is identity only (version, ordered component names, dimension): physical scales,
limits and the gripper's mapping belong to the embodiment's action manifest.

``joint_delta_so101_v0`` is the SO-101's 6-D schema: five arm joint-position deltas and one
gripper command, in LeRobot ``so_follower``'s motor order. Its physical meaning (per-joint delta
scales, joint limits, the gripper range and control rate) is the SO-101 embodiment manifest's
(S1, TASK-097), not this module's.

NumPy only; importing it never loads torch or MuJoCo.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict
from types import MappingProxyType
from typing import Any

from embodied_jepa.contracts import ACTION_SCHEMA, EE_DELTA_GRASP_V0, ActionSchema, ContractError

JOINT_DELTA_SO101_V0_VERSION = "joint_delta_so101_v0"
# LeRobot so_follower motor order (lerobot/robots/so_follower): five arm joints, then the jaw.
SO101_ACTION_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
JOINT_DELTA_SO101_V0 = ActionSchema(JOINT_DELTA_SO101_V0_VERSION, SO101_ACTION_NAMES)

ACTION_SCHEMAS: Mapping[str, ActionSchema] = MappingProxyType(
    {
        ACTION_SCHEMA: EE_DELTA_GRASP_V0,
        JOINT_DELTA_SO101_V0_VERSION: JOINT_DELTA_SO101_V0,
    }
)


def action_schema(version: Any) -> ActionSchema:
    """The registered schema called ``version``; an unknown name is an error, never a guess."""
    if not isinstance(version, str) or version not in ACTION_SCHEMAS:
        raise ContractError(
            f"unknown action schema {version!r}; registered: {sorted(ACTION_SCHEMAS)}"
        )
    return ACTION_SCHEMAS[version]


def schema_record(schema: ActionSchema) -> dict[str, Any]:
    """The JSON form stored in dataset manifests and checkpoints (names as a list)."""
    return json.loads(json.dumps(asdict(schema)))


def action_schema_from_record(record: Any) -> ActionSchema:
    """The registered schema whose stored record equals ``record`` exactly (version and order)."""
    if not isinstance(record, Mapping) or set(record) != {"version", "names"}:
        raise ContractError("action schema record must hold exactly a version and names")
    schema = action_schema(record["version"])
    if dict(record) != schema_record(schema):
        raise ContractError(f"{schema.version} action ordering/names are incompatible")
    return schema
