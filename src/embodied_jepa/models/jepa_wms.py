"""Optional CC-BY-NC upstream architecture with a project-owned training recipe.

No upstream code or weights are bundled. Imports resolve in a private namespace;
other libraries' generic ``src`` packages remain untouched.
"""

from __future__ import annotations

import builtins
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path
from types import ModuleType

from torch import nn

from embodied_jepa.contracts import ContractError
from embodied_jepa.models.base import VisualModel
from embodied_jepa.models.native import NativeJEPA

MANIFEST = json.loads(Path(__file__).with_name("jepa_wms_source.json").read_text())
JEPA_WMS_REVISION = MANIFEST["revision"]
_IMPORT_LOCK = threading.RLock()


def load_upstream(source):
    """Verify every imported file before loading unchanged source under private names."""
    source = Path(source).resolve()
    if not source.is_dir():
        raise ContractError("JEPA-WMs source missing; run scripts/fetch_jepa_wms.py")
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True, stderr=subprocess.PIPE
        ).strip()
        if revision != JEPA_WMS_REVISION:
            raise ContractError(f"JEPA-WMs requires source revision {JEPA_WMS_REVISION}")
        for relative, expected in MANIFEST["files"].items():
            if hashlib.sha256((source / relative).read_bytes()).hexdigest() != expected:
                raise ContractError(f"JEPA-WMs source integrity failure: {relative}")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError("JEPA-WMs source checkout is incomplete") from exc
    namespace = "_embodied_jepa_wms_" + hashlib.sha256(str(source).encode()).hexdigest()[:16]

    def private_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "src" or name.startswith("src."):
            name = namespace + name[3:]
        return builtins.__import__(name, globals, locals, fromlist, level)

    with _IMPORT_LOCK:
        if namespace not in sys.modules:
            try:
                for package in ("", ".utils", ".masks", ".models", ".models.utils"):
                    module = ModuleType(namespace + package)
                    module.__path__ = []
                    sys.modules[module.__name__] = module
                for relative in MANIFEST["files"]:
                    if not relative.endswith(".py"):
                        continue
                    name = namespace + relative[3:-3].replace("/", ".")
                    spec = importlib.util.spec_from_file_location(name, source / relative)
                    module = importlib.util.module_from_spec(spec)
                    module.__dict__["__builtins__"] = vars(builtins) | {
                        "__import__": private_import
                    }
                    sys.modules[name] = module
                    spec.loader.exec_module(module)
            except Exception:
                for name in list(sys.modules):
                    if name == namespace or name.startswith(namespace + "."):
                        del sys.modules[name]
                raise
        return (
            sys.modules[namespace + ".models.vision_transformer"].VisionTransformer,
            sys.modules[namespace + ".models.ac_predictor"].VisionTransformerPredictorAC,
        )


class JEPAWMs(NativeJEPA):
    """Real upstream encoder/predictor, random initialization, visual-only common mode.

    The existing Apache-owned EMA/variance training recipe is reused explicitly;
    this compact compatibility configuration is not a reproduction of the paper.
    """

    backend = "jepa_wms"
    defaults = NativeJEPA.defaults | {
        "image_size": 32,
        "latent_dim": 512,
        "patch_size": 8,
        "token_dim": 32,
        "encoder_depth": 1,
        "predictor_depth": 1,
        "attention_heads": 4,
        "source_path": "third_party/jepa-wms",
        "accept_noncommercial_source": False,
    }

    def __init__(self, state_schema, device="cpu", seed=0, config=None, metadata=None):
        if device != "cpu":
            raise ContractError("JEPA-WMs compatibility configuration is validated only on CPU")
        overrides = dict(config or {})
        merged = self.defaults | overrides
        for name in (
            "image_size",
            "patch_size",
            "token_dim",
            "encoder_depth",
            "predictor_depth",
            "attention_heads",
        ):
            if type(merged[name]) is not int or merged[name] < 1:
                raise ContractError(f"{name} must be a positive integer")
        if merged["accept_noncommercial_source"] is not True:
            raise ContractError("JEPA-WMs requires explicit accept_noncommercial_source=true")
        if (
            merged["image_size"] % merged["patch_size"]
            or merged["token_dim"] % 4
            or merged["token_dim"] % merged["attention_heads"]
        ):
            raise ContractError("image/patch, token/heads and token/4 must divide evenly")
        latent_dim = (merged["image_size"] // merged["patch_size"]) ** 2 * merged["token_dim"]
        if "latent_dim" in overrides and overrides["latent_dim"] != latent_dim:
            raise ContractError("latent_dim must equal patch count times token_dim")
        overrides["latent_dim"] = latent_dim
        VisualModel.__init__(self, state_schema, device, seed, overrides, metadata)
        import numpy as np

        if not 0 <= self.config["ema_decay"] < 1:
            raise ContractError("ema_decay must lie in [0,1)")
        for name in ("variance_weight", "covariance_weight", "multistep_weight"):
            if not np.isfinite(self.config[name]) or self.config[name] < 0:
                raise ContractError(f"{name} must be nonnegative and finite")
        encoder_class, predictor_class = load_upstream(self.config["source_path"])
        c = self.config
        with self.rng_scope():
            self.encoder = nn.Sequential(
                encoder_class(
                    input_size=c["image_size"],
                    patch_size=c["patch_size"],
                    embed_dim=c["token_dim"],
                    depth=c["encoder_depth"],
                    num_heads=c["attention_heads"],
                ),
                nn.Flatten(start_dim=1),
            )
            self.predictor = predictor_class(
                img_size=c["image_size"],
                patch_size=c["patch_size"],
                num_frames=1,
                tubelet_size=1,
                embed_dim=c["token_dim"],
                predictor_embed_dim=c["token_dim"],
                depth=c["predictor_depth"],
                num_heads=c["attention_heads"],
                action_dim=14,
                proprio_tokens=0,
                use_rope=False,
                is_frame_causal=True,
            )
        self.target_encoder = copy.deepcopy(self.encoder).requires_grad_(False)
        self.finish_init()

    @property
    def capabilities(self):
        return replace(super().capabilities, supported_devices=("cpu",))

    @property
    def source_revision(self):
        return JEPA_WMS_REVISION

    @property
    def implementation_sha256(self):
        digest = hashlib.sha256(super().implementation_sha256.encode())
        for filename in ("native.py", "jepa_wms_source.json"):
            digest.update(Path(__file__).with_name(filename).read_bytes())
        return digest.hexdigest()

    def next_embedding(self, state, actions):
        shape = state.shape
        patches = state.reshape(
            -1, self.config["latent_dim"] // self.config["token_dim"], self.config["token_dim"]
        )
        # The upstream API explicitly disables proprio tokens; no fabricated or
        # future robot state enters this call. All 14 canonical actions are used.
        predicted, _, _ = self.predictor(patches, actions.reshape(-1, 1, 14), states=None)
        return predicted.reshape(shape)
