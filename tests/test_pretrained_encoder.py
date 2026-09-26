"""TASK-063 frozen pretrained encoder: pins, input handling, floor init, read-out shapes.

Synthetic evidence only (tiny configs, random frames); no pretrained weight is downloaded or read,
and these tests make no learned-representation or control claims.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from embodied_jepa import pretrained_encoder as pe
from embodied_jepa.contracts import ContractError

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads(
    (ROOT / "benchmarks" / "manifests" / "apple-pretrained-encoder-v1.json").read_text()
)


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ----- the manifest and the module agree ------------------------------------------------------
def test_manifest_pins_match_the_module_and_the_fetch_script():
    fetch = _load("_fetch_dinov2", "scripts/fetch_dinov2.py")
    encoder = MANIFEST["encoder"]
    assert encoder["hf_repository"] == pe.HF_REPOSITORY == fetch.HF_REPOSITORY
    assert encoder["hf_revision"] == pe.HF_REVISION == fetch.HF_REVISION
    for name, want in pe.FILE_SHA256.items():
        assert encoder["files"][name]["sha256"] == want == fetch.HF_FILES[name]
        assert MANIFEST["hashes"][f"third_party/dinov2-small/{name}"] == want
    assert encoder["official_checkpoint"]["sha256"] == fetch.OFFICIAL_SHA256
    assert encoder["official_checkpoint"]["equivalence"]["identical"] is True
    assert MANIFEST["input_handling"]["token_grid"] == [pe.GRID, pe.GRID]
    assert MANIFEST["input_handling"]["normalise"] == {
        "mean": list(pe.IMAGENET_MEAN),
        "std": list(pe.IMAGENET_STD),
    }
    arms = MANIFEST["arms_decisional_in_holm_order"]
    assert [a["arm"] for a in arms] == ["P-cls", "P-tok"]
    assert [a["dim"] for a in arms] == [pe.WIDTH, pe.GRID * pe.GRID * pe.WIDTH]
    assert MANIFEST["multiplicity"]["step_thresholds"] == [0.0125, 0.025]
    assert MANIFEST["seeds"]["floor_init"] == pe.FLOOR_SEED
    assert MANIFEST["budget"]["device"] == "cpu"
    assert MANIFEST["learned_apple_to_plate_successes"] == 0
    assert MANIFEST["exemption_spent"] is False


def test_manifest_pins_the_module_bytes_and_the_reused_probe_code():
    hashes = MANIFEST["hashes"]
    for name in (
        "src/embodied_jepa/pretrained_encoder.py",
        "scripts/fetch_dinov2.py",
        "scripts/calibrate_pretrained_encoder.py",
        "scripts/probe_observation_reprobe.py",
        "src/embodied_jepa/observation_reprobe.py",
        "scripts/probe_info_ceiling.py",
        "src/embodied_jepa/info_ceiling.py",
        "src/embodied_jepa/encoder_study.py",
        "scripts/probe_encoder_study.py",
        "scripts/calibrate_encoder_study.py",
    ):
        digest = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        assert digest == hashes[name], name
    for name in (
        "outputs/task061-observation-reprobe/run-1/report.json",
        "outputs/task062-encoder-study/run-1/report.json",
        "outputs/task063-pretrained-encoder/calibration-v1.json",
    ):
        assert name in hashes


def test_manifest_outcome_rows_and_abandonment_clause():
    rows = [r["row"] for r in MANIFEST["pre_declared_outcomes_in_order"]]
    assert rows == ["V", "O-PT-POOLED", "O-PT-TOKENS", "O-PT-INCOMPLETE", "O-PT-FLOOR", "O-PT-NONE"]
    clause = MANIFEST["abandonment_clause"]
    assert clause["fires_on"] == ["O-PT-FLOOR", "O-PT-NONE"]
    assert "overview camera" in clause["context_not_a_preregistered_follow_up"]
    assert "INCONCLUSIVE" in MANIFEST["inconclusive_task_level"]


def test_importing_the_module_does_not_import_torch():
    code = (
        "import sys; import embodied_jepa.pretrained_encoder; "
        "assert 'torch' not in sys.modules and 'transformers' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


# ----- file pins --------------------------------------------------------------------------------
def test_check_files_rejects_missing_and_different_files(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    good = hashlib.sha256(b"{}").hexdigest()
    assert pe.check_files(tmp_path, {"config.json": good}) == {"config.json": good}
    with pytest.raises(pe.WeightsError, match="sha256"):
        pe.check_files(tmp_path, {"config.json": "0" * 64})
    with pytest.raises(pe.WeightsError, match="missing"):
        pe.check_files(tmp_path, {"model.safetensors": good})


# ----- torch-backed checks (skipped only where torch/transformers are not installed) ------------
def _tiny_dir(tmp_path, layers=1) -> tuple[Path, dict]:
    """A ViT-S/14-shaped config with ``layers`` blocks (the real one has 12)."""
    real = {
        "architectures": ["Dinov2Model"],
        "hidden_size": 384,
        "num_attention_heads": 6,
        "num_hidden_layers": layers,
        "patch_size": 14,
        "image_size": 518,
        "mlp_ratio": 4,
        "layerscale_value": 1.0,
        "model_type": "dinov2",
        "qkv_bias": True,
        "use_swiglu_ffn": False,
        "layer_norm_eps": 1e-6,
        "initializer_range": 0.02,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(real))
    return tmp_path, {"config.json": pe.sha256(path)}


def test_preprocess_resizes_without_crop_and_normalises():
    pytest.importorskip("torch")
    with pytest.raises(ContractError):
        pe.preprocess(np.zeros((2, 112, 112, 3), np.float32))
    with pytest.raises(ContractError):
        pe.preprocess(np.zeros((2, 224, 224, 3), np.uint8))
    gray = np.full((3, 112, 112, 3), 128, np.uint8)
    x = pe.preprocess(gray).numpy()
    assert x.shape == (3, 3, 224, 224) and x.dtype == np.float32
    want = (128 / 255 - np.array(pe.IMAGENET_MEAN)) / np.array(pe.IMAGENET_STD)
    assert np.allclose(x, want[None, :, None, None], atol=1e-5)
    # A bright left border survives (no centre crop).
    frame = np.zeros((1, 112, 112, 3), np.uint8)
    frame[:, :, :4] = 255
    y = pe.preprocess(frame).numpy()
    assert y[0, :, :, :4].mean() > y[0, :, :, 112:].mean()


def test_floor_init_is_seeded_isolated_and_reproducible(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    directory, pinned = _tiny_dir(tmp_path)
    torch.manual_seed(123)
    before = torch.rand(3)
    torch.manual_seed(123)
    first = pe.random_init(directory, pinned=pinned)
    after = torch.rand(3)
    assert torch.equal(before, after), "building the floor moved the global RNG"
    second = pe.random_init(directory, pinned=pinned)
    assert pe.weights_digest(first) == pe.weights_digest(second)
    assert not first.training
    with pytest.raises(pe.WeightsError):
        pe.random_init(directory, pinned={"config.json": "0" * 64})


def test_features_shapes_determinism_and_non_finite_rejection(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    directory, pinned = _tiny_dir(tmp_path)
    model = pe.random_init(directory, pinned=pinned)
    frames = np.random.default_rng(0).integers(0, 256, (5, 112, 112, 3), dtype=np.uint8)
    out = pe.features(model, frames, batch=2)
    assert out["cls"].shape == (5, pe.WIDTH) and out["cls"].dtype == np.float64
    assert out["tokens"].shape == (5, pe.GRID * pe.GRID * pe.WIDTH)
    again = pe.features(model, frames, batch=5)
    assert all(np.array_equal(out[k], again[k]) for k in pe.READOUT_POINTS)
    staged = pe.features(model, frames, hidden_states=True)
    assert {"patch_embedding_tokens", "block1_tokens", "tokens_mean"} <= set(staged)
    with torch.no_grad():
        model.layernorm.weight.fill_(float("nan"))
    with pytest.raises(ContractError, match="non-finite"):
        pe.features(model, frames)


def test_official_key_map_covers_the_hf_state_dict_exactly(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    fetch = _load("_fetch_dinov2", "scripts/fetch_dinov2.py")
    directory, pinned = _tiny_dir(tmp_path, layers=2)
    model = pe.random_init(directory, pinned=pinned)
    hf = model.state_dict()
    official = {k: torch.zeros(1) for k in fetch._sources_used(2)}
    official["blocks.0.attn.qkv.weight"] = torch.arange(9.0).reshape(9, 1)
    mapped = fetch.official_key_map(official, 2)
    assert set(mapped) == set(hf)
    assert mapped["encoder.layer.0.attention.attention.key.weight"].flatten().tolist() == [
        3.0,
        4.0,
        5.0,
    ]
