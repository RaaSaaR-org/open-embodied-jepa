"""Actual optional upstream architecture and import-integrity regressions."""

import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

pytest.importorskip("torch")

from embodied_jepa.contracts import ContractError, StateSchema  # noqa: E402
from embodied_jepa.models.jepa_wms import JEPAWMs, load_upstream  # noqa: E402
from embodied_jepa.planning import CEMConfig, CEMPlanner  # noqa: E402


def test_optional_core_import_is_lazy():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import embodied_jepa.config; "
            "assert not {'torch','timm'} & sys.modules.keys()",
        ],
        check=True,
    )


def test_rejects_noncommercial_optin_missing_source_and_unvalidated_device(tmp_path):
    schema = StateSchema(("q",), ("rad",), "fixture")
    with pytest.raises(ContractError, match="noncommercial"):
        JEPAWMs(schema)
    with pytest.raises(ContractError, match="source missing"):
        JEPAWMs(
            schema,
            config={"accept_noncommercial_source": True, "source_path": str(tmp_path / "absent")},
        )
    with pytest.raises(ContractError, match="only on CPU"):
        JEPAWMs(schema, device="mps")


@pytest.fixture
def source():
    pytest.importorskip("timm")
    path = Path(os.environ.get("JEPA_WMS_SOURCE", "third_party/jepa-wms"))
    if not path.exists():
        pytest.skip("optional pinned JEPA-WMs source absent")
    return path


def test_genuine_upstream_classes_private_namespace_and_cem(source, monkeypatch):
    import torch

    from embodied_jepa.contracts import RobotState

    sentinel = ModuleType("src")
    monkeypatch.setitem(sys.modules, "src", sentinel)
    encoder, predictor = load_upstream(source)
    assert encoder.__name__ == "VisionTransformer"
    assert predictor.__name__ == "VisionTransformerPredictorAC"
    assert sys.modules["src"] is sentinel
    schema = StateSchema(("q",), ("rad",), "fixture")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = JEPAWMs(
            schema,
            config={
                "source_path": str(source),
                "image_size": 16,
                "token_dim": 16,
                "accept_noncommercial_source": True,
            },
        )
        assert isinstance(model.encoder[0], encoder)
        assert isinstance(model.predictor, predictor)
        images = {"onboard_rgb": np.zeros((1, 16, 16, 3), np.uint8)}
        state = RobotState(
            np.zeros((1, 1), np.float32), np.ones((1, 1), bool), np.array([0.0]), schema
        )
        latent = model.encode(images, state)
        goal = model.encode_goal(images)
        plan = CEMPlanner(CEMConfig(horizon=2, samples=4, iterations=1, elites=2)).plan(
            model, latent, goal
        )
        assert plan.actions.shape == (1, 2, 14)
        assert np.isfinite(plan.costs).all()
    finally:
        torch.set_num_threads(threads)


def test_tamper_rejected_even_after_cached_load(source, tmp_path):
    import shutil

    # Copy only the audited closure and reference the original Git object store;
    # this test never modifies the shared or upstream checkout.
    from embodied_jepa.models.jepa_wms import MANIFEST

    for name in MANIFEST["files"]:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    (tmp_path / ".git").write_text(f"gitdir: {(source / '.git').resolve()}\n")
    load_upstream(tmp_path)
    target = tmp_path / "src/models/ac_predictor.py"
    target.write_text(target.read_text() + "\n# changed\n")
    with pytest.raises(ContractError, match="integrity failure"):
        load_upstream(tmp_path)
