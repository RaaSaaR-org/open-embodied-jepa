import subprocess
import sys

import pytest

from embodied_jepa.registry import Registry


def test_registry_validates_names_and_constructs_lazily():
    registry = Registry("fixture")
    registry.register("dict", "builtins:dict")
    assert registry.create("dict", a=1) == {"a": 1}
    with pytest.raises(ValueError, match="duplicate"):
        registry.register("dict", dict)
    with pytest.raises(ValueError, match="unknown"):
        registry.create("missing")


def test_core_import_does_not_import_optional_runtimes():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, embodied_jepa.registry; "
            "assert 'torch' not in sys.modules; assert 'mujoco' not in sys.modules",
        ],
        check=True,
    )


@pytest.mark.parametrize("factory", [":x", "x:", "x", "a:b:c"])
def test_registry_rejects_malformed_lazy_factory(factory):
    with pytest.raises(ValueError, match="module:attribute"):
        Registry("fixture").register("bad", factory)


def test_registry_rejects_non_callable_factory():
    with pytest.raises(TypeError, match="callable"):
        Registry("fixture").register("bad", 42)
