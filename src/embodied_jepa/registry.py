"""Lazy named factories keep optional runtimes outside the required core."""

from collections.abc import Callable
from importlib import import_module
from typing import Any


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._factories: dict[str, str | Callable[..., Any]] = {}

    def register(self, name: str, factory: str | Callable[..., Any]) -> None:
        if not name or name in self._factories:
            raise ValueError(f"duplicate or empty {self.kind}: {name!r}")
        if isinstance(factory, str):
            parts = factory.split(":")
            if len(parts) != 2 or not all(parts):
                raise ValueError("lazy factories must use nonempty module:attribute")
        elif not callable(factory):
            raise TypeError("factory must be callable or a module:attribute string")
        self._factories[name] = factory

    def require(self, name: str) -> None:
        if name not in self._factories:
            raise ValueError(f"unknown {self.kind} {name!r}; available: {sorted(self._factories)}")

    def create(self, name: str, **kwargs: Any) -> Any:
        self.require(name)
        factory = self._factories[name]
        if isinstance(factory, str):
            module, attribute = factory.split(":", 1)
            factory = getattr(import_module(module), attribute)
        return factory(**kwargs)


MODELS = Registry("world model")
EMBODIMENTS = Registry("embodiment")
PLANNERS = Registry("planner")
POLICIES = Registry("policy")
TASKS = Registry("task")
