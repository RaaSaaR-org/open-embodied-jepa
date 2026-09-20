"""Lazy exports keep torch and upstream model dependencies optional."""


def __getattr__(name):
    if name == "NativeJEPA":
        from .native import NativeJEPA

        return NativeJEPA
    if name == "LeWM":
        from .lewm import LeWM

        return LeWM
    raise AttributeError(name)


__all__ = ["NativeJEPA", "LeWM"]
