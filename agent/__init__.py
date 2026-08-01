"""Baseline doctor agent package."""

__all__ = ["MyDoctorAgent", "build_memory"]


def __getattr__(name: str):
    """Keep public imports lazy so module execution does not import itself twice."""
    if name == "MyDoctorAgent":
        from .agent import MyDoctorAgent

        return MyDoctorAgent
    if name == "build_memory":
        from .memory import build_memory

        return build_memory
    raise AttributeError(name)
