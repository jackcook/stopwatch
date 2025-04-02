from .db import engine, session
from .base import Base
from .benchmark import (
    Benchmark,
    benchmark_cls_factory,
)


def create_all():
    Base.metadata.create_all(engine)


__all__ = [
    "Benchmark",
    "benchmark_cls_factory",
    "create_all",
    "session",
]
