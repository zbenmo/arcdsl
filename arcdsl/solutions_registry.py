"""
Solutions registry
"""

from .dsl import Transform
from collections import defaultdict


registry = defaultdict(list)


def register(task: str):
    def wrapper(transform: Transform):
        registry[task].append(transform)
        return transform
    return wrapper
