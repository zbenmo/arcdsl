"""
Solutions registry
"""

from .dsl import Transform


registry = dict()


def register(task: str):
    def wrapper(transform: Transform):
        registry[task] = transform
        return transform
    return wrapper
