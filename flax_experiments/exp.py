# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/02_experiment.ipynb (unless otherwise specified).

__all__ = ['OK', 'Wrong']

# Cell
class OK:
    def __init__(self, x):
        self.x = x

# Cell

#export
from dataclasses import dataclass

@dataclass
class Wrong:
    x: int