"""
Disentanglement datasets module.

Contains implementations of datasets designed for evaluating disentangled
representation learning algorithms.
"""
from .datasets import DSprites, ThreeDShapes, Cars3D, MPI3DReal, SmallNORB

__all__ = [
    "DSprites",
    "ThreeDShapes",
    "Cars3D",
    "MPI3DReal",
    "SmallNORB",
]
