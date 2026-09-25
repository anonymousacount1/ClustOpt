"""Component package for the HYBRID_SCM generator.

Importing this package registers every component with the global registry
exposed via ``get_component``. The public API (``Component``, ``register``,
``get_component``, ``apply_component_operators``) matches the old
``components.py`` module so existing imports keep working.
"""
from __future__ import annotations

from .base import Component, _REGISTRY, get_component, register

# Importing each component module triggers its ``register(...)`` call at
# module load. Order preserved from the original components.py for stable
# error-message ordering.
from . import band as _band  # noqa: F401
from . import ladder as _ladder  # noqa: F401
from . import parallel_bands as _parallel_bands  # noqa: F401
from . import parabola as _parabola  # noqa: F401
from . import arch_chain as _arch_chain  # noqa: F401
from . import circle as _circle  # noqa: F401
from . import piecewise_polyline as _piecewise_polyline  # noqa: F401
from . import ring_annulus as _ring_annulus  # noqa: F401
from . import polygon as _polygon  # noqa: F401
from . import grid_lattice as _grid_lattice  # noqa: F401
from . import branching_tree as _branching_tree  # noqa: F401
from . import meander_curve as _meander_curve  # noqa: F401
from . import blob as _blob  # noqa: F401
from . import noise as _noise  # noqa: F401
from . import ellipse as _ellipse  # noqa: F401
from . import rectangle_frame as _rectangle_frame  # noqa: F401
from . import junction as _junction  # noqa: F401
from . import spiral as _spiral  # noqa: F401
from . import ridge_field as _ridge_field  # noqa: F401
from . import blob_field as _blob_field  # noqa: F401
from . import fractal_dust as _fractal_dust  # noqa: F401
from . import oriented_texture_field as _oriented_texture_field  # noqa: F401
from . import micro_blob_texture as _micro_blob_texture  # noqa: F401
from . import striped_texture as _striped_texture  # noqa: F401
from . import checker_texture as _checker_texture  # noqa: F401
from . import speckle_texture as _speckle_texture  # noqa: F401
from . import anisotropic_grain_field as _anisotropic_grain_field  # noqa: F401

from .operators import apply_component_operators

__all__ = [
    "Component",
    "_REGISTRY",
    "get_component",
    "register",
    "apply_component_operators",
]
