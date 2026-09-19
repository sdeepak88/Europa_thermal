# -*- coding: utf-8 -*-
"""
Thin loader for FEM+GravityInversion.py.

That module's filename contains a '+', which is not a legal Python
identifier, so it cannot be reached with a normal `import` statement. All
of the review-driven sensitivity-study scripts in this directory need its
physics engine (compute_temperature, run_monte_carlo, build_grid, the
physical constants, etc.), so rather than duplicate ~900 lines of solver
code into every script, this module loads it once via importlib and
re-exports its public names.

Usage:
    from _fem_core import fem   # fem.compute_temperature(...), fem.G, ...
"""

import importlib.util
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_FEM_PATH = os.path.join(_THIS_DIR, "FEM+GravityInversion.py")

_spec = importlib.util.spec_from_file_location("fem_gravity_inversion", _FEM_PATH)
fem = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fem)
