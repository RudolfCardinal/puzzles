#!/usr/bin/env python

"""
common.py

===============================================================================

    Copyright (C) 2019-2019 Rudolf Cardinal (rudolf@pobox.com).

    This is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This software is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this software. If not, see <http://www.gnu.org/licenses/>.

===============================================================================

Common constants and functions for the puzzle solvers.

"""

import logging

from mip import Constr, Model, Var

log = logging.getLogger(__name__)


# =============================================================================
# Functions for mip models
# =============================================================================

def debug_model_constraints(m: Model) -> None:
    """
    Shows constraints for a model.
    """
    lines = [f"Constraints in model {m.name!r}:"]
    for c in m.constrs:  # type: Constr
        lines.append(f"{c.name} == {c.expr}")
    log.debug("\n".join(lines))


def debug_model_vars(m: Model) -> None:
    """
    Show the names/values of model variables after fitting.
    """
    lines = [f"Variables in model {m.name!r}:"]
    for v in m.vars:  # type: Var
        lines.append(f"{v.name} == {v.x}")
    log.debug("\n".join(lines))
