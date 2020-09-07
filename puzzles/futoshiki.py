#!/usr/bin/env python

"""
futoshiki.py

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

**Solves Futoshiki puzzles.**

Like the Sudoku solver, it uses integer programming ("give me the answer") and
an attempt at a logic-based approach ("show me how").

"""


import argparse
from copy import deepcopy
import logging
import math
import sys
import traceback
from typing import Generator, List, Optional, Tuple

from cardinal_pythonlib.argparse_func import RawDescriptionArgumentDefaultsHelpFormatter  # noqa
from cardinal_pythonlib.logs import main_only_quicksetup_rootlogger
# from cardinal_pythonlib.maths_py import sum_of_integers_in_inclusive_range
from mip import BINARY, Model, xsum

from common import (
    ALMOST_ONE,
    CommonPossibilities,
    debug_model_vars,
    DISPLAY_INITIAL,
    DISPLAY_SOLVED,
    DISPLAY_UNKNOWN,
    EXIT_SUCCESS,
    EXIT_FAILURE,
    HASH,
    NEWLINE,
    run_guard,
    SPACE,
    UNKNOWN,
)

log = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# String representations
LEFT_LT_RIGHT = "<"
LEFT_GT_RIGHT = ">"
TOP_LT_BOTTOM = "^"
BOTTOM_LT_TOP = "v"
BOTTOM_LT_TOP_POSSIBLES = "vV"

DEMO_FUTOSHIKI_1 = """
# Times, 2 Dec 2019, Futoshiki no. 3576

.>.<. . .

. . .>. .
    ^
. .>. .<.

. . . . .
^   ^
.<. . . .
"""


# =============================================================================
# FutoshikiPossibilities
# =============================================================================

class FutoshikiPossibilities(CommonPossibilities):
    """
    Represents Futoshiki possibilities, for a logic-based solver.
    """
    VISUAL_INEQ_GAP = 3

    def __init__(self, other: "FutoshikiPossibilities" = None,
                 n: int = 5) -> None:
        """
        Initialize with "everything is possible", or copy from another.
        """
        super().__init__(other=other, n=n)
        if other:
            self.inequalities_right = deepcopy(other.inequalities_right)
            self.inequalities_down = deepcopy(other.inequalities_down)
        else:
            self.inequalities_right = [
                [SPACE for _c in range(n - 1)]
                for _r in range(n)
            ]
            # ... index as: self.inequalities_right[row_zb][col_zb]
            self.inequalities_down = [
                [SPACE for _c in range(n)]
                for _r in range(n - 1)
            ]
            # ... index as: self.inequalities_right[row_zb][col_zb]

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------

    # noinspection PyMethodOverriding
    def set_initial_values(self,
                           initial_values: List[List[Optional[int]]],
                           inequalities_right: List[List[str]],
                           inequalities_down: List[List[str]]) -> None:
        """
        Set up the initial values. The incoming digits are genuine (one-based),
        not zero-based.
        """
        super().set_initial_values(initial_values=initial_values)

        assert len(inequalities_right) == self.n
        for ineq_right_row in inequalities_right:
            assert len(ineq_right_row) == self.n - 1
        self.inequalities_right = inequalities_right

        assert len(inequalities_down) == self.n - 1
        for ineq_down_row in inequalities_down:
            assert len(ineq_down_row) == self.n
        self.inequalities_down = inequalities_down

    # -------------------------------------------------------------------------
    # Visuals
    # -------------------------------------------------------------------------

    def _pstr_cell_sides(self) -> Tuple[int, int]:
        """
        We show possibilities in a sx-by-sy box; this function returns ``sx,
        sy``.
        """
        sx = math.ceil(math.sqrt(self.n))
        sy = math.ceil(self.n / sx)
        return sx, sy

    def _pstr_value_row_col(self, row_zb: int, col_zb: int, digit_zb: int) \
            -> Tuple[int, int]:
        """
        For __str__(): ``row, col`` (``y, x``) coordinates.
        """
        sx, sy = self._pstr_cell_sides()
        gap = self.VISUAL_INEQ_GAP
        x_base = col_zb * (sx + gap)
        y_base = row_zb * (sy + gap)
        x_offset = digit_zb % sx
        y_offset = digit_zb // sx
        return (y_base + y_offset), (x_base + x_offset)

    def _pstr_ineq_down_row_col(self, row_zb: int, col_zb: int) \
            -> Tuple[int, int]:
        """
        For __str__(): coordinates of an inequality between two rows.
        The cell specified is the upper.
        """
        sx, sy = self._pstr_cell_sides()
        gap = self.VISUAL_INEQ_GAP
        x_base = col_zb * (sx + gap)
        y_base = row_zb * (sy + gap)
        x_offset = sx // 2
        visual_ineq_pos_y = 1
        y_offset = sy + visual_ineq_pos_y
        return (y_base + y_offset), (x_base + x_offset)

    def _pstr_ineq_right_row_col(self, row_zb: int, col_zb: int) \
            -> Tuple[int, int]:
        """
        For __str__(): coordinates of an inequality between two columns.
        The cell specified is the left.
        """
        sx, sy = self._pstr_cell_sides()
        gap = self.VISUAL_INEQ_GAP
        x_base = col_zb * (sx + gap)
        y_base = row_zb * (sy + gap)
        visual_ineq_pos_x = sx // 2
        x_offset = sx + visual_ineq_pos_x
        y_offset = sy // 2 - 1
        return (y_base + y_offset), (x_base + x_offset)

    def __str__(self) -> str:
        """
        Returns a visual representation of possibilities.
        """
        sx, sy = self._pstr_cell_sides()
        gap = self.VISUAL_INEQ_GAP
        pnx = self.n * (sx + gap) - gap
        pny = self.n * (sy + gap) - gap

        # Create grid of characters
        strings = []  # type: List[List[str]]
        for _ in range(pny):
            line = []  # type: List[str]
            for _ in range(pnx):
                line.append(SPACE)
            strings.append(line)

        # Data
        for r in range(self.n):
            for c in range(self.n):
                initial_value = self.initial_values_zb[r][c] is not None
                cell_solved = self.n_possibilities(r, c) == 1
                for d_zb in range(self.n):
                    y, x = self._pstr_value_row_col(r, c, d_zb)
                    if self.possible[r][c][d_zb]:
                        txt = str(d_zb + 1)
                    elif initial_value:
                        txt = DISPLAY_INITIAL
                    elif cell_solved:
                        txt = DISPLAY_SOLVED
                    else:
                        txt = DISPLAY_UNKNOWN
                    strings[y][x] = txt

        # Inequalities right
        for r in range(self.n):
            for c in range(self.n - 1):
                y, x = self._pstr_ineq_right_row_col(r, c)
                strings[y][x] = self.inequalities_right[r][c] or SPACE

        # Inequalities down
        for r in range(self.n - 1):
            for c in range(self.n):
                y, x = self._pstr_ineq_down_row_col(r, c)
                strings[y][x] = self.inequalities_down[r][c] or SPACE

        # Done
        return "\n".join("".join(x for x in line) for line in strings)

    # -------------------------------------------------------------------------
    # Calculation helpers
    # -------------------------------------------------------------------------

    def iter_cells_less_than(self, row_zb: int, col_zb: int) \
            -> Generator[Tuple[int, int], None, None]:
        """
        Generates ``row, col`` coordinates of cells less than this one.
        """
        # Cell above?
        if (row_zb > 0 and
                self.inequalities_down[row_zb - 1][col_zb] == TOP_LT_BOTTOM):
            other = row_zb - 1, col_zb
            yield other
            yield from self.iter_cells_less_than(*other)
        # Cell below?
        if (row_zb < self.n - 1 and
                self.inequalities_down[row_zb][col_zb] == BOTTOM_LT_TOP):
            other = row_zb + 1, col_zb
            yield other
            yield from self.iter_cells_less_than(*other)
        # Cell left?
        if (col_zb > 0 and
                self.inequalities_right[row_zb][col_zb - 1] == LEFT_LT_RIGHT):
            other = row_zb, col_zb - 1
            yield other
            yield from self.iter_cells_less_than(*other)
        # Cell right?
        if (col_zb < self.n - 1 and
                self.inequalities_right[row_zb][col_zb] == LEFT_GT_RIGHT):
            other = row_zb, col_zb + 1
            yield other
            yield from self.iter_cells_less_than(*other)

    def iter_cells_greater_than(self, row_zb: int, col_zb: int) \
            -> Generator[Tuple[int, int], None, None]:
        """
        Generates ``row, col`` coordinates of cells greater than this one.
        """
        # Cell above?
        if (row_zb > 0 and
                self.inequalities_down[row_zb - 1][col_zb] == BOTTOM_LT_TOP):
            other = row_zb - 1, col_zb
            yield other
            yield from self.iter_cells_greater_than(*other)
        # Cell below?
        if (row_zb < self.n - 1 and
                self.inequalities_down[row_zb][col_zb] == TOP_LT_BOTTOM):
            other = row_zb + 1, col_zb
            yield other
            yield from self.iter_cells_greater_than(*other)
        # Cell left?
        if (col_zb > 0 and
                self.inequalities_right[row_zb][col_zb - 1] == LEFT_GT_RIGHT):
            other = row_zb, col_zb - 1
            yield other
            yield from self.iter_cells_greater_than(*other)
        # Cell right?
        if (col_zb < self.n - 1 and
                self.inequalities_right[row_zb][col_zb] == LEFT_LT_RIGHT):
            other = row_zb, col_zb + 1
            yield other
            yield from self.iter_cells_greater_than(*other)

    def n_less_than(self, row_zb: int, col_zb: int) -> int:
        """
        Number of other cells that this cell is less than.
        """
        return len(list(self.iter_cells_less_than(row_zb, col_zb)))

    def n_greater_than(self, row_zb: int, col_zb: int) -> int:
        """
        Number of other cells that this cell is less than.
        """
        return len(list(self.iter_cells_greater_than(row_zb, col_zb)))

    def min_digit_zb_possible(self, row_zb: int, col_zb: int) -> int:
        """
        The minimum (zero-based!) digit possible.
        """
        return min(self.possible_digits(row_zb, col_zb))

    def max_digit_zb_possible(self, row_zb: int, col_zb: int) -> int:
        """
        The maximum (zero-based!) digit possible.
        """
        return max(self.possible_digits(row_zb, col_zb))

    # -------------------------------------------------------------------------
    # Tactics
    # -------------------------------------------------------------------------

    def _eliminate_simple(self) -> bool:
        """
        Where a cell is known, eliminate other possibilities in its row, cell,
        and 3x3 box.

        Returns: improved?
        """
        log.debug("Eliminating, simple...")
        source = "eliminate_simple"
        improved = False
        for row in range(self.n):
            for col in range(self.n):
                d_possible = self.possible_digits(row, col)
                if len(d_possible) == 1:
                    d = d_possible[0]
                    imp_row = self._eliminate_from_row(row, [col], [d],
                                                       source=source)
                    imp_col = self._eliminate_from_col([row], col, [d],
                                                       source=source)
                    improved = improved or imp_row or imp_col
        return improved

    def _eliminate_inequalities(self) -> bool:
        """
        Eliminate possibilities in a simple way based on inequalities.

        Returns: improved?
        """
        source_b = "eliminate_inequalities, basic"
        source_r = "eliminate_inequalities, relative"
        improved = False
        for row in range(self.n):
            for col in range(self.n):
                # -------------------------------------------------------------
                # "If I'm smaller than 2 others, I can't be 4 or 5."
                # "If I'm bigger than 2 others, I can't be 1 or 2."
                # -------------------------------------------------------------
                ng = self.n_greater_than(row, col)
                nl = self.n_less_than(row, col)
                # log.debug(f"row={row + 1}, col={col + 1}, ng={ng}, nl={nl}")

                # e.g. if we are greater than 2 digits (there are 2 digits less
                # than us), we cannot be the bottom two
                for d_zb in range(0, nl):
                    improved = self._eliminate_from_cell(
                        row, col, d_zb, source=source_b) or improved
                # e.g. if we are less than 2 digits (there are 2 digits greater
                # than us), we cannot be the top two
                for d_zb in range(self.n - ng, self.n):
                    improved = self._eliminate_from_cell(
                        row, col, d_zb, source=source_b) or improved

                # -------------------------------------------------------------
                # "I must be smaller than the minimum possible values of all
                # cells that I am smaller than."
                # "I must be bigger (etc.)..."
                # -------------------------------------------------------------
                max_of_bigger_cells = [
                    self.max_digit_zb_possible(bigger_row, bigger_col)
                    for bigger_row, bigger_col in
                    self.iter_cells_greater_than(row, col)
                ]
                i_am_smaller_than = (
                    min(max_of_bigger_cells) if max_of_bigger_cells
                    else self.n
                )
                min_of_smaller_cells = [
                    self.min_digit_zb_possible(smaller_row, smaller_col)
                    for smaller_row, smaller_col in
                    self.iter_cells_less_than(row, col)
                ]
                i_am_bigger_than = (
                    max(min_of_smaller_cells) if min_of_smaller_cells
                    else -1
                )
                if i_am_smaller_than == self.n and i_am_bigger_than == -1:
                    continue  # nothing to do
                # log.debug(
                #     f"row={row + 1}, col={col + 1}: "
                #     f"i_am_smaller_than={i_am_smaller_than + 1}, "
                #     f"i_am_bigger_than={i_am_bigger_than + 1}")
                for d_zb in range(self.n):
                    if d_zb >= i_am_smaller_than or d_zb <= i_am_bigger_than:
                        improved = self._eliminate_from_cell(
                            row, col, d_zb, source=source_r) or improved

        return improved

    def _find_only_possibilities(self) -> bool:
        """
        Where there is only one location for a digit in a row, cell, or box,
        assign it.
        """
        improved = False
        for d in range(self.n):
            # Rows
            for r in range(self.n):
                possible_cols = [c for c in range(self.n)
                                 if self.possible[r][c][d]]
                if len(possible_cols) == 1:
                    source = (
                        f"Only possibility for digit {d + 1} in row {r + 1}")
                    improved = self.assign_digit(
                        r, possible_cols[0], d, source=source) or improved
            # Columns
            for c in range(self.n):
                possible_rows = [r for r in range(self.n)
                                 if self.possible[r][c][d]]
                if len(possible_rows) == 1:
                    source = (
                        f"Only possibility for digit {d + 1} "
                        f"in column {c + 1}")
                    improved = self.assign_digit(
                        possible_rows[0], c, d, source=source) or improved
        return improved

    # -------------------------------------------------------------------------
    # Strategy
    # -------------------------------------------------------------------------

    def eliminate(self) -> bool:
        """
        Eliminates the impossible.

        Returns: improved?
        """
        improved = self._eliminate_simple()
        improved = self._eliminate_inequalities() or improved
        improved = self._find_only_possibilities() or improved
        if improved:
            return improved  # Keep it simple...

        for groupsize in range(2, self.n):
            improved = self._eliminate_groupwise_rows_cols(
                groupsize) or improved
            if improved:
                return improved

        return improved


# =============================================================================
# Futoshiki
# =============================================================================

class Futoshiki(object):
    """
    Represents and solves Futoshiki puzzles.
    """

    def __init__(self, string_version: str, n: int = 5) -> None:
        """
        Args:
            string_version:
                String representation of the puzzle. Rules as below.
            n:
                Number of rows, columns, and digits.

        - Initial/terminal blank lines are ignored
        - Use numbers 1-5 for known cells.
        - ``.`` or ``?`` represents an unknown cell.
        - ``<``, ``>``, ``v`` or ``V``, and ``^`` represent inequalities.
        - one space (and one line) between each cell.
        """
        # Create data structure
        self.n = n
        self.solved = False
        self.problem_data = [
            [
                UNKNOWN for _col_zb in range(n)
            ] for _row_zb in range(n)
        ]
        self.solution_data = [
            [
                UNKNOWN for _col_zb in range(n)
            ] for _row_zb in range(n)
        ]
        self.inequality_right = [
            [
                None
                for _col_zb in range(n - 1)
            ] for _row_zb in range(n)
        ]  # type: List[List[Optional[str]]]
        # ... index as self.inequality_right[row_zb][col_zb]
        self.inequality_down = [
            [
                None
                for _col_zb in range(n)
            ] for _row_zb in range(n - 1)
        ]  # type: List[List[Optional[str]]]
        # ... index as self.inequality_down[row_zb][col_zb]
        self.working = []  # type: List[str]

        # Check input is basically sound
        lines = string_version.splitlines()
        if not lines:
            raise ValueError("No data")

        # Remove comments
        lines = [line for line in lines if not line.startswith(HASH)]

        # Remove early blank lines
        first_real_line = next(i for i, line in enumerate(lines) if line)
        lines = lines[first_real_line:]

        n_with_gaps = n + n - 1  # e.g. 5 for cells, 4 for inequalities
        lines = lines[:n_with_gaps]
        if len(lines) < n_with_gaps:
            raise ValueError(f"Must have at least {n_with_gaps} lines; "
                             f"found {len(lines)}, which are:\n"
                             f"{lines}")

        row_cell_strings = lines[::2]
        row_ineq_strings = lines[1::2]
        # https://stackoverflow.com/questions/4988002/shortest-way-to-slice-even-odd-lines-from-a-python-array/4988012  # noqa

        # log.critical(f"string_version:\n{string_version}")
        # log.critical(f"lines: {lines}")
        # log.critical(f"row_cell_strings: {row_cell_strings}")
        # log.critical(f"row_ineq_strings: {row_ineq_strings}")

        # Read user's input.
        for row_zb in range(n):
            cell_str = row_cell_strings[row_zb]
            assert len(cell_str) == n_with_gaps, (
                f"Cell data row has wrong length: should be {n_with_gaps}, "
                f"but is {len(cell_str)} ({cell_str!r})"
            )
            cell_data = cell_str[::2]
            cell_ineq = cell_str[1::2]
            for col_zb in range(n):
                self.problem_data[row_zb][col_zb] = cell_data[col_zb]
                if col_zb < n - 1 and col_zb < len(cell_ineq):
                    ineq = cell_ineq[col_zb]
                    assert ineq in (LEFT_LT_RIGHT, LEFT_GT_RIGHT, SPACE), (
                        f"Bad horizontal inequality: {ineq!r}"
                    )
                    if ineq != SPACE:
                        self.inequality_right[row_zb][col_zb] = ineq
            if row_zb < n - 1:
                ineq_str = row_ineq_strings[row_zb][::2]
                for col_zb in range(n):
                    if col_zb < len(ineq_str):
                        ineq = ineq_str[col_zb]
                        assert (ineq == TOP_LT_BOTTOM or
                                ineq in BOTTOM_LT_TOP_POSSIBLES or
                                ineq == SPACE), (
                            f"Bad vertical inequality: {ineq!r}"
                        )
                        if ineq != SPACE:
                            self.inequality_down[row_zb][col_zb] = (
                                TOP_LT_BOTTOM if ineq == TOP_LT_BOTTOM
                                else BOTTOM_LT_TOP
                            )

    # -------------------------------------------------------------------------
    # String representations
    # -------------------------------------------------------------------------

    def __str__(self) -> str:
        return self.solution_str() if self.solved else self.problem_str()

    def problem_str(self) -> str:
        """
        Creates the string representation of the problem.
        """
        return self._make_string(self.problem_data)

    def solution_str(self) -> str:
        """
        Creates the string representation of the problem.
        """
        return self._make_string(self.solution_data)

    def _make_string(self, data: List[List[str]]) -> str:
        x = ""
        for row_zb in range(self.n):
            for col_zb in range(self.n):
                # Cell, inequality, cell, inequality...
                x += data[row_zb][col_zb]
                if col_zb < self.n - 1:
                    x += self.inequality_right[row_zb][col_zb] or SPACE
            if row_zb < self.n - 1:
                x += NEWLINE
            if row_zb < self.n - 1:
                # Row-to-row inequalities
                for col_zb in range(self.n):
                    if col_zb > 0:
                        x += SPACE
                    x += self.inequality_down[row_zb][col_zb] or SPACE
                x += NEWLINE
        return x

    # -------------------------------------------------------------------------
    # Solve via integer programming
    # -------------------------------------------------------------------------

    def solve(self) -> None:
        """
        Solves the problem, writing to :attr:`solved` and
        :attr:`solution_data`.
        """
        if self.solved:
            log.info("Already solved")
            return

        m = Model("Futoshiki solver")
        n = self.n

        # BINARY or INTEGER?
        # - If BINARY, the tricky bit is specifying inequalities.
        #
        #   - "Only one digit per row", "... per column", and "... per cell"
        #     are all easy.
        #   - Achieved.
        #
        # - If INTEGER, the tricky bit is specifying row/column constraints
        #   (e.g. "only one of each digit per row"). In mip, we can specify
        #   "==", ">=", and "<=", but not "!=" (:class:`mip.model.Var` does not
        #   provide :meth:`__ne__`).
        #
        #   - Not achieved. Abandoned.

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Variables
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        x = [
            [
                [
                    m.add_var(f"x(row_zb={r}, col_zb={c}, digit_zb={d})",
                              var_type=BINARY)
                    for d in range(n)
                ] for c in range(n)
            ] for r in range(n)
        ]  # index as: x[row_zb][col_zb][digit_zb]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Constraint helper functions
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        def implement_less_than(r1: int, c1: int, r2: int, c2: int) -> None:
            """
            Implements the constraint value[r1, c1] < value[r2, c2].
            """
            nonlocal m
            m += (
                xsum(x[r1][c1][d] * d for d in range(n))
                <= xsum(x[r2][c2][d] * d for d in range(n)) - 1
            )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Constraints
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # One digit per cell
        for r in range(n):
            for c in range(n):
                m += xsum(x[r][c][d] for d in range(n)) == 1, f"cell(r={r},c={c})"  # noqa
        for d in range(n):
            # One of each digit per row
            for r in range(n):
                m += xsum(x[r][c][d] for c in range(n)) == 1, f"row(r={r},d={d})"  # noqa
            # One of each digit per column
            for c in range(n):
                m += xsum(x[r][c][d] for r in range(n)) == 1, f"col(r={c},d={d})"  # noqa
        # Horizontal inequalities
        for r in range(n):
            for c in range(n - 1):
                ineq = self.inequality_right[r][c]
                if ineq == LEFT_LT_RIGHT:
                    implement_less_than(r, c, r, c + 1)
                elif ineq == LEFT_GT_RIGHT:
                    implement_less_than(r, c + 1, r, c)
        # Vertical inequalities
        for r in range(n - 1):
            for c in range(n):
                ineq = self.inequality_down[r][c]
                if ineq == TOP_LT_BOTTOM:
                    implement_less_than(r, c, r + 1, c)
                elif ineq == BOTTOM_LT_TOP:
                    implement_less_than(r + 1, c, r, c)
        # Starting values
        for r in range(n):
            for c in range(n):
                if self.problem_data[r][c] != UNKNOWN:
                    d_zb = int(self.problem_data[r][c]) - 1
                    m += x[r][c][d_zb] == 1

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Solve
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        m.optimize()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Read out answers
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        debug_model_vars(m)
        if m.num_solutions:
            self.solved = True
            self.working.append("Solved via integer programming method")
            for r in range(n):
                for c in range(n):
                    for d_zb in range(n):
                        if x[r][c][d_zb].x > ALMOST_ONE:
                            self.solution_data[r][c] = str(d_zb + 1)
                            break

    # -------------------------------------------------------------------------
    # Solve via puzzle logic and show working
    # -------------------------------------------------------------------------

    def solve_logic(self, no_guess: bool = False) -> None:
        """
        Solve via conventional logic, and save our working.
        """
        if self.solved:
            log.info("Already solved")
            return

        p = FutoshikiPossibilities(n=self.n)  # start from scratch

        # Set starting values.
        n = self.n
        initial_values = [
            [
                None for _c in range(n)
            ] for _r in range(n)
        ]  # type: List[List[Optional[int]]]
        for r in range(n):
            for c in range(n):
                known_digit_str = self.problem_data[r][c]
                if known_digit_str != UNKNOWN:
                    initial_values[r][c] = int(known_digit_str)
        p.set_initial_values(
            initial_values=initial_values,
            inequalities_down=self.inequality_down,
            inequalities_right=self.inequality_right,
        )

        # Eliminate and iterate
        p.solve(no_guess=no_guess)

        # Read out answers
        self.solved = True
        for r in range(n):
            for c in range(n):
                d_zb = next(i for i, v in enumerate(p.possible[r][c]) if v)
                self.solution_data[r][c] = str(d_zb + 1)


# =============================================================================
# main
# =============================================================================

def main() -> None:
    """
    Command-line entry point.
    """
    cmd_demo = "demo"
    cmd_solve = "solve"
    cmd_working = "working"

    help_filename = (
        "Puzzle filename to read. Must contain text in format as above.")

    parser = argparse.ArgumentParser(
        formatter_class=RawDescriptionArgumentDefaultsHelpFormatter,
        description=(
            f"Solve Futoshiki puzzles. Format is:\n\n"
            f"{DEMO_FUTOSHIKI_1}"
        )
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Be verbose")
    subparsers = parser.add_subparsers(
        dest="command",
        # required=True,  # needs Python 3.7
        help="Append --help for more help")

    parser_solve = subparsers.add_parser(
        cmd_solve, help="Solve from a file (via integer programming)")
    parser_solve.add_argument(
        "filename", type=str, default="", help=help_filename)

    parser_working = subparsers.add_parser(
        cmd_working,
        help="Solve from a file, via puzzle logic, showing working")
    parser_working.add_argument(
        "filename", type=str, default="", help=help_filename)
    parser_working.add_argument(
        "--noguess", action="store_true", help="Prevent guessing")

    _parser_demo = subparsers.add_parser(cmd_demo, help="Run demo")

    args = parser.parse_args()
    main_only_quicksetup_rootlogger(level=logging.DEBUG if args.verbose
                                    else logging.INFO)

    if not args.command:
        print("Must specify command")
        sys.exit(EXIT_FAILURE)
    if args.command == cmd_demo:
        problem = Futoshiki(DEMO_FUTOSHIKI_1)
        log.info(f"Solving:\n{problem}")
        problem.solve()
    else:
        log.info(f"Reading {args.filename}")
        with open(args.filename, "rt") as f:
            string_version = f.read()
        problem = Futoshiki(string_version)
        log.info(f"Solving:\n{problem}")
        if args.command == cmd_solve:
            problem.solve()
        else:
            problem.solve_logic(no_guess=args.noguess)
    log.info(f"Answer:\n{problem}")
    sys.exit(EXIT_SUCCESS)


# =============================================================================
# Command-line entry point
# =============================================================================

if __name__ == "__main__":
    run_guard(main)
