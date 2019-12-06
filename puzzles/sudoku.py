#!/usr/bin/env python

"""
sudoku.py

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

**Solves Sudoku puzzles.**

"""


import argparse
from copy import deepcopy
from itertools import combinations
import logging
import sys
from typing import Iterable, List, Optional, Sequence, Tuple

from cardinal_pythonlib.argparse_func import RawDescriptionArgumentDefaultsHelpFormatter  # noqa
from cardinal_pythonlib.logs import main_only_quicksetup_rootlogger
from mip import BINARY, Model, xsum

from common import debug_model_constraints, debug_model_vars

log = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

UNKNOWN = "."
NEWLINE = "\n"
SPACE = " "
COMMENT = "#"
HASH = "#"
DISPLAY_INITIAL = "■"
DISPLAY_UNKNOWN = "·"
DISPLAY_SOLVED = "♦"
ALMOST_ONE = 0.99
N = 9
T = 3

DEMO_SUDOKU_1 = """
# Coton 54, Coton Community News Dec 2019-Jan 2020

... ... ...
..2 3.1 45.
.1. ... .6.

.47 .5. 38.
... 7.3 ...
.36 ... 14.

.7. ... .9.
.91 4.5 6..
... ..9 ...
"""


# =============================================================================
# Exceptions
# =============================================================================

class SolutionFailure(Exception):
    pass


# =============================================================================
# SudokuPossibilities
# =============================================================================

class SudokuPossibilities(object):
    """
    Represents Sudoku possibilities, for a logic-based eliminator.

    Based on SuDokuSolver (C++, 2005) by me, at
    http://egret.psychol.cam.ac.uk/code/index.html.

    Strategy:

    1.  Attempt to solve current state of puzzle analytically.

    2.  If unsolved, guess one value and attempt to solve that, recursively,
        until all possible uncertain values have been guessed or a solution has
        been found.

    Tactics:

    1.  Analyse using a three-dimensional matrix (x, y, digit) of boolean
        variables indicating the possibility of a digit being in a particular
        location or cell.

    2.  Where a digit is known, eliminate it as a possibility from other cells
        in the same row/column/3x3 square.

    3.  Where there is only one possible cell for a digit in a
        row/column/square, confirm that digit by eliminating other possible
        digits from that cell.

    4.  Where there are n identical sets of n possible locations for n digits
        (e.g. for n=2: digit 4 can only be in locations A or B, and digit 8 can
        only be in locations A or B), eliminate all other possible digits from
        those locations. This is an extension of the previous tactic (for which
        n=1).

    5.  Repeat these tactics until the puzzle is solved or there is no
        improvement in knowledge.

    """
    def __init__(self, other: "SudokuPossibilities" = None) -> None:
        """
        Initialize with "everything is possible", or copy from another.
        """
        self.guess_level = 0
        if other:
            self.initial_values_zb = deepcopy(other.initial_values_zb)
            self.possible = deepcopy(other.possible)
            self.working = deepcopy(other.working)
        else:
            self.initial_values_zb = [
                [
                    None for _c in range(N)
                ] for _r in range(N)
            ]  # type: List[List[Optional[int]]]
            # ... index as: self.initial[row_zb][col_zb]
            self.possible = [
                [
                    [
                        True for _d in range(N)
                    ] for _c in range(N)
                ] for _r in range(N)
            ]  # type: List[List[List[bool]]]
            # ... index as: self.possible[row_zb][col_zb][digit_zb]
            self.working = []  # type: List[str]

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------

    def set_initial_values(self,
                           initial_values: List[List[Optional[int]]]) -> None:
        """
        Set up the initial values. The incoming digits are genuine (one-based),
        not zero-based.
        """
        assert len(initial_values) == N
        for r in range(N):
            row = initial_values[r]
            assert len(row) == N
            for c in range(N):
                value = row[c]
                if value is not None:
                    d_zb = value - 1
                    self.initial_values_zb[r][c] = d_zb
                    self.assign_digit(r, c, d_zb, source="Starting value")
                else:
                    self.initial_values_zb[r][c] = None

    # -------------------------------------------------------------------------
    # Visuals
    # -------------------------------------------------------------------------

    def note(self, msg: str) -> None:
        """
        Save some working.
        """
        self.working.append(msg)
        log.info(msg)

    @staticmethod
    def _pstr_row_col(row_zb: int, col_zb: int, digit_zb: int) \
            -> Tuple[int, int]:
        """
        For __str__(): ``row, col`` (``y, x``) coordinates.
        """
        x_base = col_zb * (T + 1)
        y_base = row_zb * (T + 1)
        x_offset = digit_zb % T
        y_offset = digit_zb // T
        return (y_base + y_offset), (x_base + x_offset)

    def __str__(self) -> str:
        """
        Returns a visual representation of possibilities.
        """
        pn = N * 4 - 1

        # Create grid of characters
        # BEWARE: DO NOT DO THIS: strings = [[" "] * pn] * pn
        # ... it creates copies of identical strings/lists, so when you
        # modify one, it modifies them all.
        strings = []  # type: List[List[str]]
        for _ in range(pn):
            line = []  # type: List[str]
            for _ in range(pn):
                line.append(" ")
            strings.append(line)

        # Prettify
        cell_boundaries = ((T + 1) * T - 1, (T + 1) * (T * 2) - 1)
        for r in cell_boundaries:
            for i in range(pn):
                strings[r][i] = "-"
        for c in cell_boundaries:
            for i in range(pn):
                strings[i][c] = "|"
        for r in cell_boundaries:
            for c in cell_boundaries:
                strings[r][c] = "+"

        # Data
        for r in range(N):
            for c in range(N):
                initial_value = self.initial_values_zb[r][c] is not None
                cell_solved = self.n_possibilities(r, c) == 1
                for d_zb in range(N):
                    y, x = self._pstr_row_col(r, c, d_zb)
                    if self.possible[r][c][d_zb]:
                        txt = str(d_zb + 1)
                    elif initial_value:
                        txt = DISPLAY_INITIAL
                    elif cell_solved:
                        txt = DISPLAY_SOLVED
                    else:
                        txt = DISPLAY_UNKNOWN
                    strings[y][x] = txt
        return "\n".join("".join(x for x in line) for line in strings)

    # -------------------------------------------------------------------------
    # Calculation helpers
    # -------------------------------------------------------------------------

    def n_unknown_cells(self) -> int:
        """
        Number of unsolved cells. Maximum is 81.
        """
        return sum(
            1 if self.n_possibilities(r, c) != 1 else 0
            for r in range(N)
            for c in range(N)
        )

    def n_possibilities_overall(self) -> int:
        """
        Number of row/cell/digit possibilities overall.
        Minimum is 81 (solved). Maximum is 729.
        """
        return sum(
            self.n_possibilities(r, c)
            for r in range(N)
            for c in range(N)
        )

    def n_possibilities(self, row_zb: int, col_zb: int) -> int:
        """
        Number of possible digits for a cell.
        """
        return sum(self.possible[row_zb][col_zb])

    def possible_digits(self, row_zb: int, col_zb: int) -> List[int]:
        """
        Returns possible digits, in ZERO-BASED format, for a given cell.
        They are always returned in ascending order.
        """
        return list(
            d for d, v in enumerate(self.possible[row_zb][col_zb]) if v
        )

    def solved(self) -> bool:
        """
        Are we there yet?
        """
        for r in range(N):
            for c in range(N):
                if self.n_possibilities(r, c) != 1:
                    return False
        return True

    @staticmethod
    def square_extremes_for_cell(row_zb: int, col_zb: int) \
            -> Tuple[int, int, int, int]:
        """
        Defines the boundaries of the 3x3 square containing a particular cell.

        Returns ``row_min, row_max, col_min, col_max``.
        """
        row_min = (row_zb // T) * T
        row_max = row_min + T - 1
        col_min = (col_zb // T) * T
        col_max = col_min + T - 1
        return row_min, row_max, col_min, col_max

    @staticmethod
    def top_left_cell_for_square(square_zb: int) -> Tuple[int, int]:
        """
        Returns ``row_zb, col_zb`` for the top-left cell in a 3x3 square
        (numbered from 0 to N - 1).
        """
        row = (square_zb // T) * T
        col = (square_zb % T) * T
        return row, col

    @classmethod
    def square_extremes_for_square(cls, square_zb: int) \
            -> Tuple[int, int, int, int]:
        """
        Defines the boundaries of the 3x3 square, numbered from 0 to (N - 1).

        Returns ``row_min, row_max, col_min, col_max``.
        """
        row, col = cls.top_left_cell_for_square(square_zb)
        return cls.square_extremes_for_cell(row, col)

    # -------------------------------------------------------------------------
    # Computations
    # -------------------------------------------------------------------------

    def assign_digit(self, row_zb: int, col_zb: int, digit_zb: int,
                     source: str = "?") -> bool:
        """
        Assign a digit to a cell.

        Returns: improved?
        """
        improved = False
        for d in range(N):
            if d == digit_zb:
                assert self.possible[row_zb][col_zb][d], (
                    f"{source}: Assigning digit {digit_zb + 1} to "
                    f"row={row_zb + 1}, col={col_zb + 1} "
                    f"where that is known to be impossible"
                )
            else:
                improved = improved or self.possible[row_zb][col_zb][d]
                self.possible[row_zb][col_zb][d] = False
        if improved:
            self.note(f"{source}: Assigning digit {digit_zb + 1} to "
                      f"row={row_zb + 1}, col={col_zb + 1}")
        return improved

    def _restrict_cells(self, cells: Sequence[Tuple[int, int]],
                        digits_zb_to_keep: Sequence[int],
                        source: str = "?") -> bool:
        """
        Restricts a cell or cells to a specific set of digits.
        Cells are specified as ``row_zb, col_zb`` tuples.
        Digits are also zero-based.
        """
        improved = False
        for row, col in cells:
            initial_n = self.n_possibilities(row, col)
            for d in range(N):
                if d in digits_zb_to_keep:
                    continue
            final_n = self.n_possibilities(row, col)
            if final_n == 0:
                raise SolutionFailure()
            this_cell_improved = final_n < initial_n
            if this_cell_improved and final_n == 1:  # unlikely!
                self.note(f"{source}: "
                          f"row={row + 1}, col={col + 1} must be "
                          f"{self.possible_digits(row, col)[0] + 1}")
            improved = improved or this_cell_improved
        return improved

    def _eliminate_from_cell(self, row_zb: int, col_zb: int,
                             digit_zb: int, source: str = "?") -> bool:
        """
        Eliminates a digit as a possibility from a cell.
        Returns: improved?
        """
        if not self.possible[row_zb][col_zb][digit_zb]:
            return False
        self.possible[row_zb][col_zb][digit_zb] = False
        n = self.n_possibilities(row_zb, col_zb)
        if n == 0:
            raise SolutionFailure()
        if n == 1:  # improved down to 1
            self.note(f"{source}: "
                      f"row={row_zb + 1}, col={col_zb + 1} must be "
                      f"{self.possible_digits(row_zb, col_zb)[0] + 1}")
        return True

    def _eliminate_from_row(self, row_zb: int, except_cols_zb: List[int],
                            digits_zb_to_eliminate: Iterable[int],
                            source: str = "?") -> bool:
        """
        Eliminates a digit or digits from all cells in a row except the one(s)
        specified.

        Returns: Improved?
        """
        improved = False
        for c in range(N):
            if c in except_cols_zb:
                continue
            for d in digits_zb_to_eliminate:
                zsource = f"{source}: eliminate_from_row, eliminating {d + 1}"
                improved = self._eliminate_from_cell(
                    row_zb, c, d, source=zsource) or improved
        return improved

    def _eliminate_from_col(self, except_rows_zb: List[int], col_zb: int,
                            digits_zb_to_eliminate: Iterable[int],
                            source: str = "?") -> bool:
        """
        Eliminates a digit or digits from all cells in a column except the
        one(s) specified.

        Returns: Improved?
        """
        improved = False
        for r in range(N):
            if r in except_rows_zb:
                continue
            for d in digits_zb_to_eliminate:
                zsource = f"{source}: eliminate_from_col, eliminating {d + 1}"
                improved = self._eliminate_from_cell(
                    r, col_zb, d, source=zsource) or improved
        return improved

    def _eliminate_from_square(self, except_cells_zb: List[Tuple[int, int]],
                               digits_zb_to_eliminate: Iterable[int],
                               source: str = "?") -> bool:
        """
        Eliminates a digit or digits from all cells in a 3x3 "square" except
        the cell(s) specified.

        Returns: Improved?
        """
        row_min, row_max, col_min, col_max = self.square_extremes_for_cell(
            row_zb=min(t[0] for t in except_cells_zb),
            col_zb=min(t[1] for t in except_cells_zb)
        )
        assert (row_min, row_max, col_min, col_max) == (
            self.square_extremes_for_cell(
                row_zb=max(t[0] for t in except_cells_zb),
                col_zb=max(t[1] for t in except_cells_zb)
            )
        ), "except_cells_zb spans more than one 3x3 square!"
        improved = False
        for r in range(row_min, row_max + 1):
            for c in range(col_min, col_max + 1):
                if (r, c) in except_cells_zb:
                    continue
                for d in digits_zb_to_eliminate:
                    zsource = (
                        f"{source}: eliminate_from_square, eliminating {d + 1}"
                    )
                    improved = self._eliminate_from_cell(
                        r, c, d, source=zsource) or improved
        return improved

    def _eliminate_simple(self) -> bool:
        """
        Where a cell is known, eliminate other possibilities in its row, cell,
        and 3x3 square.

        Returns: improved?
        """
        log.debug("Eliminating, simple...")
        source = "eliminate_simple"
        improved = False
        for row in range(N):
            for col in range(N):
                d_possible = self.possible_digits(row, col)
                if len(d_possible) == 1:
                    d = d_possible[0]
                    imp_row = self._eliminate_from_row(row, [col], [d],
                                                       source=source)
                    imp_col = self._eliminate_from_col([row], col, [d],
                                                       source=source)
                    imp_square = self._eliminate_from_square([(row, col)], [d],
                                                             source=source)
                    improved = improved or imp_row or imp_col or imp_square
        return improved

    def _find_only_possibilities(self) -> bool:
        """
        Where there is only one location for a digit in a row, cell, or square,
        assign it.
        """
        improved = False
        for d in range(N):
            source = f"find_only_possibilities, digit={d + 1}"
            # Rows
            for r in range(N):
                possible_cols = [c for c in range(N) if self.possible[r][c][d]]
                if len(possible_cols) == 1:
                    improved = self.assign_digit(
                        r, possible_cols[0], d, source=source) or improved
            # Columns
            for c in range(N):
                possible_rows = [r for r in range(N) if self.possible[r][c][d]]
                if len(possible_rows) == 1:
                    improved = self.assign_digit(
                        possible_rows[0], c, d, source=source) or improved
            # Squares
            for s in range(N):
                row_min, row_max, col_min, col_max = \
                    self.square_extremes_for_square(s)
                possible_cells = [
                    (r, c)
                    for r in range(row_min, row_max + 1)
                    for c in range(col_min, col_max + 1)
                    if self.possible[r][c][d]
                ]
                if len(possible_cells) == 1:
                    improved = self.assign_digit(
                        possible_cells[0][0],
                        possible_cells[0][1], d, source=source) or improved
        return improved

    def _eliminate_constraintwise(self) -> bool:
        """
        Example: if we don't know where the 5 is in a particular 3x3 square,
        but we know that it's in the first row, then we can eliminate "5" from
        that row in all other cells.

        Returns: improved?
        """
        log.debug(f"Eliminating, constraint-wise...")
        improved = False
        for s in range(N):
            row_min, row_max, col_min, col_max = \
                self.square_extremes_for_square(s)
            rownums = list(range(row_min, row_max + 1))
            colnums = list(range(col_min, col_max + 1))
            for d in range(N):
                source = (
                    f"eliminate_constraintwise from square {s + 1}, "
                    f"eliminating {d + 1}"
                )
                possible_cells_for_digit = [
                    (r, c)
                    for r in rownums
                    for c in colnums
                    if d in self.possible_digits(r, c)
                ]
                possible_rows = list(set(
                    rc[0] for rc in possible_cells_for_digit))
                possible_cols = list(set(
                    rc[1] for rc in possible_cells_for_digit))
                if len(possible_rows) == 1:
                    improved = self._eliminate_from_row(
                        row_zb=possible_rows[0],
                        except_cols_zb=colnums,
                        digits_zb_to_eliminate=[d],
                        source=source
                    ) or improved
                if len(possible_cols) == 1:
                    improved = self._eliminate_from_col(
                        col_zb=possible_cols[0],
                        except_rows_zb=rownums,
                        digits_zb_to_eliminate=[d],
                        source=source
                    ) or improved
        return improved

    def _eliminate_groupwise(self, groupsize: int) -> bool:
        """
        Scan through all numbers in groups of n. If there are exactly n
        possible cells for each of those n numbers in a row/column/square, and
        those possibilities are all the same, eliminate other possible digits
        for those cells.

        Example: in row 1, if digit 4 could be in columns 5/7 only, and digit 8
        could be in columns 5/7 only, then either '4' goes in column 5 and '8'
        goes in column 7, or vice versa - but these cells can't possibly
        contain anything other than '4' or '8'.

        We've done n=1 above, so our caller will begin with n=2. No point going
        to n=9 (that means we have no information!) or n=8 (if 8 wholly
        uncertain digits, then 1 completely certain digit, and that situation
        was dealt with above, in :meth:`_eliminate_simple`).

        Returns: improved?
        """
        log.debug(f"Eliminating, groupwise, group size {groupsize}...")
        improved = False
        for digit_combo in combinations(range(N), r=groupsize):
            pretty_digits = [d + 1 for d in digit_combo]
            source = f"eliminate_groupwise, digits={pretty_digits}"
            # Rows
            combo_set = set(digit_combo)
            for row in range(N):
                columns_with_only_these_digits = [
                    col for col in range(N)
                    if combo_set.issubset(self.possible_digits(row, col))
                ]
                columns_with_any_of_these_digits = [
                    col for col in range(N)
                    if combo_set.intersection(self.possible_digits(row, col))
                ]
                if (len(columns_with_only_these_digits) == groupsize and
                        len(columns_with_any_of_these_digits) == groupsize):
                    prettycol = [c + 1 for c in columns_with_only_these_digits]
                    log.debug(
                        f"In row {row + 1}, only columns {prettycol} "
                        f"could contain digits {pretty_digits}")
                    improved = self._eliminate_from_row(
                        row_zb=row,
                        except_cols_zb=columns_with_only_these_digits,
                        digits_zb_to_eliminate=digit_combo,
                        source=source
                    ) or improved
                    improved = self._restrict_cells(
                        cells=[(row, c)
                               for c in columns_with_only_these_digits],
                        digits_zb_to_keep=digit_combo,
                        source=source
                    ) or improved
            # Columns
            for col in range(N):
                rows_with_only_these_digits = [
                    row for row in range(N)
                    if combo_set.issubset(self.possible_digits(row, col))
                ]
                rows_with_any_of_these_digits = [
                    row for row in range(N)
                    if combo_set.intersection(self.possible_digits(row, col))
                ]
                if (len(rows_with_only_these_digits) == groupsize and
                        len(rows_with_any_of_these_digits) == groupsize):
                    prettyrow = [r + 1 for r in rows_with_only_these_digits]
                    log.debug(
                        f"In column {col + 1}, only rows {prettyrow} "
                        f"could contain digits {pretty_digits}")
                    improved = self._eliminate_from_col(
                        col_zb=col,
                        except_rows_zb=rows_with_only_these_digits,
                        digits_zb_to_eliminate=digit_combo,
                        source=source
                    ) or improved
                    improved = self._restrict_cells(
                        cells=[(r, col)
                               for r in rows_with_only_these_digits],
                        digits_zb_to_keep=digit_combo,
                        source=source
                    ) or improved
            # Squares
            for square in range(N):
                row_min, row_max, col_min, col_max = \
                    self.square_extremes_for_square(square)
                cells_with_only_these_digits = [
                    (row, col)
                    for row in range(row_min, row_max + 1)
                    for col in range(col_min, col_max + 1)
                    if combo_set.issubset(self.possible_digits(row, col))
                ]
                cells_with_any_of_these_digits = [
                    (row, col)
                    for row in range(row_min, row_max + 1)
                    for col in range(col_min, col_max + 1)
                    if combo_set.intersection(self.possible_digits(row, col))
                ]
                if (len(cells_with_only_these_digits) == groupsize and
                        len(cells_with_any_of_these_digits) == groupsize):
                    prettycell = [(r + 1, c+1)
                                  for r, c in cells_with_only_these_digits]
                    log.debug(
                        f"In 3x3 square #{square + 1}, "
                        f"only cells {prettycell} "
                        f"could contain digits {pretty_digits}")
                    improved = self._eliminate_from_square(
                        except_cells_zb=cells_with_only_these_digits,
                        digits_zb_to_eliminate=digit_combo,
                        source=source
                    ) or improved
                    improved = self._restrict_cells(
                        cells=cells_with_only_these_digits,
                        digits_zb_to_keep=digit_combo,
                        source=source
                    ) or improved
        return improved

    def eliminate(self) -> bool:
        """
        Eliminates the impossible.

        Returns: improved?
        """
        improved = self._eliminate_simple()
        improved = self._find_only_possibilities() or improved
        if improved:
            return improved  # Keep it simple...

        improved = self._eliminate_constraintwise()
        if improved:
            return improved  # Keep it simple...

        for groupsize in range(2, N):  # from 2 to 8
            improved = self._eliminate_groupwise(groupsize) or improved
            if improved:
                return improved

        return improved

    def solve(self) -> None:
        """
        Solves by elimination.
        """
        iteration = 0
        while not self.solved():
            log.debug(
                f"Iteration {iteration}. "
                f"Unsolved cells: {self.n_unknown_cells()}. "
                f"Possible digit assignments: "
                f"{self.n_possibilities_overall()} (target {N * N}). "
                f"Possibilities:\n{self}")
            improved = self.eliminate()
            if not improved:
                self.note("No improvement; guessing")
                log.debug(f"Possibilities:\n{self}")
                self.guess()
            iteration += 1

    def guess(self) -> None:
        """
        Implements the "guess" method!

        Using this means that the algorithm has deficiencies (or the puzzle
        does).
        """
        for r in range(N):
            for c in range(N):
                digits = self.possible_digits(r, c)
                if len(digits) == 1:
                    continue
                for d in digits:
                    p = SudokuPossibilities(self)
                    p.guess_level = self.guess_level + 1
                    log.warning("Guessing")
                    p.assign_digit(r, c, d,
                                   source=f"guess level {p.guess_level}")
                    try:
                        p.solve()
                        assert p.solved()
                        log.info(f"Guess level {p.guess_level} was good")
                        # no need to copy initial_values_zb
                        self.possible = p.possible
                        self.working = p.working
                        return
                    except SolutionFailure:
                        log.info(
                            f"Bad guess at level {p.guess_level}; moving on")


# =============================================================================
# Sudoku
# =============================================================================

class Sudoku(object):
    """
    Represents and solves Sudoku puzzles.
    """

    def __init__(self, string_version: str) -> None:
        """
        Args:
            string_version:
                String representation of the puzzle. Rules as below.

        - Initial/terminal blank lines are ignored
        - Use numbers 1-9 for known cells.
        - ``.`` represents an unknown cell.
        - one space (and one line) between each cell.
        """
        # Create data structure
        self.solved = False
        self.problem_data = [
            [
                UNKNOWN for _col_zb in range(N)
            ] for _row_zb in range(N)
        ]
        self.solution_data = [
            [
                UNKNOWN for _col_zb in range(N)
            ] for _row_zb in range(N)
        ]
        self.working = []  # type: List[str]

        # Check input is basically sound
        lines = string_version.splitlines()
        if not lines:
            raise ValueError("No data")

        # Remove comments
        lines = [line for line in lines if not line.startswith(HASH)]

        lines = ["".join(line.split())
                 for line in lines if line]  # remove blank lines/columns
        if len(lines) != N:
            raise ValueError(f"Must have {N} active lines; "
                             f"found {len(lines)}, which are:\n"
                             f"{lines}")

        # Read user's input.
        for row_zb in range(N):
            line = lines[row_zb]
            assert len(line) == N, (
                f"Data line has wrong non-blank length: should be {N}, "
                f"but is {len(line)} ({line!r})"
            )
            for col_zb in range(N):
                self.problem_data[row_zb][col_zb] = line[col_zb]

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

    @staticmethod
    def _make_string(data: List[List[str]]) -> str:
        x = ""
        for row_zb in range(N):
            for col_zb in range(N):
                # Cell, inequality, cell, inequality...
                x += data[row_zb][col_zb]
                if col_zb % T == 2:
                    x += SPACE
            if row_zb < N - 1:
                x += NEWLINE
                if row_zb % T == 2:
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

        m = Model("Sudoku solver")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Variables
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        x = [
            [
                [
                    m.add_var(f"x(row={r + 1}, col={c + 1}, digit={d + 1})",
                              var_type=BINARY)
                    for d in range(N)
                ] for c in range(N)
            ] for r in range(N)
        ]  # index as: x[row_zb][col_zb][digit_zb]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Constraints
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # One digit per cell
        for r in range(N):
            for c in range(N):
                m += xsum(x[r][c][d] for d in range(N)) == 1
        for d in range(N):
            # One of each digit per row
            for r in range(N):
                m += xsum(x[r][c][d] for c in range(N)) == 1
            # One of each digit per column
            for c in range(N):
                m += xsum(x[r][c][d] for r in range(N)) == 1
        # One of each digit in each 3x3 square:
        for d in range(N):
            for square_row in range(T):
                for square_col in range(T):
                    row_base = square_row * T
                    col_base = square_col * T
                    m += xsum(
                        x[row_base + row_offset][col_base + col_offset][d]
                        for row_offset in range(T)
                        for col_offset in range(T)
                    ) == 1
        # Starting values
        for r in range(N):
            for c in range(N):
                if self.problem_data[r][c] != UNKNOWN:
                    d_zb = int(self.problem_data[r][c]) - 1
                    m += x[r][c][d_zb] == 1

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Solve
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        debug_model_constraints(m)
        m.optimize()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Read out answers
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        debug_model_vars(m)
        if m.num_solutions:
            self.solved = True
            self.working.append("Solved via integer programming method")
            for r in range(N):
                for c in range(N):
                    for d_zb in range(N):
                        if x[r][c][d_zb].x > ALMOST_ONE:
                            self.solution_data[r][c] = str(d_zb + 1)
                            break
        else:
            log.error("Unable to solve!")

    # -------------------------------------------------------------------------
    # Solve via puzzle logic and show working
    # -------------------------------------------------------------------------

    def solve_logic(self) -> None:
        """
        Solve via conventional logic, and save our working.
        """
        if self.solved:
            log.info("Already solved")
            return

        p = SudokuPossibilities()  # start from scratch

        # Set starting values.
        initial_values = [
            [
                None for _c in range(N)
            ] for _r in range(N)
        ]  # type: List[List[Optional[int]]]
        for r in range(N):
            for c in range(N):
                known_digit_str = self.problem_data[r][c]
                if known_digit_str != UNKNOWN:
                    initial_values[r][c] = int(known_digit_str)
        p.set_initial_values(initial_values)

        # Eliminate and iterate
        p.solve()

        # Read out answers
        self.solved = True
        for r in range(N):
            for c in range(N):
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
            f"Solve Sudoku puzzles. Format is:\n\n"
            f"{DEMO_SUDOKU_1}"
        )
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Be verbose")
    subparsers = parser.add_subparsers(
        dest="command",
        # required=True,  # needs Python 3.7
        help="Append --help for more help")

    parser_solve = subparsers.add_parser(cmd_solve, help="Solve from a file")
    parser_solve.add_argument(
        "filename", type=str, default="", help=help_filename)

    parser_working = subparsers.add_parser(
        cmd_working,
        help="Solve from a file, via puzzle logic, showing working")
    parser_working.add_argument(
        "filename", type=str, default="", help=help_filename)

    _parser_demo = subparsers.add_parser(cmd_demo, help="Run demo")

    args = parser.parse_args()
    main_only_quicksetup_rootlogger(level=logging.DEBUG if args.verbose
                                    else logging.INFO)

    if not args.command:
        print("Must specify command")
        sys.exit(1)
    if args.command == cmd_demo:
        problem = Sudoku(DEMO_SUDOKU_1)
        log.info(f"Solving:\n{problem}")
        problem.solve()
    else:
        assert args.filename, "Must specify parameter: --filename"
        with open(args.filename, "rt") as f:
            string_version = f.read()
        problem = Sudoku(string_version)
        log.info(f"Solving:\n{problem}")
        if args.command == cmd_solve:
            problem.solve()
        else:
            problem.solve_logic()
    log.info(f"Answer:\n{problem}")
    sys.exit(0)


# =============================================================================
# Command-line entry point
# =============================================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.critical(str(e))
        raise
