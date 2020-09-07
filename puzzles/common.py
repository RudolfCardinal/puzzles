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

from copy import deepcopy
from itertools import combinations
import logging
import sys
import traceback
from typing import Callable, List, Iterable, Optional, Sequence, Tuple

from mip import Constr, Model, Var

log = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

UNKNOWN = "."
NEWLINE = "\n"
SPACE = " "
HASH = "#"
DISPLAY_INITIAL = "♦"
DISPLAY_UNKNOWN = "·"
DISPLAY_SOLVED = "■"
ALMOST_ONE = 0.99

EXIT_FAILURE = 1
EXIT_SUCCESS = 0


# =============================================================================
# Exceptions
# =============================================================================

class SolutionFailure(Exception):
    pass


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


# =============================================================================
# Generic helper functions
# =============================================================================

def run_guard(function: Callable[[], None]) -> None:
    try:
        function()
    except Exception as e:
        log.critical(str(e))
        traceback.print_exc()
        sys.exit(EXIT_FAILURE)


# =============================================================================
# CommonPossibilities
# =============================================================================

class CommonPossibilities(object):
    def __init__(self, other: "CommonPossibilities" = None,
                 n: int = 5) -> None:
        """
        Initialize with "everything is possible", or copy from another.
        """
        self.n = n
        if other:
            assert other.n == self.n
            self.guess_level = other.guess_level
            self.initial_values_zb = deepcopy(other.initial_values_zb)
            self.possible = deepcopy(other.possible)
            self.working = deepcopy(other.working)
        else:
            self.guess_level = 0
            self.initial_values_zb = [
                [
                    None for _c in range(n)
                ] for _r in range(n)
            ]  # type: List[List[Optional[int]]]
            # ... index as: self.initial[row_zb][col_zb]
            self.possible = [
                [
                    [
                        True for _d in range(n)
                    ] for _c in range(n)
                ] for _r in range(n)
            ]  # type: List[List[List[bool]]]
            # ... index as: self.possible[row_zb][col_zb][digit_zb]
            self.working = []  # type: List[str]

    def clone(self) -> "CommonPossibilities":
        # noinspection PyTypeChecker
        return self.__class__(other=self, n=self.n)

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------

    def set_initial_values(self,
                           initial_values: List[List[Optional[int]]]) -> None:
        """
        Set up the initial values. The incoming digits are genuine (one-based),
        not zero-based.
        """
        assert len(initial_values) == self.n
        for r in range(self.n):
            row = initial_values[r]
            assert len(row) == self.n
            for c in range(self.n):
                value = row[c]
                if value is not None:
                    d_zb = value - 1
                    self.initial_values_zb[r][c] = d_zb
                    self.assign_digit(r, c, d_zb, source="Starting value")
                else:
                    self.initial_values_zb[r][c] = None

    # -------------------------------------------------------------------------
    # Show your working
    # -------------------------------------------------------------------------

    def note(self, msg: str) -> None:
        """
        Save some working.
        """
        self.working.append(msg)
        log.info(msg)

    # -------------------------------------------------------------------------
    # Calculation helpers
    # -------------------------------------------------------------------------

    def n_unknown_cells(self) -> int:
        """
        Number of unsolved cells. Maximum is 81.
        """
        return sum(
            1 if self.n_possibilities(r, c) != 1 else 0
            for r in range(self.n)
            for c in range(self.n)
        )

    def n_possibilities_overall(self) -> int:
        """
        Number of row/cell/digit possibilities overall.
        Minimum is n^2 (solved). Maximum is n^3.
        """
        return sum(
            self.n_possibilities(r, c)
            for r in range(self.n)
            for c in range(self.n)
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
        for r in range(self.n):
            for c in range(self.n):
                if self.n_possibilities(r, c) != 1:
                    return False
        return True

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
        for d in range(self.n):
            if d == digit_zb:
                assert self.possible[row_zb][col_zb][d], (
                    f"{source}: Assigning digit {digit_zb + 1} to "
                    f"(row={row_zb + 1}, col={col_zb + 1}) "
                    f"where that is known to be impossible"
                )
            else:
                improved = improved or self.possible[row_zb][col_zb][d]
                self.possible[row_zb][col_zb][d] = False
        if improved:
            self.note(f"{source}: Assigning digit {digit_zb + 1} to "
                      f"(row={row_zb + 1}, col={col_zb + 1})")
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
            for d in range(self.n):
                if d not in digits_zb_to_keep:
                    self.possible[row][col][d] = False
            final_n = self.n_possibilities(row, col)
            if final_n == 0:
                raise SolutionFailure()
            this_cell_improved = final_n < initial_n
            if this_cell_improved and final_n == 1:  # unlikely!
                self.note(f"{source}: "
                          f"(row={row + 1}, col={col + 1}) must be "
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
        # log.critical(f"{source}: Eliminating {digit_zb + 1} "
        #              f"from ({row_zb + 1},{col_zb + 1})")
        self.possible[row_zb][col_zb][digit_zb] = False
        n = self.n_possibilities(row_zb, col_zb)
        if n == 0:
            raise SolutionFailure()
        if n == 1:  # improved down to 1
            # Sudoku Snake: "Naked Singles".
            self.note(f"{source}: "
                      f"(row={row_zb + 1}, col={col_zb + 1}) must be "
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
        for c in range(self.n):
            if c in except_cols_zb:
                continue
            for d in digits_zb_to_eliminate:
                zsource = (f"{source}: Eliminating digit {d + 1} "
                           f"from row {row_zb + 1}")
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
        for r in range(self.n):
            if r in except_rows_zb:
                continue
            for d in digits_zb_to_eliminate:
                zsource = (f"{source}: Eliminating digit {d + 1} "
                           f"from column {col_zb + 1}")
                improved = self._eliminate_from_cell(
                    r, col_zb, d, source=zsource) or improved
        return improved

    def _eliminate_groupwise_rows_cols(self, groupsize: int) -> bool:
        """
        Scan through all numbers in groups of n. If there are exactly n
        possible cells for each of those n numbers in a row/column/box, and
        those possibilities are all the same, eliminate other possible digits
        for those cells.

        Returns: improved?

        Example: in row 1, if digit 4 could be in columns 5/7 only, and digit 8
        could be in columns 5/7 only, then either '4' goes in column 5 and '8'
        goes in column 7, or vice versa - but these cells can't possibly
        contain anything other than '4' or '8'.

        That's an easy situation. We can illustrate it in more detail, e.g.
        with six cells:

        .. code-block:: none

            <?>                 # Inference: cannot contain 4 or 8
            4, 8    # Cell A
            <?>                 # Inference: cannot contain 4 or 8
            <?>                 # Inference: cannot contain 4 or 8
            4, 8    # Cell B
            <?>                 # Inference: cannot contain 4 or 8

        Here, there are 2 cells with the same 2 possible values, and no other
        cells can contain those values. The really obvious logic is:

        - cell A must contain 4 or 8
        - cell B must contain 4 or 8
        - cell B must contain whichever of {4, 8} that cell A does not contain,
          and vice versa
        - no other cell can contain 4 or 8

        There is a closely related but different inference to be made if the
        cells in the set contain other values, but no other cells do:

        .. code-block:: none

            <not_4_not_8>
            4, 8, 9             # Cell A. Inference: can only contain 4, 8.
            <not_4_not_8>
            <not_4_not_8>
            4, 8                # Cell B. (Inference: can only contain 4, 8.)
            <not_4_not_8>

        Here, the logic is:

        - there are only two locations for 4 and 8, and they are cells A and B;
        - since there are only two locations for two digits, they must be in
          those cells, and no other cells can contain them.

        This can be combined into a single algorithm:

        - If set {A, B, ...} of length n is present in n cells and no others,
          then eliminate everything not in the set from those cells, and
          everything in the set from all other cells.

        To scale this up to larger set sizes... The situation where n=1 is part
        of our basic elimination process, so our caller will begin with n=2. No
        point going to n=9 (that means we have no information!) or n=8 (if 8
        wholly uncertain digits, then 1 completely certain digit, and that
        situation was dealt with earlier, in :meth:`_eliminate_simple`).

        See also: http://pi.math.cornell.edu/~mec/Summer2009/Mahmood/Solve.html
        """
        # Sudoku Snake: "Hidden Subsets" (restricting affected cells).
        # Sudoku Snake: "Naked Subsets" (eliminating from other cells).
        log.debug(f"Eliminating, groupwise, group size {groupsize}...")
        n = self.n
        improved = False
        for digit_combo in combinations(range(n), r=groupsize):
            pretty_digits = [d + 1 for d in digit_combo]
            combo_set = set(digit_combo)

            # -----------------------------------------------------------------
            # Rows
            # -----------------------------------------------------------------
            source = f"eliminate_groupwise, by row, digits={pretty_digits}"
            for row in range(n):
                columns_with_all_these_digits = [
                    col for col in range(n)
                    if combo_set.issubset(self.possible_digits(row, col))
                ]
                columns_with_any_of_these_digits = [
                    col for col in range(n)
                    if combo_set.intersection(self.possible_digits(row, col))
                ]
                if (len(columns_with_all_these_digits) == groupsize and
                        len(columns_with_any_of_these_digits) == groupsize):
                    prettycol = [c + 1 for c in columns_with_all_these_digits]
                    this_improved = self._eliminate_from_row(
                        row_zb=row,
                        except_cols_zb=columns_with_all_these_digits,
                        digits_zb_to_eliminate=digit_combo,
                        source=source
                    )
                    this_improved = self._restrict_cells(
                        cells=[(row, c)
                               for c in columns_with_all_these_digits],
                        digits_zb_to_keep=digit_combo,
                        source=source
                    ) or this_improved
                    if this_improved:
                        self.note(
                            f"In row {row + 1}, only columns {prettycol} "
                            f"could contain digits {pretty_digits}")
                    improved = improved or this_improved

            # -----------------------------------------------------------------
            # Columns
            # -----------------------------------------------------------------
            source = f"eliminate_groupwise, by column, digits={pretty_digits}"
            for col in range(n):
                rows_with_all_these_digits = [
                    row for row in range(n)
                    if combo_set.issubset(self.possible_digits(row, col))
                ]
                rows_with_any_of_these_digits = [
                    row for row in range(n)
                    if combo_set.intersection(self.possible_digits(row, col))
                ]
                if (len(rows_with_all_these_digits) == groupsize and
                        len(rows_with_any_of_these_digits) == groupsize):
                    prettyrow = [r + 1 for r in rows_with_all_these_digits]
                    this_improved = self._eliminate_from_col(
                        col_zb=col,
                        except_rows_zb=rows_with_all_these_digits,
                        digits_zb_to_eliminate=digit_combo,
                        source=source
                    )
                    this_improved = self._restrict_cells(
                        cells=[(r, col)
                               for r in rows_with_all_these_digits],
                        digits_zb_to_keep=digit_combo,
                        source=source
                    ) or this_improved
                    if this_improved:
                        self.note(
                            f"In column {col + 1}, only rows {prettyrow} "
                            f"could contain digits {pretty_digits}")
                    improved = improved or this_improved

        return improved

    # -------------------------------------------------------------------------
    # Strategy
    # -------------------------------------------------------------------------

    def eliminate(self) -> bool:
        """
        Eliminates the impossible.

        Returns: improved?
        """
        raise NotImplementedError

    def solve(self, no_guess: bool = False) -> None:
        """
        Solves by elimination.

        Args:
            no_guess: prohibit guessing
        """
        iteration = 0
        while not self.solved():
            log.debug(
                f"Iteration {iteration}. "
                f"Unsolved cells: {self.n_unknown_cells()}. "
                f"Possible digit assignments: "
                f"{self.n_possibilities_overall()} "
                f"(target {self.n * self.n}). "
                f"Possibilities:\n{self}")
            improved = self.eliminate()
            if not improved:
                self.note("No improvement; need to guess")
                if no_guess:
                    log.info(f"Possibilities:\n{self}")
                    raise ValueError("Would need to guess, but prohibited")
                else:
                    log.debug(f"Possibilities:\n{self}")
                self.guess()
            iteration += 1

    # -------------------------------------------------------------------------
    # Guessing
    # -------------------------------------------------------------------------

    def guess(self) -> None:
        """
        Implements the "guess" method!

        Using this means that the algorithm has deficiencies (or the puzzle
        does).
        """
        for r in range(self.n):
            for c in range(self.n):
                digits = self.possible_digits(r, c)
                if len(digits) == 1:
                    continue
                for d in digits:
                    p = self.clone()
                    p.guess_level = self.guess_level + 1
                    log.warning("Guessing")
                    p.assign_digit(r, c, d,
                                   source=f"guess level {p.guess_level}")
                    try:
                        p.solve()
                        assert p.solved()
                        log.info(f"Guess level {p.guess_level} was good")
                        self._assign_from_other(p)  # copy back from p
                        return
                    except SolutionFailure:
                        log.info(
                            f"Bad guess at level {p.guess_level}; moving on")

    def _assign_from_other(self, other: "CommonPossibilities") -> None:
        """
        Copies another's attributes to self.
        Used by guessing.
        """
        # no need to copy initial_values_zb
        self.possible = deepcopy(other.possible)
        self.working = deepcopy(other.working)
