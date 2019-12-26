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

It uses two approaches:

- Integer programming, which is close to magic.
  You say "here are my constraints; go" and a few milliseconds later you have
  a valid answer.

- Tedious logic, like a human would do.
  Note that there are a large variety of these from a human's perspective
  (though only the simplest are required for the majority of puzzles
  encountered in the wild); see e.g. http://www.sudokusnake.com/techniques.php.

- Sudoku Snake technique names currently implemented:

  - Beginner:

    - Hidden Singles
    - Hidden Singles by Box
    - Naked Singles
    - Pointing
    - *** TO DO: Claiming

  - Intermediate

    - Hidden Subsets
    - Naked Subsets

  - Advanced

  - Master

  - Ludicrous

"""


import argparse
from itertools import combinations
import logging
import sys
import traceback
from typing import Generator, Iterable, List, Optional, Tuple

from cardinal_pythonlib.argparse_func import RawDescriptionArgumentDefaultsHelpFormatter  # noqa
from cardinal_pythonlib.logs import main_only_quicksetup_rootlogger
from mip import BINARY, Model, xsum

from common import (
    ALMOST_ONE,
    CommonPossibilities,
    debug_model_constraints,
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

DEFAULT_RANK = 3


# =============================================================================
# Box
# =============================================================================

class Box(object):
    """
    Represents a 3x3 box within the Sudoku grid.
    """
    def __init__(self, box_zb: int, rank: int = DEFAULT_RANK) -> None:
        """
        Boxes are numbered 0 to N - 1. For a standard 9x9 (rank 3) Sudoku, with
        3x3 boxes, they are numbered 0-8.

        Args:
            box_zb: box number, as above; zero-based
        """
        assert 0 <= box_zb < rank ** 2, (
            f"box_zb was {box_zb}; must be in range 0 to {rank ** 2} inclusive"
        )
        self.rank = rank
        self.box_zb = box_zb

    def __str__(self) -> str:
        """
        Coordinate-based description for a 3x3 box.
        """
        return f"{{{self.boxrow + 1},{self.boxcol + 1}}}"

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def boxrow(self) -> int:
        """
        Zero-based row number of the box (not its cells).
        """
        return self.box_zb // self.rank

    @property
    def boxcol(self) -> int:
        """
        Zero-based row number of the box (not its cells).
        """
        return self.box_zb % self.rank

    def __eq__(self, other: "Box") -> bool:
        """
        Coordinate-based description for a 3x3 box.
        """
        return (
            self.rank == other.rank and
            self.box_zb == other.box_zb
        )

    def top_left_cell(self) -> Tuple[int, int]:
        """
        Returns ``row_zb, col_zb`` for the top-left cell in the 3x3 box
        (numbered from 0 to N - 1).
        """
        t = self.rank
        row = (self.box_zb // t) * t
        col = (self.box_zb % t) * t
        return row, col

    def extremes(self) -> Tuple[int, int, int, int]:
        """
        Defines the boundaries of the 3x3 box, numbered from 0 to (N - 1).

        Returns ``row_min, row_max, col_min, col_max``.
        """
        row_min, col_min = self.top_left_cell()
        row_max = row_min + 2
        col_max = col_min + 2
        return row_min, row_max, col_min, col_max

    @property
    def row_min(self) -> int:
        row_min, row_max, col_min, col_max = self.extremes()
        return row_min

    @property
    def row_max(self) -> int:
        row_min, row_max, col_min, col_max = self.extremes()
        return row_max

    @property
    def col_min(self) -> int:
        row_min, row_max, col_min, col_max = self.extremes()
        return col_min

    @property
    def col_max(self) -> int:
        row_min, row_max, col_min, col_max = self.extremes()
        return col_max

    @classmethod
    def containing(cls, row_zb: int, col_zb: int,
                   rank: int = DEFAULT_RANK) -> "Box":
        """
        Returns the box containing this cell.

        Args:
            row_zb: zero-based row number
            col_zb: zero-based column number
            rank: rank
        """
        t = rank
        n = rank ** 2
        assert 0 <= row_zb < n
        assert 0 <= col_zb < n
        boxrow = row_zb // t
        boxcol = col_zb // t
        return cls.from_boxrowcol(boxrow, boxcol, rank=rank)

    @classmethod
    def from_boxrowcol(cls, boxrow: int, boxcol: int,
                       rank: int = DEFAULT_RANK) -> "Box":
        """
        Returns the box at this boxrow/boxcol.
        """
        assert 0 <= boxrow < rank
        assert 0 <= boxcol < rank
        box_zb = boxrow * rank + boxcol
        return cls(box_zb=box_zb, rank=rank)

    def gen_cells(self) -> Generator[Tuple[int, int], None, None]:
        """
        Generates ``(row_zb, col_zb)`` tuples for all the cells in this box.
        """
        row_min, row_max, col_min, col_max = self.extremes()
        for r in range(row_min, row_max + 1):
            for c in range(col_min, col_max + 1):
                yield r, c

    @property
    def rownums(self) -> List[int]:
        """
        Row numbers (zero-based) covered by this box.
        """
        return list(range(self.row_min, self.row_max + 1))

    @property
    def colnums(self) -> List[int]:
        """
        Row numbers (zero-based) covered by this box.
        """
        return list(range(self.col_min, self.col_max + 1))

    @property
    def other_boxes_same_row(self) -> List["Box"]:
        """
        Other boxes with the same row number.
        """
        boxrow = self.boxrow
        boxcol = self.boxcol
        return [
            self.from_boxrowcol(boxrow=boxrow, boxcol=c, rank=self.rank)
            for c in range(self.rank)
            if c != boxcol
        ]

    @property
    def other_boxes_same_col(self) -> List["Box"]:
        """
        Other boxes with the same column number.
        """
        boxrow = self.boxrow
        boxcol = self.boxcol
        return [
            self.from_boxrowcol(boxrow=r, boxcol=boxcol, rank=self.rank)
            for r in range(self.rank)
            if r != boxrow
        ]

    def gen_cells_other_boxes_same_row(self) \
            -> Generator[Tuple[int, int], None, None]:
        """
        Generates ``(row, col)`` cell coordinates for boxes OTHER than this one
        but in the same box row.
        """
        for box in self.other_boxes_same_row:
            for r, c in box.gen_cells():
                yield r, c

    def gen_cells_other_boxes_same_col(self) \
            -> Generator[Tuple[int, int], None, None]:
        """
        Generates ``(row, col)`` cell coordinates for boxes OTHER than this one
        but in the same box row.
        """
        for box in self.other_boxes_same_col:
            for r, c in box.gen_cells():
                yield r, c


# =============================================================================
# SudokuPossibilities
# =============================================================================

class SudokuPossibilities(CommonPossibilities):
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
        in the same row/column/3x3 box.

    3.  Where there is only one possible cell for a digit in a
        row/column/box, confirm that digit by eliminating other possible
        digits from that cell.

    4.  Where there are n identical sets of n possible locations for n digits
        (e.g. for n=2: digit 4 can only be in locations A or B, and digit 8 can
        only be in locations A or B), eliminate all other possible digits from
        those locations. This is an extension of the previous tactic (for which
        n=1).

    5.  Repeat these tactics until the puzzle is solved or there is no
        improvement in knowledge.

    """
    def __init__(self, other: "SudokuPossibilities" = None,
                 rank: int = DEFAULT_RANK) -> None:
        """
        Initialize with "everything is possible", or copy from another.

        Args:
            other:
                other object to copy
            rank:
                rank of the puzzle (3 for normal 9x9 Sudoku)

        """
        n = rank ** 2
        self.rank = rank
        super().__init__(other=other, n=n)

    def clone(self) -> "SudokuPossibilities":
        return self.__class__(other=self, rank=self.rank)

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------

    def set_initial_values(self,
                           initial_values: List[List[Optional[int]]]) -> None:
        """
        Set up the initial values. The incoming digits are genuine (one-based),
        not zero-based.
        """
        super().set_initial_values(initial_values=initial_values)
        n_distinct = len(set(x
                             for rowlist in initial_values
                             for x in rowlist
                             if x is not None))
        if n_distinct < self.n - 1:
            log.warning(
                f"Not a well-formed Sudoku: {n_distinct} distinct initial "
                f"values given, but need {self.n - 1} to be well-formed.")
            # http://pi.math.cornell.edu/~mec/Summer2009/Mahmood/More.html
            # Need (rank ^ 2 - 1) distinct values, i.e. (n - 1) values.

    # -------------------------------------------------------------------------
    # Visuals
    # -------------------------------------------------------------------------

    def _pstr_row_col(self, row_zb: int, col_zb: int, digit_zb: int) \
            -> Tuple[int, int]:
        """
        For __str__(): ``row, col`` (``y, x``) coordinates.
        """
        t = self.rank
        x_base = col_zb * (t + 1)
        y_base = row_zb * (t + 1)
        x_offset = digit_zb % t
        y_offset = digit_zb // t
        return (y_base + y_offset), (x_base + x_offset)

    def __str__(self) -> str:
        """
        Returns a visual representation of possibilities.
        """
        pn = self.n * 4 - 1
        t = self.rank

        # Create grid of characters
        # BEWARE: DO NOT DO THIS: strings = [[" "] * pn] * pn
        # ... it creates copies of identical strings/lists, so when you
        # modify one, it modifies them all.
        strings = []  # type: List[List[str]]
        for _ in range(pn):
            line = []  # type: List[str]
            for _ in range(pn):
                line.append(SPACE)
            strings.append(line)

        # Prettify
        cell_boundaries = ((t + 1) * t - 1, (t + 1) * (t * 2) - 1)
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
        for r in range(self.n):
            for c in range(self.n):
                initial_value = self.initial_values_zb[r][c] is not None
                cell_solved = self.n_possibilities(r, c) == 1
                for d_zb in range(self.n):
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
    # Computations
    # -------------------------------------------------------------------------

    def _eliminate_from_box(self, except_cells_zb: List[Tuple[int, int]],
                            digits_zb_to_eliminate: Iterable[int],
                            source: str = "?") -> bool:
        """
        Eliminates a digit or digits from all cells in a 3x3 "box" except
        the cell(s) specified.

        Returns: Improved?
        """
        box = Box.containing(
            row_zb=min(t[0] for t in except_cells_zb),
            col_zb=min(t[1] for t in except_cells_zb),
            rank=self.rank
        )
        assert box == Box.containing(
            row_zb=max(t[0] for t in except_cells_zb),
            col_zb=max(t[1] for t in except_cells_zb),
            rank=self.rank
        ), "except_cells_zb spans more than one box!"
        improved = False
        for r, c in box.gen_cells():
            if (r, c) in except_cells_zb:
                continue
            for d in digits_zb_to_eliminate:
                zsource = f"{source}: eliminate_from_box, eliminating {d + 1}"
                improved = self._eliminate_from_cell(
                    r, c, d, source=zsource) or improved
        return improved

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
                    imp_box = self._eliminate_from_box([(row, col)], [d],
                                                       source=source)
                    improved = improved or imp_row or imp_col or imp_box
        return improved

    def _find_only_possibilities(self) -> bool:
        """
        Where there is only one location for a digit in a row, cell, or box,
        assign it.
        """
        improved = False
        for d in range(self.n):
            # Rows
            # Sudoku Snake: "Hidden Singles".
            for r in range(self.n):
                possible_cols = [c for c in range(self.n)
                                 if self.possible[r][c][d]]
                if len(possible_cols) == 1:
                    source = (
                        f"Only possibility for digit {d + 1} in row {r + 1}")
                    improved = self.assign_digit(
                        r, possible_cols[0], d, source=source) or improved
            # Columns
            # Sudoku Snake: "Hidden Singles".
            for c in range(self.n):
                possible_rows = [r for r in range(self.n)
                                 if self.possible[r][c][d]]
                if len(possible_rows) == 1:
                    source = (
                        f"Only possibility for digit {d + 1} "
                        f"in column {c + 1}")
                    improved = self.assign_digit(
                        possible_rows[0], c, d, source=source) or improved
            # Boxes
            # Sudoku Snake: "Hidden Singles By Box".
            for b in range(self.n):
                box = Box(b)
                possible_cells = [
                    (r, c)
                    for r, c in box.gen_cells()
                    if self.possible[r][c][d]
                ]
                if len(possible_cells) == 1:
                    source = f"Only possibility for digit {d + 1} in box {b}"
                    improved = self.assign_digit(
                        possible_cells[0][0],
                        possible_cells[0][1], d, source=source) or improved
        return improved

    def _eliminate_boxwise(self) -> bool:
        """
        The generalized rule:

        - For each box in turn...

        - For each digit in turn...

        - Iterate through the (rank - 1) other boxes in a box row (or box
          column).

        - If a combination of x boxes permit only x rows (or columns) for the
          digit (e.g. 1 box permits only one row, or 2 boxes permit only 2
          rows), then our box cannot have that digit in those rows.

        Example for a "single" box:

            If we don't know where the 5 is in a particular 3x3 box, but we
            know that it's in the first row, then we can eliminate "5" from
            that row in all other boxes.

        Example for a "pair" of boxes:

            Consider the three boxes making up the left-hand column of boxes.
            If the "5" digit can only be in columns 2 and 3 in the middle box,
            and can only be in columns 2 and 3 in the bottom box, then it must
            be in column 1 in the top box (and we can eliminate accordingly).

        Returns: improved?

        Sudoku Snake terminology: "Pointing".
        """
        log.debug(f"Eliminating, box-wise...")
        improved = False
        for groupsize in range(1, self.rank):  # 1 to rank - 1
            # log.critical(f"boxwise: groupsize {groupsize}")
            for b in range(self.n):
                box = Box(b)
                for d in range(self.n):
                    # ---------------------------------------------------------
                    # By row
                    # ---------------------------------------------------------
                    for box_combo in combinations(box.other_boxes_same_row,
                                                  r=groupsize):
                        possible_rows_other_boxes = set(
                            r
                            for otherbox in box_combo
                            for r, c in otherbox.gen_cells()
                            if d in self.possible_digits(r, c)
                        )
                        # log.critical(
                        #     f"digit = {d + 1}; box = {box}; "
                        #     f"box_combo = {box_combo}; "
                        #     f"possible_rows_other_boxes = "
                        #     f"{possible_rows_other_boxes}")
                        if len(possible_rows_other_boxes) == groupsize:
                            source = (
                                f"eliminate_boxwise for digit {d + 1} in box "
                                f"{box} with constraints from other boxes "
                                f"{list(box_combo)} in the same row"
                            )
                            impossible_rows_this_box = possible_rows_other_boxes  # noqa
                            for r in impossible_rows_this_box:
                                for c in box.colnums:
                                    improved = self._eliminate_from_cell(
                                        r, c, d, source) or improved

                    # ---------------------------------------------------------
                    # By column
                    # ---------------------------------------------------------
                    for box_combo in combinations(box.other_boxes_same_col,
                                                  r=groupsize):
                        possible_cols_other_boxes = set(
                            c
                            for otherbox in box_combo
                            for r, c in otherbox.gen_cells()
                            if d in self.possible_digits(r, c)
                        )
                        if len(possible_cols_other_boxes) == groupsize:
                            source = (
                                f"eliminate_boxwise for digit {d + 1} in box "
                                f"{box} with constraints from other boxes "
                                f"{list(box_combo)} in the same column"
                            )
                            impossible_cols_this_box = possible_cols_other_boxes  # noqa
                            for r in box.rownums:
                                for c in impossible_cols_this_box:
                                    improved = self._eliminate_from_cell(
                                        r, c, d, source) or improved

        return improved

    def _eliminate_groupwise(self, groupsize: int) -> bool:
        """
        Scan through all numbers in groups of n. If there are exactly n
        possible cells for each of those n numbers in a row/column/box, and
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
        # Sudoku Snake: "Hidden Subsets" (restricting affected cells).
        # Sudoku Snake: "Naked Subsets" (eliminating from other cells).
        log.debug(f"Eliminating, groupwise, group size {groupsize}...")
        improved = self._eliminate_groupwise_rows_cols(groupsize)
        for digit_combo in combinations(range(self.n), r=groupsize):
            pretty_digits = [d + 1 for d in digit_combo]
            combo_set = set(digit_combo)
            source = f"eliminate_groupwise, by box, digits={pretty_digits}"
            # Boxes
            for b in range(self.n):
                box = Box(b, rank=self.rank)
                cells_with_all_these_digits = [
                    (r, c)
                    for r, c in box.gen_cells()
                    if combo_set.issubset(self.possible_digits(r, c))
                ]
                cells_with_any_of_these_digits = [
                    (r, c)
                    for r, c in box.gen_cells()
                    if combo_set.intersection(self.possible_digits(r, c))
                ]
                if (len(cells_with_all_these_digits) == groupsize and
                        len(cells_with_any_of_these_digits) == groupsize):
                    prettycell = [(r + 1, c+1)
                                  for r, c in cells_with_all_these_digits]
                    self.note(
                        f"In box {box}, only cells {prettycell} "
                        f"could contain digits {pretty_digits}")
                    improved = self._eliminate_from_box(
                        except_cells_zb=cells_with_all_these_digits,
                        digits_zb_to_eliminate=digit_combo,
                        source=source
                    ) or improved
                    improved = self._restrict_cells(
                        cells=cells_with_all_these_digits,
                        digits_zb_to_keep=digit_combo,
                        source=source
                    ) or improved
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
        improved = self._find_only_possibilities() or improved
        if improved:
            return improved  # Keep it simple...

        improved = self._eliminate_boxwise()
        if improved:
            return improved  # Keep it simple...

        for groupsize in range(2, self.n):  # from 2 to 8
            improved = self._eliminate_groupwise(groupsize) or improved
            if improved:
                return improved

        return improved


# =============================================================================
# Sudoku
# =============================================================================

class Sudoku(object):
    """
    Represents and solves Sudoku puzzles.
    """

    def __init__(self, string_version: str, rank: int = DEFAULT_RANK) -> None:
        """
        Args:
            string_version:
                String representation of the puzzle. Rules as below.
            rank:
                rank of the puzzle (3 for normal 9x9 Sudoku)

        - Initial/terminal blank lines are ignored
        - Use numbers 1-9 for known cells.
        - ``.`` represents an unknown cell.
        - one space (and one line) between each cell.
        """
        self.rank = rank
        self.n = rank ** 2
        # Create data structure
        self.solved = False
        self.problem_data = [
            [
                UNKNOWN for _col_zb in range(self.n)
            ] for _row_zb in range(self.n)
        ]
        self.solution_data = [
            [
                UNKNOWN for _col_zb in range(self.n)
            ] for _row_zb in range(self.n)
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
        if len(lines) != self.n:
            raise ValueError(f"Must have {self.n} active lines; "
                             f"found {len(lines)}, which are:\n"
                             f"{lines}")

        # Read user's input.
        for row_zb in range(self.n):
            line = lines[row_zb]
            assert len(line) == self.n, (
                f"Data line has wrong non-blank length: should be {self.n}, "
                f"but is {len(line)} ({line!r})"
            )
            for col_zb in range(self.n):
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

    def _make_string(self, data: List[List[str]]) -> str:
        x = ""
        for row_zb in range(self.n):
            for col_zb in range(self.n):
                # Cell, inequality, cell, inequality...
                x += data[row_zb][col_zb]
                if col_zb % self.rank == 2:
                    x += SPACE
            if row_zb < self.n - 1:
                x += NEWLINE
                if row_zb % self.rank == 2:
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
        n = self.n
        rank = self.rank

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Variables
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        x = [
            [
                [
                    m.add_var(f"x(row={r + 1}, col={c + 1}, digit={d + 1})",
                              var_type=BINARY)
                    for d in range(self.n)
                ] for c in range(self.n)
            ] for r in range(self.n)
        ]  # index as: x[row_zb][col_zb][digit_zb]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Constraints
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # One digit per cell
        for r in range(n):
            for c in range(n):
                m += xsum(x[r][c][d] for d in range(n)) == 1
        for d in range(n):
            # One of each digit per row
            for r in range(n):
                m += xsum(x[r][c][d] for c in range(n)) == 1
            # One of each digit per column
            for c in range(n):
                m += xsum(x[r][c][d] for r in range(n)) == 1
        # One of each digit in each 3x3 box:
        for d in range(n):
            for box_row in range(rank):
                for box_col in range(rank):
                    row_base = box_row * rank
                    col_base = box_col * rank
                    m += xsum(
                        x[row_base + row_offset][col_base + col_offset][d]
                        for row_offset in range(rank)
                        for col_offset in range(rank)
                    ) == 1
        # Starting values
        for r in range(n):
            for c in range(n):
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
            for r in range(n):
                for c in range(n):
                    for d_zb in range(n):
                        if x[r][c][d_zb].x > ALMOST_ONE:
                            self.solution_data[r][c] = str(d_zb + 1)
                            break
        else:
            log.error("Unable to solve!")

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

        p = SudokuPossibilities()  # start from scratch
        n = self.n

        # Set starting values.
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
        p.set_initial_values(initial_values)

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
        problem = Sudoku(DEMO_SUDOKU_1)
        log.info(f"Solving:\n{problem}")
        problem.solve()
    else:
        log.info(f"Reading {args.filename}")
        with open(args.filename, "rt") as f:
            string_version = f.read()
        problem = Sudoku(string_version)
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
