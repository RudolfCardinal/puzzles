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
import logging
import sys
from typing import List, Tuple

from cardinal_pythonlib.argparse_func import RawDescriptionArgumentDefaultsHelpFormatter  # noqa
from cardinal_pythonlib.logs import main_only_quicksetup_rootlogger
from mip import BINARY, Constr, Model, Var, xsum

from common import debug_model_constraints, debug_model_vars

log = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

UNKNOWN = "."
NEWLINE = "\n"
SPACE = " "
COMMENT = "#"
INTERPUNCT = "·"
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
        lines = [line for line in lines if not line.startswith(COMMENT)]

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
        # One of each digit in each 3x3 box:
        for d in range(N):
            for box_row in range(T):
                for box_col in range(T):
                    row_base = box_row * T
                    col_base = box_col * T
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

        possible = [
            [
                [
                    True for _d in range(N)
                ] for _c in range(N)
            ] for _r in range(N)
        ]  # ... index as: possible[row_zb][col_zb][digit_zb]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Visuals
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        def note(msg: str) -> None:
            """
            Save some working.
            """
            self.working.append(msg)
            log.info(msg)

        def pstr_row_col(row_zb: int, col_zb: int, digit_zb: int) \
                -> Tuple[int, int]:
            """
            For possible_str(): ``row, col`` (``y, x``) coordinates.
            """
            x_base = col_zb * (T + 1)
            y_base = row_zb * (T + 1)
            x_offset = digit_zb % T
            y_offset = digit_zb // T
            return (y_base + y_offset), (x_base + x_offset)

        def possible_str() -> str:
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
                    for d_zb in range(N):
                        y, x = pstr_row_col(r, c, d_zb)
                        txt = str(d_zb + 1) if possible[r][c][d_zb] else INTERPUNCT  # noqa
                        strings[y][x] = txt
            return "\n".join("".join(x for x in line) for line in strings)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Calculation helpers
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        def n_possibilities(row_zb: int, col_zb: int) -> None:
            """
            Number of possible digits for a cell.
            """
            return sum(possible[row_zb][col_zb])

        def possible_digits(row_zb: int, col_zb: int) -> List[int]:
            """
            Returns possible digits, in ZERO-BASED format, for a given cell.
            """
            return list(d for d, v in enumerate(possible[row_zb][col_zb]) if v)

        def solved() -> bool:
            """
            Are we there yet?
            """
            for r in range(N):
                for c in range(N):
                    if n_possibilities(r, c) != 1:
                        return False
            return True

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Computations
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        def assign_digit(row_zb: int, col_zb: int, digit_zb: int,
                         start: bool = False) -> None:
            """
            Assign a digit to a cell.
            """
            nonlocal possible
            text = "Starting value:" if start else "Calculated:"
            note(f"{text} Assigning digit {digit_zb + 1} to "
                 f"row={row_zb + 1}, col={col_zb + 1}")
            for d in range(N):
                if d == digit_zb:
                    assert possible[row_zb][col_zb][d], (
                        f"{text} Assigning digit {digit_zb + 1} to "
                        f"row={row_zb + 1}, col={col_zb + 1} "
                        f"where that is known to be impossible"
                    )
                else:
                    possible[row_zb][col_zb][d] = False

        def eliminate_from_row(row_zb: int, except_cols_zb: List[int],
                               digits_zb_to_eliminate: List[int]) -> None:
            nonlocal possible
            for c in range(N):
                if c in except_cols_zb:
                    continue
                for d in digits_zb_to_eliminate:
                    possible[row_zb][c][d] = False

        def eliminate_from_col(except_rows_zb: List[int], col_zb: int,
                               digits_zb_to_eliminate: List[int]) -> None:
            nonlocal possible
            for r in range(N):
                if r in except_rows_zb:
                    continue
                for d in digits_zb_to_eliminate:
                    possible[r][col_zb][d] = False

        def eliminate_from_box(except_cells_zb: List[Tuple[int, int]],
                               digits_zb_to_eliminate: List[int]) -> None:
            nonlocal possible
            except_row_min = min(t[0] for t in except_cells_zb)
            except_row_max = max(t[0] for t in except_cells_zb)
            except_col_min = min(t[1] for t in except_cells_zb)
            except_col_max = max(t[1] for t in except_cells_zb)
            startrow = (except_row_min // T) * T
            endrow = startrow + T
            assert endrow > except_row_max
            startcol = (except_col_min // T) * T
            endcol = startcol + T
            assert endcol > except_col_max
            # log.critical(
            #     f"except_cells_zb={except_cells_zb}, "
            #     f"digits_zb_to_eliminate={digits_zb_to_eliminate}, "
            #     f"startrow={startrow}, endrow={endrow}, "
            #     f"startcol={startcol}, endcol={endcol}"
            # )
            for r in range(startrow, endrow):
                for c in range(startcol, endcol):
                    if (r, c) in except_cells_zb:
                        continue
                    for d in digits_zb_to_eliminate:
                        possible[r][c][d] = False

        def eliminate() -> None:
            """
            Eliminates the impossible.
            """
            for row in range(N):
                for col in range(N):
                    d_possible = possible_digits(row, col)
                    if len(d_possible) == 1:
                        d = d_possible[0]
                        eliminate_from_row(row, [col], [d])
                        eliminate_from_col([row], col, [d])
                        eliminate_from_box([(row, col)], [d])

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Main process
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Set starting values.
        for r in range(N):
            for c in range(N):
                known_digit_str = self.problem_data[r][c]
                if known_digit_str != UNKNOWN:
                    d_zb = int(known_digit_str) - 1
                    assign_digit(r, c, d_zb, start=True)

        # Eliminate and iterate
        iteration = 0
        while not solved():
            log.debug(f"Iteration {iteration}. Possibilities:\n{possible_str()}")  # noqa
            if iteration >= 1:
                crash
            eliminate()
            iteration += 1
            # *** do more

        # Read out answers
        self.solved = True
        for r in range(N):
            for c in range(N):
                d_zb = next(
                    i for i, v in enumerate(self.possibilities[r, c]) if v)
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
