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

"""


import argparse
import logging
import sys
from typing import List, Optional

from cardinal_pythonlib.argparse_func import RawDescriptionArgumentDefaultsHelpFormatter  # noqa
from cardinal_pythonlib.logs import main_only_quicksetup_rootlogger
# from cardinal_pythonlib.maths_py import sum_of_integers_in_inclusive_range
from mip import BINARY, Model, xsum

from puzzles.common import debug_model_vars

log = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# String representations
LT = "<"
GT = ">"
TOP_LT_BOTTOM = "^"
BOTTOM_LT_TOP = "v"
BOTTOM_LT_TOP_POSSIBLES = "vV"
UNKNOWN_STR = "."
UNKNOWN_POSSIBLES = ".?"
SPACE = " "
NEWLINE = "\n"
HASH = "#"

ALMOST_ONE = 0.99

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
                UNKNOWN_STR for _col_zb in range(n)
            ] for _row_zb in range(n)
        ]
        self.solution_data = [
            [
                UNKNOWN_STR for _col_zb in range(n)
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
                    assert ineq in (LT, GT, SPACE), (
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
        # Constraints
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
                if ineq == LT:
                    implement_less_than(r, c, r, c + 1)
                elif ineq == GT:
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
                if self.problem_data[r][c] != UNKNOWN_STR:
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

    _parser_demo = subparsers.add_parser(cmd_demo, help="Run demo")

    args = parser.parse_args()
    main_only_quicksetup_rootlogger(level=logging.DEBUG if args.verbose
                                    else logging.INFO)

    if not args.command:
        print("Must specify command")
        sys.exit(1)
    if args.command == cmd_demo:
        problem = Futoshiki(DEMO_FUTOSHIKI_1)
        log.info(f"Solving:\n{problem}")
        problem.solve()
    else:
        assert args.filename, "Must specify parameter: --filename"
        with open(args.filename, "rt") as f:
            string_version = f.read()
        problem = Futoshiki(string_version)
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
