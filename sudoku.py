#!/usr/bin/env python

"""
sudoku.py

===============================================================================

    Copyright (C) 2015-2019 Rudolf Cardinal (rudolf@pobox.com).

    This file is part of CRATE.

    CRATE is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    CRATE is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with CRATE. If not, see <http://www.gnu.org/licenses/>.

===============================================================================

**Solves Sudoku puzzles.**

"""


import argparse
import logging
import sys
from typing import List

from cardinal_pythonlib.argparse_func import RawDescriptionArgumentDefaultsHelpFormatter  # noqa
from cardinal_pythonlib.logs import main_only_quicksetup_rootlogger
from mip import BINARY, Constr, Model, Var, xsum

log = logging.getLogger(__name__)


# =============================================================================
# Futoshiki
# =============================================================================

UNKNOWN = "."
NEWLINE = "\n"
SPACE = " "
ALMOST_ONE = 0.99
N = 9


class Sudoko(object):
    """
    Represents and solves Sudoko puzzles.
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

        # Check input is basically sound
        lines = string_version.splitlines()
        if not lines:
            raise ValueError("No data")

        lines = ["".join(line.split())
                 for line in lines if line]  # remove blank lines/columns
        if len(lines) != N:
            raise ValueError(f"Must have {N} active lines; "
                             f"found {len(lines)}, which are:\n"
                             f"{lines}")

        # log.critical(f"string_version:\n{string_version}")
        # log.critical(f"lines: {lines}")

        # Read user's input.
        for row_zb in range(N):
            line = lines[row_zb]
            assert len(line) == N, (
                f"Data line has wrong non-blank length: should be {N}, "
                f"but is {len(line)} ({line!r})"
            )
            for col_zb in range(N):
                self.problem_data[row_zb][col_zb] = line[col_zb]

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
                if col_zb % 3 == 2:
                    x += SPACE
            x += NEWLINE
            if row_zb % 3 == 2:
                x += NEWLINE
        return x

    @staticmethod
    def _debug_model_constraints(m: Model) -> None:
        """
        Shows constraints for a model.
        """
        lines = [f"Constraints in model {m.name!r}:"]
        for c in m.constrs:  # type: Constr
            lines.append(f"{c.name} == {c.expr}")
        log.debug("\n".join(lines))

    @staticmethod
    def _debug_model_vars(m: Model) -> None:
        """
        Show the names/values of model variables after fitting.
        """
        lines = [f"Variables in model {m.name!r}:"]
        for v in m.vars:  # type: Var
            lines.append(f"{v.name} == {v.x}")
        log.debug("\n".join(lines))

    def solve(self) -> None:
        """
        Solves the problem, writing to :attr:`solved` and
        :attr:`solution_data`.
        """
        if self.solved:
            log.info("Already solved")
            return

        m = Model("Sudoku solver")

        # ---------------------------------------------------------------------
        # Variables
        # ---------------------------------------------------------------------
        x = [
            [
                [
                    m.add_var(f"x(row={r + 1}, col={c + 1}, digit={d + 1})",
                              var_type=BINARY)
                    for d in range(N)
                ] for c in range(N)
            ] for r in range(N)
        ]  # index as: x[row_zb][col_zb][digit_zb]

        # ---------------------------------------------------------------------
        # Constraints
        # ---------------------------------------------------------------------

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
        t = 3
        for d in range(N):
            for box_row in range(t):
                for box_col in range(t):
                    row_base = box_row * t
                    col_base = box_col * t
                    m += xsum(
                        x[row_base + row_offset][col_base + col_offset][d]
                        for row_offset in range(t)
                        for col_offset in range(t)
                    ) == 1
        # Starting values
        for r in range(N):
            for c in range(N):
                if self.problem_data[r][c] != UNKNOWN:
                    d_zb = int(self.problem_data[r][c]) - 1
                    m += x[r][c][d_zb] == 1

        # ---------------------------------------------------------------------
        # Solve
        # ---------------------------------------------------------------------
        self._debug_model_constraints(m)
        m.optimize()

        # ---------------------------------------------------------------------
        # Read out answers
        # ---------------------------------------------------------------------
        self._debug_model_vars(m)
        if m.num_solutions:
            self.solved = True
            for r in range(N):
                for c in range(N):
                    for d_zb in range(N):
                        if x[r][c][d_zb].x > ALMOST_ONE:
                            self.solution_data[r][c] = str(d_zb + 1)
                            break


DEMO_SUDOKU_1 = Sudoko("""
... ... ...
..2 3.1 45.
.1. ... .6.

.47 .5. 38.
... 7.3 ...
.36 ... 14.

.7. ... .9.
.91 4.5 6..
... ..9 ...
""")  # Coton 54, Coton Community News Dec 2019-Jan 2020


# =============================================================================
# main
# =============================================================================

def main() -> None:
    """
    Command-line entry point.
    """
    cmd_solve = "solve"
    cmd_demo = "demo"

    parser = argparse.ArgumentParser(
        formatter_class=RawDescriptionArgumentDefaultsHelpFormatter,
        description=(
            f"Solve Sudoku puzzles. Format is:\n\n"
            f"{DEMO_SUDOKU_1}"
        )
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Be verbose"
    )
    subparsers = parser.add_subparsers(
        dest="command",
        # required=True,  # needs Python 3.7
        help="Append --help for more help"
    )

    parser_solve = subparsers.add_parser(cmd_solve, help="Solve from a file")
    parser_solve.add_argument(
        "filename", type=str, default="",
        help=f"Puzzle filename to read. Must contain text in format as above."
    )

    _parser_demo = subparsers.add_parser(cmd_demo, help="Run demo")

    args = parser.parse_args()
    main_only_quicksetup_rootlogger(level=logging.DEBUG if args.verbose
                                    else logging.INFO)

    if not args.command:
        print("Must specify command")
        sys.exit(1)
    if args.command == cmd_demo:
        problem = DEMO_SUDOKU_1
    else:
        assert args.filename, "Must specify parameter: --filename"
        with open(args.filename, "rt") as f:
            string_version = f.read()
        problem = Sudoko(string_version)

    log.info(f"Solving:\n{problem}")
    problem.solve()
    log.info(f"Answer:\n{problem}")
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.critical(str(e))
        raise
