#!/bin/bash
set -e
shopt -s globstar

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SOLVER_DIR=${THIS_DIR}/puzzles

SUDOKU=${SOLVER_DIR}/sudoku.py
FUTOSHIKI=${SOLVER_DIR}/futoshiki.py

# VERBOSE=--verbose
VERBOSE=

NOGUESS=--noguess
# NOGUESS=

# https://apple.stackexchange.com/questions/49042/how-do-i-make-find-fail-if-exec-fails

for x in "${THIS_DIR}"/**/futoshiki*.txt; do
    # echo $x
    python "${FUTOSHIKI}" ${VERBOSE} working "${x}" ${NOGUESS}
done

for x in "${THIS_DIR}"/**/sudoku*.txt; do
    # echo $x
    python "${SUDOKU}" ${VERBOSE} working "${x}" ${NOGUESS}
done
