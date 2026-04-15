# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**RUMOO** is a Rust port of [PyMOO](https://pymoo.org/) (Python Multi-Objective Optimization library v0.6.1.6). The goal is to re-implement PyMOO's evolutionary algorithms in Rust for performance. The full Python PyMOO library lives in `pymoo/` as the reference implementation; `src/` is the nascent Rust port.

**Current state**: Very early. `src/main.rs` is a placeholder, and `src/core/problem.rs` currently contains Python source (a copy of `pymoo/core/problem.py`) — it has not yet been converted to Rust.

## Commands

### Rust
```bash
cargo build           # debug build
cargo build --release # release build
cargo test            # run Rust tests
```

### Python (reference implementation & tests)
```bash
# Install Python dependencies
pip install -r tests/requirements.txt

# Run all tests
pytest tests/

# Run a single test file
pytest tests/algorithms/test_nsga2.py

# Run tests and overwrite stored expected outputs
pytest tests/ --overwrite
```

## Architecture

### `pymoo/` — Reference Python Implementation

The Python code is the authoritative source for algorithm semantics. Port Rust code to match behavior here.

**Core abstractions** (in `pymoo/core/`):
- `Problem` — base class for all optimization problems. Users subclass and implement `_evaluate(x, out)`, writing objectives to `out["F"]`, inequality constraints to `out["G"]`, equality constraints to `out["H"]`. Supports vectorized (batch) and elementwise evaluation modes.
- `Algorithm` — base for all optimization algorithms; `GeneticAlgorithm` subclasses it for population-based EAs.
- `Population` / `Individual` — solution representation; solutions carry attributes `F`, `G`, `H`, `rank`, `crowding`, etc.
- `Result` — returned by `minimize()`; holds the final population and Pareto front.

**Algorithm hierarchy**:
- `pymoo/algorithms/moo/` — multi-objective: NSGA-II, NSGA-III, MOEA/D, RVEA, SPEA2, etc.
- `pymoo/algorithms/soo/` — single-objective: GA, DE, PSO, CMA-ES, Nelder-Mead, etc.

**Operators** (`pymoo/operators/`): crossover (SBX, DE, PMX…), mutation (Polynomial, Gaussian, Bit-flip…), sampling (LHS, random…), selection (tournament, random…), survival (rank-and-crowding, reference-direction…).

**Benchmark problems** (`pymoo/problems/`): ZDT, DTLZ, WFG, ZCAT suites; single-, multi-, and many-objective variants.

**Entry point**: `pymoo.optimize.minimize(problem, algorithm, termination)` — wires everything together and returns a `Result`.

### `src/` — Rust Port

Mirrors the `pymoo/core/` structure. As modules are ported, a corresponding `src/core/<module>.rs` should replace the Python reference. No Cargo dependencies are declared yet — add them to `Cargo.toml` as needed when porting numeric kernels (e.g., `ndarray`, `rand`).

### `tests/` — Python Test Suite

Tests target the Python reference implementation. Useful for verifying that Rust ports produce numerically identical results when called via FFI or a Python binding layer.

## Key Porting Notes

- `Problem._evaluate` receives `X` (shape `[N, n_var]` for vectorized, shape `[n_var]` for elementwise) and writes shaped arrays into `out`: `F` → `(N, n_obj)`, `G` → `(N, n_ieq_constr)`, `H` → `(N, n_eq_constr)`.
- Constraint values follow PyMOO convention: `G ≤ 0` satisfies inequality constraints (positive values are infeasible).
- Non-dominated sorting is the computational bottleneck in NSGA-II; `pymoo/util/nds/` has multiple implementations including a Cython-compiled fast variant.
- `pymoo/functions/compiled/` contains Cython extensions for hot paths — these are the first targets for Rust replacement.
