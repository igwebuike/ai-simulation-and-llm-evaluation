# FSM Repair, Validation, and Normalization for LLM Training

## Overview
This repository contains a structured AI evaluation task centered on validating, repairing, and normalizing finite state machines (FSMs) under strict deterministic and constraint-based rules.

## What the task demonstrates
- Complex structured input validation
- Deterministic repair logic
- Constraint enforcement
- Byte-stable output normalization
- Edge-case handling and automated testing

## Task summary
The system takes an FSM definition, an event sequence, and constraints, then validates whether the FSM is structurally correct, repairable, deterministic, and executable for the required sequence.

It must:
- remove unreachable states
- repair missing transitions deterministically
- generate missing target states when allowed
- enforce regex and max-state constraints
- reject invalid or non-repairable inputs
- produce byte-identical normalized JSON output for identical inputs

## Repository structure
- `instruction.md` — full specification
- `input/` — FSM, events, and constraints inputs
- `output/` — repaired FSM output
- `solution/` — implementation logic
- `tests/` — validation and correctness checks
- `environment/` — environment/task setup
- `task.toml` — task configuration

## Why this matters for AI systems
This kind of work reflects the design of controlled, rule-driven environments for AI training and evaluation. It requires precision, deterministic behavior, scenario handling, and structured transformation logic, which map closely to synthetic data generation and simulation-oriented AI workflows.

## Background
This task is representative of the type of structured work I have done in LLM evaluation and AI training workflows, including Snorkel-related projects involving validation logic, normalization, rule enforcement, and quality control.

## Notes
The emphasis in this project is correctness, determinism, reproducibility, and strict adherence to specification.