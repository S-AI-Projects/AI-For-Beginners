# Applied Lab R38-T053: Dependency-Plan Validation and Ordering

## Overview
Extends the Multi-Agent Systems lesson by validating multi-agent plan dependencies, detecting circular deadlocks and missing agent prerequisites before execution.

## Contract & Requirements
- **Input**: Dictionary of `{task_id: [prerequisite_ids]}`.
- **Valid Order**: Returns topologically sorted execution sequence.
- **Cycle Detection**: Circular dependencies (e.g., `A -> B -> A`) raise `CycleDetectedError`.
- **Missing Dependency**: Dependencies on undeclared task IDs raise `MissingDependencyError` reporting the exact ID.
- **Two Approaches**: Implements and compares Kahn's queue algorithm and an iterative dependency table check.
