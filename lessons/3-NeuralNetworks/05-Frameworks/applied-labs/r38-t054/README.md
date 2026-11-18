# Applied Lab R38-T054: Intent Routing and Auditable Decision Trace

## Overview
Demonstrates rule-based intent routing with deterministic priority arbitration and an auditable step-by-step state trace.

## Routing Rules & Invariants
- Priority 1: `help` keyword -> handler `support`
- Priority 2: `calculate` keyword -> handler `calculator`
- Default Fallback: `unknown` requests -> handler `fallback`
- Ambiguous Inputs: Evaluates all rules, resolves by priority, logs decision record.
