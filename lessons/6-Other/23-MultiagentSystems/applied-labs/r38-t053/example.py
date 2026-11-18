"""
R38-T053: Dependency-plan validation — compare two approaches
Extend “Multi-Agent Systems” at lessons/6-Other/23-MultiagentSystems/README.md.
Validate and topologically order a small task plan; detect missing dependencies and cycles before execution.
A has no dependencies, B depends on A, C on B: order A,B,C; A->B->A is rejected;
an unknown dependency is reported by task ID.
"""
from typing import Dict, List, Sequence

class PlanValidationError(Exception):
    pass

class MissingDependencyError(PlanValidationError):
    def __init__(self, task_id: str, missing_dep: str):
        super().__init__(f"Task '{task_id}' depends on unknown task '{missing_dep}'")
        self.task_id = task_id
        self.missing_dep = missing_dep

class CycleDetectedError(PlanValidationError):
    def __init__(self, cycle_nodes: Sequence[str]):
        super().__init__(f"Cycle detected in task plan: {list(cycle_nodes)}")
        self.cycle_nodes = list(cycle_nodes)

def topological_sort_kahn(plan: Dict[str, Sequence[str]]) -> List[str]:
    """
    Approach 1: Kahn's Algorithm (in-degree based).
    plan maps task_id -> list of prerequisite task_ids that must complete first.
    """
    all_tasks = set(plan.keys())
    for task_id, deps in plan.items():
        for dep in deps:
            if dep not in all_tasks:
                raise MissingDependencyError(task_id, dep)

    in_degree: Dict[str, int] = {t: len(plan[t]) for t in all_tasks}
    dependents: Dict[str, List[str]] = {t: [] for t in all_tasks}
    for task_id, deps in plan.items():
        for dep in deps:
            dependents[dep].append(task_id)

    queue = [t for t in sorted(all_tasks) if in_degree[t] == 0]
    order: List[str] = []

    while queue:
        node = queue.pop(0)
        order.append(node)
        for dep_task in sorted(dependents[node]):
            in_degree[dep_task] -= 1
            if in_degree[dep_task] == 0:
                queue.append(dep_task)

    if len(order) != len(all_tasks):
        unresolved = [t for t, deg in in_degree.items() if deg > 0]
        raise CycleDetectedError(sorted(unresolved))

    return order

def topological_sort_table(plan: Dict[str, Sequence[str]]) -> List[str]:
    """
    Approach 2: Verification using step-by-step dependency resolution table.
    """
    all_tasks = set(plan.keys())
    for task_id, deps in plan.items():
        for dep in deps:
            if dep not in all_tasks:
                raise MissingDependencyError(task_id, dep)

    completed: List[str] = []
    remaining = dict(plan)

    while remaining:
        ready = [t for t in sorted(remaining.keys()) if all(d in completed for d in remaining[t])]
        if not ready:
            raise CycleDetectedError(sorted(remaining.keys()))
        next_task = ready[0]
        completed.append(next_task)
        del remaining[next_task]

    return completed

def main():
    plan = {"A": [], "B": ["A"], "C": ["B"]}
    print("Kahn ordering:", topological_sort_kahn(plan))
    print("Table ordering:", topological_sort_table(plan))

if __name__ == "__main__":
    main()
