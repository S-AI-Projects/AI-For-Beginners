"""
R38-T054: Agent-component routing — invariant and state trace
Extend “Neural Network Frameworks” at lessons/3-NeuralNetworks/05-Frameworks/README.md.
Route incoming requests to a named handler using explicit intent rules, a fallback handler and an auditable decision record.
help->support, calculate->calculator, unknown->fallback; ambiguous intent follows the documented priority and logs the chosen rule.
"""
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class StepTrace:
    step_index: int
    rule_name: str
    matched: bool
    details: str

@dataclass
class DecisionRecord:
    request_text: str
    chosen_handler: str
    chosen_rule: str
    trace: List[StepTrace] = field(default_factory=list)

class ComponentRouter:
    """
    Explicit intent routing rules with documented priority:
    Priority 1: 'help' keyword -> handler 'support'
    Priority 2: 'calculate' keyword -> handler 'calculator'
    Fallback: -> handler 'fallback'
    """
    RULES = [
        ("rule_help", "help", "support", 1),
        ("rule_calculate", "calculate", "calculator", 2),
    ]

    def route(self, request_text: str) -> DecisionRecord:
        normalized = request_text.lower().strip()
        trace: List[StepTrace] = []
        chosen_handler = None
        chosen_rule = None

        step = 0
        for rule_name, keyword, handler, priority in self.RULES:
            matched = keyword in normalized
            trace.append(StepTrace(
                step_index=step,
                rule_name=rule_name,
                matched=matched,
                details=f"Keyword '{keyword}' found={matched} at priority {priority}"
            ))
            step += 1
            if matched and chosen_handler is None:
                chosen_handler = handler
                chosen_rule = rule_name

        if chosen_handler is None:
            chosen_handler = "fallback"
            chosen_rule = "rule_fallback"
            trace.append(StepTrace(
                step_index=step,
                rule_name="rule_fallback",
                matched=True,
                details="No prior rule matched; activating fallback handler"
            ))

        return DecisionRecord(
            request_text=request_text,
            chosen_handler=chosen_handler,
            chosen_rule=chosen_rule,
            trace=trace
        )

def main():
    router = ComponentRouter()
    for req in ["Need help with billing", "Please calculate total", "Unknown query", "Help calculate this"]:
        rec = router.route(req)
        print(f"'{req}' -> handler={rec.chosen_handler} (rule={rec.chosen_rule})")

if __name__ == "__main__":
    main()
