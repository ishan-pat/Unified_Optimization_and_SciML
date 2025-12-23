"""Composite stopping criteria (logical combinations)."""

from typing import Any, Dict, List
import jax.numpy as jnp
from unified_opt.core.objective import Objective
from unified_opt.core.stopping_rule import StoppingRule


class CompositeStopping(StoppingRule):
    """
    Combine multiple stopping rules with logical operators.
    
    Supports AND and OR combinations.
    """
    
    def __init__(
        self,
        rules: List[StoppingRule],
        operator: str = 'OR'
    ):
        """
        Initialize composite stopping rule.
        
        Args:
            rules: List of stopping rules to combine
            operator: 'AND' or 'OR' (default: 'OR')
                     - 'OR': stops if any rule triggers
                     - 'AND': stops only if all rules trigger
        """
        self.rules = rules
        self.operator = operator.upper()
        if self.operator not in ['AND', 'OR']:
            raise ValueError(f"Operator must be 'AND' or 'OR', got {operator}")
    
    def should_stop(
        self,
        x: jnp.ndarray,
        objective: Objective,
        iteration: int,
        history: Dict[str, Any] | None = None,
    ) -> tuple[bool, Dict[str, Any]]:
        """Check composite stopping criteria."""
        results = []
        all_info = {}
        
        for i, rule in enumerate(self.rules):
            should_stop, info = rule.should_stop(x, objective, iteration, history)
            results.append(should_stop)
            all_info[f'rule_{i}'] = info
        
        # Combine results
        if self.operator == 'OR':
            should_stop = any(results)
        else:  # AND
            should_stop = all(results)
        
        if should_stop:
            # Find which rule(s) triggered
            triggered = [i for i, r in enumerate(results) if r]
            all_info['reason'] = f'composite_{self.operator}'
            all_info['triggered_rules'] = triggered
        
        return should_stop, all_info

