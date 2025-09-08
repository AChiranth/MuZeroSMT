from typing import Tuple, List, Optional
import z3


class Z3TacticEnv:
    """MuZero-like environment wrapper around Z3 Goals and Tactics.

    State  := z3.Goal
    Action := tactic name (str) drawn from config.tactic_set
    Reward := +1 when solved (goal reduces to empty); -1 on invalid move/dead-end; 0 otherwise
    Done   := True when solved or failure/limit reached
    """

    def __init__(self, root_expr: z3.BoolRef, tactic_names: List[str], max_depth: int = 20):
        self.tactic_names = tactic_names
        self.max_depth = max_depth
        self._init_goal(root_expr)
        self.steps = 0

    def _init_goal(self, root_expr: z3.BoolRef):
        self.goal = z3.Goal()
        self.goal.add(root_expr)
        self.done = False
        self.last_status: Optional[str] = None

    def clone(self) -> "Z3TacticEnv":
        # Note: Goal deep copy via string-serialize-parse to be safe
        cloned = Z3TacticEnv(z3.And([f for f in self.goal]), self.tactic_names, self.max_depth)
        cloned.steps = self.steps
        cloned.done = self.done
        cloned.last_status = self.last_status
        return cloned

    def available_actions(self) -> List[str]:
        if self.done:
            return []
        return list(self.tactic_names)

    def step(self, action: str) -> Tuple[z3.Goal, float, bool, dict]:
        if self.done:
            return self.goal, 0.0, True, {"status": self.last_status}

        self.steps += 1
        info = {}
        try:
            tactic = z3.Tactic(action)
            result = tactic(self.goal)
            if len(result) == 0:
                # Solved/closed
                self.done = True
                self.last_status = "solved"
                reward = 1.0
                self.goal = z3.Goal()  # empty
            else:
                # Continue with first sub-goal (simple baseline); could branch later
                self.goal = result[0]
                reward = 0.0
                self.last_status = "progress"
        except z3.Z3Exception as e:
            reward = -1.0
            self.done = True
            self.last_status = f"exception:{type(e).__name__}"
            info["error"] = str(e)

        if self.steps >= self.max_depth and not self.done:
            self.done = True
            self.last_status = "depth_limit"
            # small penalty for exceeding depth
            reward -= 0.1

        info["status"] = self.last_status
        return self.goal, reward, self.done, info