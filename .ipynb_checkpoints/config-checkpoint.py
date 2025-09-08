from dataclasses import dataclass
from typing import List


@dataclass
class MuZeroConfig:
    # Training
    seed: int = 42
    device: str = "mps"
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    total_training_steps: int = 100_000

    # Self-play / search
    num_simulations: int = 50
    dirichlet_alpha: float = 0.3
    exploration_fraction: float = 0.25
    discount: float = 0.997
    temperature_schedule: List[float] = None  # e.g., [1.0, 1.0, 0.5, 0.25]

    # Replay buffer
    replay_buffer_size: int = 100_000
    td_steps: int = 5
    num_unroll_steps: int = 5

    # Model
    latent_dim: int = 128
    action_dim: int = 32  # len(tactic_set)
    hidden_dim: int = 256

    # Z3 / Environment
    max_depth: int = 20
    tactic_set: List[str] = None

    def __post_init__(self):
        if self.tactic_set is None:
            self.tactic_set = [
                # Keep this small initially; expand as learning stabilizes
                "simplify", "solve-eqs", "elim-uncnstr", "propagate-values",
                "split-clause", "tseitin-cnf"
            ]
        if self.temperature_schedule is None:
            self.temperature_schedule = [1.0, 1.0, 0.5, 0.25]
        self.action_dim = len(self.tactic_set)