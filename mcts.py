from dataclasses import dataclass
from typing import Dict, Optional
import math
import torch


@dataclass
class Node:
    prior: float
    visit_count: int = 0
    value_sum: float = 0.0
    children: Dict[int, "Node"] = None
    latent: Optional[torch.Tensor] = None
    reward: float = 0.0

    def __post_init__(self):
        if self.children is None:
            self.children = {}

    @property
    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0


def ucb_score(parent: Node, child: Node, c_puct: float = 1.25) -> float:
    prior_score = c_puct * child.prior * math.sqrt(parent.visit_count + 1) / (child.visit_count + 1)
    return child.value + prior_score


def select_child(node: Node) -> int:
    return max(node.children.items(), key=lambda kv: ucb_score(node, kv[1]))[0]


def run_mcts(model, root_latent: torch.Tensor, priors: torch.Tensor, num_simulations: int):
    root = Node(prior=1.0, latent=root_latent)
    # Expand root
    for a in range(priors.shape[-1]):
        root.children[a] = Node(prior=float(torch.softmax(priors, -1)[a]))

    for _ in range(num_simulations):
        node = root
        search_path = [node]
        actions_taken = []

        # SELECTION
        while len(node.children) > 0:
            a = select_child(node)
            actions_taken.append(a)
            node = node.children[a]
            search_path.append(node)
            if node.latent is None:
                break

        # EXPANSION + EVALUATION
        parent = search_path[-2] if len(search_path) >= 2 else None
        parent_latent = parent.latent if parent is not None else root.latent
        a_idx = torch.tensor([actions_taken[-1]], dtype=torch.long, device=parent_latent.device)
        next_latent, reward, policy_logits, value = model.recurrent_inference(parent_latent, a_idx)
        node.latent = next_latent
        priors = torch.softmax(policy_logits.squeeze(0), dim=-1)
        for a in range(priors.shape[-1]):
            if a not in node.children:
                node.children[a] = Node(prior=float(priors[a]))
        node.reward = float(reward.item())

        # BACKUP
        bootstrap_value = float(value.item())
        for bnode in reversed(search_path):
            bnode.value_sum += bootstrap_value
            bnode.visit_count += 1
            bootstrap_value = bnode.reward + bootstrap_value  # discounted backup could be added

    # Return visit-count distribution as improved policy
    visits = torch.tensor([child.visit_count for _, child in sorted(root.children.items())], dtype=torch.float32)
    policy = visits / visits.sum()
    return policy, root