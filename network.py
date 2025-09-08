from typing import Tuple
import torch
import torch.nn as nn



class RepresentationNet(nn.Module):
    """Representation nn to predict next hidden state from current raw state"""

    def __init__(self, input_size: int, num_planes: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, num_planes),
            nn.ReLU(),
            nn.Linear(num_planes, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Given raw state x, predict the hidden state."""
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        hidden_state = self.net(x)
        return hidden_state

class DynamicsNet(nn.Module):
    """Dynamics nn to predict following hidden state and reward, based upon output of RepresentationNet"""
    
    def __init__(
        self,
        num_actions: int,
        num_planes: int,
        hidden_dim: int,
        support_size: int,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions

        self.transition_net = nn.Sequential(
            nn.Linear(hidden_dim + num_actions, num_planes),
            nn.ReLU(),
            nn.Linear(num_planes, hidden_dim),
        )

        self.reward_net = nn.Sequential(
            nn.Linear(hidden_dim, num_planes),
            nn.ReLU(),
            nn.Linear(num_planes, support_size),
        )

    def forward(self, hidden_state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given hidden_state state and action, predict the state transition and reward."""

        assert hidden_state.shape[0] == action.shape[0]
        B = hidden_state.shape[0]

        # [batch_size, num_actions]
        onehot_action = torch.zeros((B, self.num_actions), dtype=torch.float32, device=hidden_state.device)
        onehot_action.scatter_(1, action, 1.0)
        x = torch.cat([hidden_state, onehot_action], dim=1)

        hidden_state = self.transition_net(x)
        reward_logits = self.reward_net(hidden_state)
        return hidden_state, reward_logits


class PredictionNet(nn.Module):
    """Prediction nn to take hidden state as an input and predict policy distribution for all possible actions and value estimate of hidden state"""
    
    def __init__(
        self,
        num_actions: int,
        num_planes: int,
        hidden_dim: int,
        support_size: int,
    ) -> None:
        super().__init__()

        self.policy_net = nn.Sequential(
            nn.Linear(hidden_dim, num_planes),
            nn.ReLU(),
            nn.Linear(num_planes, num_actions),
        )

        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, num_planes),
            nn.ReLU(),
            nn.Linear(num_planes, support_size),
        )

    def forward(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given hidden state, predict the action probability and state value."""

        # Predict action distributions wrt policy
        pi_logits = self.policy_net(hidden_state)

        # Predict winning probability for current player's perspective.
        value_logits = self.value_net(hidden_state)

        return pi_logits, value_logits

    
class MuZeroNet(nn.Module):
    