from collections import deque
from typing import Deque, Dict, List
import random


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: Deque[Dict] = deque(maxlen=capacity)

    def add(self, item: Dict):
        self.buffer.append(item)

    def sample(self, batch_size: int) -> List[Dict]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)