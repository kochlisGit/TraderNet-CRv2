import numpy as np
from abc import ABC, abstractmethod


class Rule(ABC):
    @abstractmethod
    def filter(self, action: int) -> int:
        pass
