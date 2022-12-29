from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def train(self, **kwargs) -> list:
        pass

    @abstractmethod
    def eval(self, **kwargs) -> float:
        pass

    @abstractmethod
    def compute_action(self, **kwargs):
        pass

    @abstractmethod
    def get_action_probabilities(self, **kwargs):
        pass
