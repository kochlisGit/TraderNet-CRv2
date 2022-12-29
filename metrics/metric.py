from abc import ABC, abstractmethod


class Metric(ABC):
    def __init__(self, name: str):
        self._name = name
        self._episode_metrics = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def episode_metrics(self) -> list[float]:
        return self._episode_metrics

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self, log_pnl: float):
        pass

    @abstractmethod
    def result(self) -> float:
        pass

    def register(self):
        self._episode_metrics.append(self.result())
