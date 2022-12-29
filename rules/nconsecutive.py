import numpy as np
from environments.actions import Action
from rules.rule import Rule


class NConsecutive(Rule):
    def __init__(self, window_size: int):
        self._window_size = window_size
        self._actions_queue = []

    def filter(self, action: int) -> int:
        if len(self._actions_queue) < self._window_size:
            self._actions_queue.insert(0, action)
            return Action.HOLD.value

        self._actions_queue.pop(-1)
        self._actions_queue.insert(0, action)
        return action if len(set(self._actions_queue)) == 1 else Action.HOLD.value
