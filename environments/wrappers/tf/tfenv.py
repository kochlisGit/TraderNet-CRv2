import numpy as np
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.specs.array_spec import BoundedArraySpec
from tf_agents.trajectories import time_step
from environments.actions import Action
from environments.environment import TradingEnvironment
from metrics.metric import Metric
from rules.rule import Rule


class TFTradingEnvironment(PyEnvironment):
    def __init__(self, env: TradingEnvironment):
        super().__init__()

        self._env = env

        self._action_spec = BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=len(Action) - 1, name='action'
        )
        self._observation_spec = BoundedArraySpec(
            shape=self._env.observation_space.shape, dtype=self._env.observation_space.dtype,
            minimum=self._env.observation_space.low, maximum=self._env.observation_space.high, name='observation'
        )
        self._done = False
        self._discount_rate = 1.0

    def get_metrics(self) -> list[Metric]:
        return self._env.metrics

    def get_episode_metrics(self) -> dict[str, float]:
        metrics = self._env.metrics
        return {metric.name: metric.result() for metric in metrics}

    def action_spec(self) -> BoundedArraySpec:
        return self._action_spec

    def observation_spec(self) -> BoundedArraySpec:
        return self._observation_spec

    def _reset(self) -> time_step.TimeStep:
        self._done = False
        observation = self._env.reset()
        return time_step.restart(observation=observation)

    def _step(self, action_spec) -> time_step.TimeStep:
        if self._done:
            return self._reset()

        action = action_spec.item()
        next_observation, reward, self._done = self._env.step(action)

        if self._done:
            return time_step.termination(observation=next_observation, reward=reward)
        else:
            return time_step.transition(observation=next_observation, reward=reward, discount=self._discount_rate)

    def render(self, **kwargs):
        self._env.render()


class TFRuleTradingEnvironment(TFTradingEnvironment):
    def __init__(self, env: TradingEnvironment, rules: list[Rule] or None):
        super().__init__(env=env)

        self._rules = rules

    def _step(self, action_spec) -> time_step.TimeStep:
        if self._done:
            return self._reset()

        action = action_spec.item()

        if self._rules is not None:
            for rule in self._rules:
                action = rule.filter(action=action)

        next_observation, reward, self._done = self._env.step(action)

        if self._done:
            return time_step.termination(observation=next_observation, reward=reward)
        else:
            return time_step.transition(observation=next_observation, reward=reward, discount=self._discount_rate)
