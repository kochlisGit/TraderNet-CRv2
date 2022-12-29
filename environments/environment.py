import gym
import numpy as np
from environments.actions import Action
from metrics.metric import Metric


class TradingEnvironment(gym.Env):
    def __init__(self, env_config: dict):
        assert 'states' in env_config, 'AssertionError: Expected "states" in env_config'
        assert 'reward_fn' in env_config, 'AssertionError: Expected "reward_function" in env_config'
        assert 'episode_steps' in env_config, 'AssertionError: Expected "episode_steps" in env_config'
        assert 'metrics' in env_config, 'AssertionError: Expected "metrics" in env_config'

        self._states = env_config['states']
        self._reward_function = env_config['reward_fn']
        self._episode_steps = env_config['episode_steps']

        self._metrics = env_config['metrics']

        if self._metrics is None:
            self._metrics = []

        self._num_states = self._states.shape[0] - 1

        assert self._num_states >= self._episode_steps, \
            'AssertionError: Not enough states are provided in the environment: ' \
            f'num_states = {self._num_states}, episode_steps = {self._episode_steps}'

        self._state_index = 0

        self.observation_space = gym.spaces.Box(
            low=self._states.min(axis=0),
            high=self._states.max(axis=0),
            shape=self._states.shape[1:],
            dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(n=len(Action))

        assert self._states.dtype == self.observation_space.dtype, \
            f'AssertionError: Expected states to have dtype = {self.observation_space.dtype}, got {self._states.dtype}'

    @property
    def metrics(self) -> list[Metric]:
        return self._metrics

    def update_metrics(self, log_pnl: float):
        for metric in self._metrics:
            metric.update(log_pnl=log_pnl)

    def register_metrics(self):
        for metric in self._metrics:
            metric.register()

    def reset(self) -> np.ndarray:
        for metric in self._metrics:
            metric.reset()
        return self._states[self._state_index]

    def step(self, action: int) -> (np.ndarray, float, bool):
        reward = self._reward_function.get_reward(i=self._state_index, action=action)

        self._state_index += 1
        next_state = self._states[self._state_index]

        if self._state_index == self._num_states:
            done = True
            self._state_index = 0
        elif self._state_index % self._episode_steps == 0:
            done = True
        else:
            done = False

        log_pnl = 0.0 if action == Action.HOLD.value else reward
        self.update_metrics(log_pnl=log_pnl)

        if done:
            self.register_metrics()
        return next_state, reward, done

    def render(self, mode=None):
        print('\n--- Current State ---')
        print(self._states[self._state_index])
