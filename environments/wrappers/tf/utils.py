from tf_agents.environments import utils
from tf_agents.environments.py_environment import PyEnvironment
from typing import Callable


def validate_environment(env: PyEnvironment, episodes: int, observation_action_splitter: Callable or None = None):
    utils.validate_py_environment(
        environment=env,
        episodes=episodes,
        observation_and_action_constraint_splitter=observation_action_splitter
    )
