import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from typing import Callable
from tf_agents.drivers.tf_driver import TFDriver
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.metrics.tf_metrics import AverageReturnMetric
from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.utils.common import Checkpointer, function
from tf_agents.agents.tf_agent import TFAgent
from agents.agent import Agent


class TFAgentBase(Agent, ABC):
    def __init__(
            self,
            agent: TFAgent,
            checkpoint_filepath: str,
            env_batch_size: int,
            replay_memory_capacity: int,
            replay_memory_batch_size: int,
            trajectory_num_steps: int,
            clear_memory_after_train_iteration: bool
    ):
        self._agent = agent
        self._checkpointer = Checkpointer(
            ckpt_dir=checkpoint_filepath,
            max_to_keep=1,
            agent=agent,
            policy=agent.policy
        )

        self._env_batch_size = env_batch_size
        self._replay_memory_capacity = replay_memory_capacity
        self._replay_memory_batch_size = replay_memory_batch_size
        self._trajectory_num_steps = trajectory_num_steps
        self._clear_memory_after_train_iteration = clear_memory_after_train_iteration

    @property
    def train_step_counter(self) -> tf.Variable:
        return self._agent.train_step_counter

    @property
    def policy(self):
        return self._agent.policy

    @property
    def collect_policy(self):
        return self._agent.collect_policy

    def initialize(self):
        self._agent.initialize()
        self._checkpointer.initialize_or_restore()

    def save(self):
        self._checkpointer.save(self._agent.train_step_counter.value())

    def load(self):
        self._checkpointer.initialize_or_restore()

    def _get_replay_memory(self) -> TFUniformReplayBuffer:
        return TFUniformReplayBuffer(
            data_spec=self._agent.collect_data_spec,
            batch_size=self._env_batch_size,
            max_length=self._replay_memory_capacity
        )

    def _get_replay_memory_dataset(self, replay_memory: TFUniformReplayBuffer) -> tf.data.Dataset:
        num_steps = self._trajectory_num_steps + 1

        return replay_memory.as_dataset(
            sample_batch_size=self._replay_memory_batch_size,
            num_steps=num_steps,
            num_parallel_calls=num_steps + 1,
        ).prefetch(num_steps + 1)

    def _get_step_driver(
            self,
            env: TFPyEnvironment,
            policy: TFPolicy,
            observers: list,
            num_steps: int
    ):
        return DynamicStepDriver(
            env=env,
            policy=policy,
            observers=observers,
            num_steps=num_steps
        )

    def _get_episode_driver(
            self,
            env: TFPyEnvironment,
            policy: TFPolicy,
            observers: list,
            num_episodes: int
    ):
        return DynamicEpisodeDriver(env=env, policy=policy, observers=observers, num_episodes=num_episodes)

    @abstractmethod
    def _get_collect_driver(self, train_env: TFPyEnvironment, observers: list) -> TFDriver:
        pass

    @abstractmethod
    def _get_train_step_fn(self) -> Callable:
        pass

    def train(
            self,
            train_env: TFPyEnvironment,
            eval_env: TFPyEnvironment,
            train_iterations: int,
            eval_episodes: int,
            iterations_per_eval: int,
            iterations_per_log: int,
            iterations_per_checkpoint: int,
            save_best_only: bool
    ) -> list:
        assert train_env.batch_size == self._env_batch_size, \
            f'AssertionError: Expected environment batch size to be {self._env_batch_size}, got {train_env.batch_size}'

        replay_memory = TFUniformReplayBuffer(
            data_spec=self._agent.collect_data_spec,
            batch_size=self._env_batch_size,
            max_length=self._replay_memory_capacity
        )
        dataset = replay_memory.as_dataset(
            sample_batch_size=self._replay_memory_batch_size,
            num_steps=None if self._trajectory_num_steps is None else self._trajectory_num_steps + 1,
            num_parallel_calls=self._trajectory_num_steps
        ).prefetch(1 if self._trajectory_num_steps is None else self._trajectory_num_steps)
        collect_driver = self._get_collect_driver(train_env=train_env, observers=[replay_memory.add_batch])
        collect_driver.run = function(collect_driver.run)

        return self._train(
            train_env=train_env,
            eval_env=eval_env,
            train_iterations=train_iterations,
            eval_episodes=eval_episodes,
            iterations_per_eval=iterations_per_eval,
            iterations_per_log=iterations_per_log,
            iterations_per_checkpoint=iterations_per_checkpoint,
            replay_memory=replay_memory,
            replay_memory_dataset=dataset,
            train_collect_driver=collect_driver,
            save_best_only=save_best_only
        )

    def _train(
            self,
            train_env: TFPyEnvironment,
            eval_env: TFPyEnvironment,
            train_iterations: int,
            eval_episodes: int,
            iterations_per_eval: int,
            iterations_per_log: int,
            iterations_per_checkpoint: int,
            replay_memory: TFUniformReplayBuffer,
            replay_memory_dataset: tf.data.Dataset,
            train_collect_driver: TFDriver,
            save_best_only: bool
    ) -> list:
        average_returns = []
        max_avg_reward = -np.inf

        eval_metric = AverageReturnMetric(batch_size=eval_env.batch_size, buffer_size=200)
        eval_env_driver = self._get_episode_driver(
            env=eval_env,
            policy=self._agent.policy,
            observers=[eval_metric],
            num_episodes=eval_episodes
        )
        dataset_iter = iter(replay_memory_dataset)

        eval_env_driver.run = function(eval_env_driver.run)
        self._agent.train = function(self._agent.train)
        train_step = self._get_train_step_fn()

        print('Training has started...')

        eval_env_driver.run()
        avg_reward = eval_metric.result().numpy()
        average_returns.append(avg_reward)
        eval_metric.reset()

        if max_avg_reward < avg_reward:
            max_avg_reward = avg_reward

            if save_best_only:
                print(
                    f'\nNew best average return found at {max_avg_reward}! '
                    f'Saving checkpoint at iteration {0}'
                )
                self.save()

        for i in range(1, train_iterations + 1):
            train_collect_driver.run()
            train_loss = train_step(replay_memory=replay_memory, dataset_iter=dataset_iter)

            if self._clear_memory_after_train_iteration:
                replay_memory.clear()

            if i % iterations_per_eval == 0:
                eval_env_driver.run()
                avg_reward = eval_metric.result().numpy()
                average_returns.append(avg_reward)
                eval_metric.reset()

                if max_avg_reward < avg_reward:
                    max_avg_reward = avg_reward

                    if save_best_only:
                        print(
                            f'\nNew best average return found at {max_avg_reward}! '
                            f'Saving checkpoint at iteration {i}'
                        )
                        self.save()

            if not save_best_only and i % iterations_per_checkpoint == 0:
                print(f'\nSaving checkpoint at iteration {i}')
                self.save()

            if i % iterations_per_log == 0:
                print(f'\nIteration: {i}'
                      f'\nTrain Loss: {train_loss}'
                      f'\nAverage Return: {avg_reward}')
        return average_returns

    def eval(self, eval_env: TFPyEnvironment, num_episodes: int) -> float:
        eval_metric = AverageReturnMetric(batch_size=eval_env.batch_size, buffer_size=200)

        eval_env_driver = self._get_episode_driver(
            env=eval_env,
            policy=self._agent.policy,
            observers=[eval_metric],
            num_episodes=num_episodes
        )
        eval_env_driver.run = function(eval_env_driver.run)
        eval_env_driver.run()
        return eval_metric.result().numpy()

    def _get_action_step(self, time_step, policy_state=None, use_greedy_policy: bool = True):
        policy = self._agent.policy if use_greedy_policy else self._agent.collect_policy

        if policy_state is None:
            policy_state = policy.get_initial_state(batch_size=self._env_batch_size)
        return policy.action(time_step=time_step, policy_state=policy_state)

    def compute_action(self, time_step, policy_state=None, use_greedy_policy: bool = True) -> int:
        return self._get_action_step(
            time_step=time_step,
            policy_state=policy_state,
            use_greedy_policy=use_greedy_policy
        ).action

    def get_action_probabilities(self, time_step, policy_state=None, use_greedy_policy: bool = True) -> np.ndarray:
        return self._get_action_step(
            time_step=time_step,
            policy_state=policy_state,
            use_greedy_policy=use_greedy_policy
        ).info
