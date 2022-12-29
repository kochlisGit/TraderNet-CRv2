import tensorflow as tf
from typing import Callable
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers.tf_driver import TFDriver
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.utils.common import function
from agents.tfagents.tfagent import TFAgentBase
from agents.tfagents.networks.q_network import QNetwork


class DQNAgent(TFAgentBase):
    def __init__(
            self,
            input_tensor_spec,
            action_spec,
            time_step_spec,
            env_batch_size: int,
            checkpoint_filepath: str,
            fc_layers: list or None,
            conv_layers: list[tuple[int, int, int]] or None = None,
            conv_type: str = '1d',
            preprocessing_layers: list or dict = None,
            preprocessing_combiner: list or dict or Callable = None,
            n_step: int = 3,
            double_dqn: bool = True,
            tau: float = 0.005,
            target_update_steps: int = 1,
            epsilon_init: float = 0.1,
            epsilon_min: float = 0.1,
            epsilon_decay_steps: int = 0,
            gamma: float = 0.99,
            replay_memory_capacity: int = 50000,
            replay_memory_batch_size: int = 64,
            initial_collect_steps: int = 10000,
            collection_steps_per_iteration: int = 1,
            train_step_counter: tf.Variable = tf.Variable(initial_value=0)
    ):
        assert epsilon_decay_steps == 0 or \
               epsilon_decay_steps > 0 and epsilon_init is not None and epsilon_min < epsilon_init <= 1.0, \
            'AssertionError: epsilon_decay_steps is expected to be zero or epsilon_min < epsilon_init <= 1.0' \
            f'got epsilon_decay_steps={epsilon_decay_steps}, epsilon_init={epsilon_init}, epsilon_min={epsilon_min}'

        if epsilon_decay_steps > 0:
            epsilon_greedy_fn = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=epsilon_init,
                decay_steps=epsilon_decay_steps,
                end_learning_rate=epsilon_min
            )
            epsilon_greedy = lambda: epsilon_greedy_fn(train_step_counter)
        else:
            epsilon_greedy = epsilon_init

        q_network = QNetwork(
            input_tensor_spec=input_tensor_spec,
            action_spec=action_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            fc_layer_params=fc_layers,
            conv_layer_params=conv_layers,
            conv_type=conv_type,
            activation_fn='gelu'
        )
        target_network = QNetwork(
            input_tensor_spec=input_tensor_spec,
            action_spec=action_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            fc_layer_params=fc_layers,
            conv_layer_params=conv_layers,
            conv_type=conv_type,
            activation_fn='gelu'
        )

        if double_dqn:
            agent = dqn_agent.DdqnAgent(
                time_step_spec=time_step_spec,
                action_spec=action_spec,
                q_network=q_network,
                target_q_network=target_network,
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                n_step_update=n_step,
                target_update_tau=tau,
                target_update_period=target_update_steps,
                gamma=gamma,
                epsilon_greedy=epsilon_greedy,
                train_step_counter=train_step_counter
            )
        else:
            agent = dqn_agent.DqnAgent(
                time_step_spec=time_step_spec,
                action_spec=action_spec,
                q_network=q_network,
                target_q_network=target_network,
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                n_step_update=n_step,
                target_update_tau=tau,
                target_update_period=target_update_steps,
                gamma=gamma,
                epsilon_greedy=epsilon_greedy,
                train_step_counter=train_step_counter
            )

        super().__init__(
            agent=agent,
            checkpoint_filepath=checkpoint_filepath,
            env_batch_size=env_batch_size,
            replay_memory_capacity=replay_memory_capacity,
            replay_memory_batch_size=replay_memory_batch_size,
            trajectory_num_steps=n_step,
            clear_memory_after_train_iteration=False
        )

        self._n_step = n_step
        self._initial_collect_steps = initial_collect_steps
        self._collection_steps_per_iteration = collection_steps_per_iteration

    def _get_collect_driver(self, train_env: TFPyEnvironment, observers: list) -> TFDriver:
        return super()._get_step_driver(
            env=train_env,
            policy=self.collect_policy,
            observers=observers,
            num_steps=self._collection_steps_per_iteration * self._n_step
        )

    def _train_step_fn(self, replay_memory: TFUniformReplayBuffer, dataset_iter: iter) -> float:
        trajectories, _ = next(dataset_iter)
        return self._agent.train(trajectories).loss

    def _get_train_step_fn(self) -> Callable:
        return function(self._train_step_fn)

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
        print('Collecting Initial Samples...')

        initial_collect_policy = RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())
        initial_collect_driver = super()._get_step_driver(
            env=train_env,
            policy=initial_collect_policy,
            observers=[replay_memory.add_batch],
            num_steps=self._initial_collect_steps
        )
        initial_collect_driver.run = function(initial_collect_driver.run)
        initial_collect_driver.run()

        return super()._train(
            train_env=train_env,
            eval_env=eval_env,
            train_iterations=train_iterations,
            eval_episodes=eval_episodes,
            iterations_per_eval=iterations_per_eval,
            iterations_per_log=iterations_per_log,
            iterations_per_checkpoint=iterations_per_checkpoint,
            replay_memory=replay_memory,
            replay_memory_dataset=replay_memory_dataset,
            train_collect_driver=train_collect_driver,
            save_best_only=save_best_only
        )
