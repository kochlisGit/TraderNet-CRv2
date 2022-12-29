import math
import tensorflow as tf
from typing import Callable
from tf_agents.agents.ppo.ppo_clip_agent import PPOClipAgent
from tf_agents.drivers.tf_driver import TFDriver
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.utils.common import function
from agents.tfagents.tfagent import TFAgentBase
from agents.tfagents.networks.actor_distribution_network import ActorDistributionNetwork
from agents.tfagents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork
from agents.tfagents.networks.value_network import ValueNetwork
from agents.tfagents.networks.value_rnn_network import ValueRnnNetwork


class PPOAgent(TFAgentBase):
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
            lstm_layers: list[int] or None = None,
            train_sequence_len: int = 1,
            preprocessing_layers: list or dict = None,
            preprocessing_combiner: list or dict or Callable = None,
            greedy_eval: bool = True,
            epsilon_clipping: float = 0.3,
            lambda_value: float = 0.95,
            entropy_regularization_init: float = 0.0,
            entropy_regularization_min: float = 0.0,
            entropy_regularization_decay_steps: int = 0,
            num_epochs: int = 40,
            use_gae: bool = True,
            gamma: float = 0.99,
            replay_memory_capacity: int = 1000,
            replay_memory_batch_size: int or None = None,
            collection_episodes_per_iteration: int = 5,
            train_step_counter: tf.Variable = tf.Variable(initial_value=0)
    ):
        assert train_sequence_len >= 1, \
            f'AssertionError: train_sequence_len is expected to be >= 1, got {train_sequence_len}'
        assert lstm_layers is None or train_sequence_len > 1, \
            'AssertionError: train_sequence_len is expected to be greater than 1 if lstm_layers is not None'

        assert entropy_regularization_decay_steps == 0 or entropy_regularization_init == 0.0 or \
               entropy_regularization_min < entropy_regularization_init <= 1.0, \
            'AssertionError: entropy_regularization_decay_steps is expected to be zero ' \
            'or entropy_regularization_init is expected to be zero or ' \
            'or entropy_regularization_min < entropy_regularization_init <= 1.0' \
            f'got entropy_regularization_decay_steps={entropy_regularization_decay_steps}, ' \
            f'entropy_regularization_init={entropy_regularization_init}, ' \
            f'entropy_regularization_min={entropy_regularization_min}'

        if entropy_regularization_decay_steps > 0:
            entropy_regularization_fn = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=entropy_regularization_init,
                decay_steps=entropy_regularization_decay_steps,
                end_learning_rate=entropy_regularization_min
            )
            entropy_regularization = lambda: entropy_regularization_fn(train_step_counter)
        else:
            entropy_regularization = entropy_regularization_init

        if lstm_layers is None:
            actor_network = ActorDistributionNetwork(
                input_tensor_spec=input_tensor_spec,
                output_tensor_spec=action_spec,
                preprocessing_layers=preprocessing_layers,
                preprocessing_combiner=preprocessing_combiner,
                fc_layer_params=fc_layers,
                conv_layer_params=conv_layers,
                conv_type=conv_type,
                activation_fn='gelu'
            )
            value_network = ValueNetwork(
                input_tensor_spec=input_tensor_spec,
                preprocessing_layers=preprocessing_layers,
                preprocessing_combiner=preprocessing_combiner,
                fc_layer_params=fc_layers,
                conv_layer_params=conv_layers,
                conv_type=conv_type,
                activation_fn='gelu'
            )
        else:
            actor_network = ActorDistributionRnnNetwork(
                input_tensor_spec=input_tensor_spec,
                output_tensor_spec=action_spec,
                preprocessing_layers=preprocessing_layers,
                preprocessing_combiner=preprocessing_combiner,
                lstm_size=lstm_layers,
                output_fc_layer_params=fc_layers,
                conv_layer_params=conv_layers,
                conv_type=conv_type,
                activation_fn='gelu'
            )
            value_network = ValueRnnNetwork(
                input_tensor_spec=input_tensor_spec,
                preprocessing_layers=preprocessing_layers,
                preprocessing_combiner=preprocessing_combiner,
                lstm_size=lstm_layers,
                output_fc_layer_params=fc_layers,
                conv_layer_params=conv_layers,
                conv_type=conv_type,
                activation_fn='gelu'
            )

        agent = PPOClipAgent(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            actor_net=actor_network,
            value_net=value_network,
            greedy_eval=greedy_eval,
            importance_ratio_clipping=epsilon_clipping,
            lambda_value=lambda_value,
            discount_factor=gamma,
            entropy_regularization=entropy_regularization,
            num_epochs=num_epochs,
            use_gae=use_gae,
            train_step_counter=train_step_counter
        )

        super().__init__(
            agent=agent,
            checkpoint_filepath=checkpoint_filepath,
            env_batch_size=env_batch_size,
            replay_memory_capacity=replay_memory_capacity,
            replay_memory_batch_size=replay_memory_batch_size,
            trajectory_num_steps=train_sequence_len,
            clear_memory_after_train_iteration=True
        )

        self._replay_memory_batch_size = replay_memory_batch_size
        self._collection_episodes_per_iteration = collection_episodes_per_iteration

        self._optimized_minibatch_train_fn = function(self._train_step_minibatch_fn)

    def _get_collect_driver(self, train_env: TFPyEnvironment, observers: list) -> TFDriver:
        return super()._get_episode_driver(
            env=train_env,
            policy=self.collect_policy,
            observers=observers,
            num_episodes=self._collection_episodes_per_iteration
        )

    def _train_step_batch_fn(self, replay_memory: TFUniformReplayBuffer, dataset_iter: iter) -> float:
        trajectories = replay_memory.gather_all()
        return self._agent.train(experience=trajectories).loss

    def _train_step_minibatch_fn(self, dataset_iter: iter) -> float:
        trajectories, _ = next(dataset_iter)
        return self._agent.train(experience=trajectories).loss

    def _train_step_minibatch(self, replay_memory: TFUniformReplayBuffer, dataset_iter: iter) -> float:
        train_loss = 0
        optimized_minibatch_train_fn = self._optimized_minibatch_train_fn

        num_memory_items = replay_memory.num_frames()
        num_batches = math.ceil(num_memory_items / self._replay_memory_batch_size)

        for _ in range(num_batches):
            train_loss += optimized_minibatch_train_fn(dataset_iter=dataset_iter)
        return train_loss / num_batches

    def _get_train_step_fn(self) -> Callable:
        if self._replay_memory_batch_size is None:
            return function(self._train_step_batch_fn)
        else:
            return self._train_step_minibatch
