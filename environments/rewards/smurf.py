from environments.rewards.function import RewardFunction


class SmurfRewardFunction:
    def __init__(
            self,
            reward_function: RewardFunction
    ):
        self._reward_function = reward_function

        smurf_rf = self._reward_function.reward_fn
        smurf_rf[:, 2] = 0.0055
        self._reward_function.reward_fn = smurf_rf
        print(self._reward_function.reward_fn[0])

    def __call__(self, i: int, action: int) -> float:
        return self.get_reward(i=i, action=action)

    def get_reward(self, i: int, action: int) -> float:
        return self._reward_function.get_reward(i=i, action=action)

    def get_reward_fn_shape(self):
        return self._reward_function.get_reward_fn_shape()
