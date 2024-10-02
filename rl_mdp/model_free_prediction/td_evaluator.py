import numpy as np
from rl_mdp.mdp.abstract_mdp import AbstractMDP
from rl_mdp.model_free_prediction.abstract_evaluator import AbstractEvaluator
from rl_mdp.policy.abstract_policy import AbstractPolicy


class TDEvaluator(AbstractEvaluator):
    def __init__(self,
                 env: AbstractMDP,
                 alpha: float):
        """
        Initializes the TD(0) Evaluator.

        :param env: A mdp object.
        :param alpha: The step size.
        """
        self.env = env
        self.alpha = alpha
        self.value_fun = np.zeros(self.env.num_states)    # Estimate of state-value function.

    def evaluate(self, policy: AbstractPolicy, num_episodes: int) -> np.ndarray:
        """
        Perform the TD prediction algorithm.

        :param policy: A policy object that provides action probabilities for each state.
        :param num_episodes: Number of episodes to run for estimating V(s).
        :return: The state-value function V(s) for the associated policy.
        """
        self.value_fun.fill(0)              # Reset value function.

        for _ in range(num_episodes):
            self._update_value_function(policy)

        return self.value_fun.copy()

    def _update_value_function(self, policy: AbstractPolicy) -> None:
        """
        Runs a single episode using the TD(0) method to update the value function.
        This is for a single episode.
        
        :param policy: A policy object that provides action probabilities for each state.
        """
        
        # slide 22
        
        done: bool = False
        S: int = self.env.reset()
        
        while not done:
            # action given by pi for S
            A: int = policy.sample_action(S)

            # take action A, observe R, S'
            next_state: int = None
            reward: float = None
            done: bool = None
            next_state, reward, done = self.env.step(A)

            # delta = R + gamma * V(S') - V(S)
            delta: float = None
            delta = reward + self.env.discount_factor * self.value_fun[next_state] - self.value_fun[S]

            # V(s) <-- V(s) + alpha * delta
            self.value_fun[S] = self.value_fun[S] + self.alpha * delta

            S = next_state
