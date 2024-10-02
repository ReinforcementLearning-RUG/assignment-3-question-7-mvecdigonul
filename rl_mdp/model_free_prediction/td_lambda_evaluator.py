import numpy as np
from rl_mdp.mdp.abstract_mdp import AbstractMDP
from rl_mdp.model_free_prediction.abstract_evaluator import AbstractEvaluator
from rl_mdp.policy.abstract_policy import AbstractPolicy


class TDLambdaEvaluator(AbstractEvaluator):
    def __init__(self,
                 env: AbstractMDP,
                 alpha: float,
                 lambd: float):
        """
        Initializes the TD(λ) Evaluator.

        :param env: A mdp object.
        :param alpha: The step size.
        :param lambd: The trace decay parameter (λ).
        """
        self.env = env
        self.alpha = alpha
        self.lambd = lambd
        self.value_fun = np.zeros(self.env.num_states)    # Estimate of state-value function.
        self.eligibility_traces = np.zeros(self.env.num_states)

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
        Runs a single episode using the TD(λ) method to update the value function.

        :param policy: A policy object that provides action probabilities for each state.
        """
        done: bool = False
        
        # pseudo code in the question
        
        S: int = self.env.reset()
        et: np.ndarray = np.zeros(self.env.num_states)
        
        while not done: 
            A: int = policy.sample_action(S)

            next_state: int = None
            reward: float = None
            done: bool = None
            next_state, reward, done = self.env.step(A)

            # Take action A and Observe R and S' from the environment
            delta: float = None
            delta = reward + self.env.discount_factor * self.value_fun[next_state] - self.value_fun[S]
            
            et[S] = et[S] + 1 # update eligibility trace
            
            for s in range(self.env.num_states):
                # V(s) <-- V(s) + alpha * delta * e(s)
                self.value_fun[s] = self.value_fun[s] + self.alpha * delta * et[s]

                # e(s) <-- gamma * lambda * e(s)
                et[s] = self.env.discount_factor * self.lambd * et[s]

            S = next_state
