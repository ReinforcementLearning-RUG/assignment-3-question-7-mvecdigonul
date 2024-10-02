from rl_mdp.util import create_mdp, create_policy_1, create_policy_2
from rl_mdp.model_free_prediction.monte_carlo_evaluator import MCEvaluator
from rl_mdp.model_free_prediction.td_evaluator import TDEvaluator
from rl_mdp.model_free_prediction.td_lambda_evaluator import TDLambdaEvaluator
import numpy as np
import random

def main() -> None:
    """
    Starting point of the program, you can instantiate any classes, run methods/functions here as needed.
    """
    
    # set seed
    np.random.seed(42)
    random.seed(42)

    mdp = create_mdp()

    policy_1 = create_policy_1()
    policy_2 = create_policy_2()

    
    
    ### A - run MC first visit for 1000 episodes
    print("Monte Carlo First Visit")
    mc_evaluator = MCEvaluator(mdp)
    mc_p1_eval = mc_evaluator.evaluate(policy_1, 1000).tolist()
    mc_p2_eval = mc_evaluator.evaluate(policy_2, 1000).tolist()
    
    print("The value function for policy 1 is: ", mc_p1_eval)
    print("The value function for policy 2 is: ", mc_p2_eval)
    print("----------------------------------")
    
    
    ### B - run TD(0) for 1000 episodes, alpha = 0.1
    print("TD(0)")
    td0_evaluator = TDEvaluator(mdp, 0.1)
    td0_p1_eval = td0_evaluator.evaluate(policy_1, 1000).tolist()
    td0_p2_eval = td0_evaluator.evaluate(policy_2, 1000).tolist()
    
    print("The value function for policy 1 is: ", td0_p1_eval)
    print("The value function for policy 2 is: ", td0_p2_eval)
    print("----------------------------------")


    
    ### C - run TD(λ) for 1000 episodes, alpha = 0.1, lambda = 0.5
    print("TD(λ)")
    td_lambda_evaluator = TDLambdaEvaluator(mdp, 0.1, 0.5)
    td_lambda_p1_eval = td_lambda_evaluator.evaluate(policy_1, 1000).tolist()
    td_lambda_p2_eval = td_lambda_evaluator.evaluate(policy_2, 1000).tolist()
    
    print("The value function for policy 1 is: ", td_lambda_p1_eval)
    print("The value function for policy 2 is: ", td_lambda_p2_eval)
    print("----------------------------------")

if __name__ == "__main__":
    main()
