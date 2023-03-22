import constants as c
import random
import math
import torch
##------------------------------------------Agent class------------------------------------------------##
## GENERAL - The agent class is used to gather some important functions for choosing choosing actions in
##           the environment. It has all the methods needed for the decisionmaking part of the algorithm.
##           It determines:
##                   1. If the agent should explore or exploit the environment based on the Epsilon greedy
##                      strategy
##                   2. The action to take based on the policy network that is given.
##
##          While the agent class is small and the function could also directly be implemented in the main
##          file, it makes for cleaner code when setup in a seperate object class. Also this makes future
##          coding with different epsilon greedy strategies and action selection using different kind
##          of neural networks possible.
##
##
## Functions:
##
## determine_exploration_rate() - This function generates the exploration rate which is used to select
##                                whether the agent should explore or explore the environment. It is
##                                calculated using an exponential epsilon decay.
##
## select_action() - Depending on the exploration rate, this function returns the action the agent should
##                   take. The action selection is based on the exploration rate determined by the function
##                   determine_exploration_rate(). The select_action() function generates a random number
##                   which it compares to the exploratino rate. If the exploration rate is greater than the
##                   random number the function will select a random action. Otherwise the best possible action
##                   according to the policy_network is selected using the argmax of the output. Because the
##                   exploration rate decays over time the agent will tend to explore less after multiple
##                   episodes and exploit more.
##
##
class DQN_policyNet_targetNet_Agent():
    def __init__(self, policy_net):

        self.policy_network = policy_net
        self.exploration_rate = 0

    def determine_exploration_rate_exponential(self, episode):
        self.exploration_rate = c.EPSILON_END_EXP + (c.EPSILON_START - c.EPSILON_END_EXP) \
                           * math.exp(-1. * episode * c.EPSILON_DECAY_EXP)
        return

    def determine_exploration_rate_linear(self, episode):
        self.exploration_rate = c.EPSILON_START - (episode * c.EPSILON_DECAY_LINEAR)

        if self.exploration_rate < c.EPSILON_END_LINEAR:
            self.exploration_rate = c.EPSILON_END_LINEAR

        return

    def select_action(self, episode, state):
        self.determine_exploration_rate_linear(episode)
        random_number = random.random() #Select random number between 0 and 1

        if self.exploration_rate > random_number:
            action = random.randrange(c.ACTION_SPACE)
            return action

        with torch.no_grad(): #Disabling the gradient calculation saves memory capacity
            return self.policy_network.forward(state).argmax().item()



