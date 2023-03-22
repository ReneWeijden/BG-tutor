import torch
import gym

##------------------------------Constants for the environment-------------------------------------##
## ENVIRONMENT - The name of the environment to be used. This can be filled in by the user
##
## GYM_ENV - The constants file makes the gym environment according the given name, later in the constants
##           file this is used to retrieve information about the environment such as the amount of actions

ENVIRONMENT = 'CartPole-v1'
GYM_ENV     = gym.make(ENVIRONMENT)
MAXIMUM_REWARD_ENVIRONMENT = GYM_ENV.spec.reward_threshold

##------------------------------Constants for neural_networks-------------------------------------##
## DEVICE - Checks wheter the GPU is available, if so the GPU is used for the neural network. Else the CPU.
##
## OBSERVATION_SPACE - Returns the observation space of the environment, i.e. how many variables there are
##                     in the state of the environment. Supported environments: CartPole-v1, LunarLander-v2.
##                     This constant is generally used to determine the amount of inputs of a neural network
##
## ACTION_SPACE - Returns the amount of action that are possible in the chosen environment.
##                Supported environments: CartPole-v1, LunarLander-v2. This constant is generally used to
##                determine the amount of outputs of a neural network.
##
##LEARNING_RATE - A parameter used for the optimizer of a neural network. It determines how much the model
##                is tuned according to the calculated error of the model. See -> https://tinyurl.com/mry2z3kh
##
## GAMMA - To Do -> https://arxiv.org/pdf/1512.02011.pdf -> Increasing gamma rate?
##

DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
OBSERVATION_SPACE = len(GYM_ENV.observation_space.low)
ACTION_SPACE      = GYM_ENV.action_space.n
LEARNING_RATE     = 0.001
GAMMA             = 0.99
DQN_SEED          = 1432
HEBBIAN_SEED      = 1432
##------------------------------Constants for Replay Memory-------------------------------------##
## MEMORY_CAPACITY - Indicates how many experiences can be stored in the replay memory. This is set
##                   to a fixed value since not every experience should be used. After a fixed amount
##                   of experiences the old experience should be thrown away and replaced. In this
##                   way the agent doesn't learn from very old experience which are based on way
##                   different values than when the agent has learned a lot.
##                   Value is set to 1 million imitating the paper by Mnih et al,2015.
##                   https://dx.doi.org/10.1038/nature14236
##                   In this case the nunber is arbitrary and just copied from the paper.
##                   Choosing a different value can be done but has to be motivated
##
## BATCH_SIZE - This indicates the size of the batch that is sampled from the replay memory which is
##              used to perform a gradient update on the policy network. Based on the values of the batch the
##              algorithm computes the loss of the network and does a gradient update.
##              A batch consists of multiple samples in which each sample contains a random sample of:
##              state, action, next_state,reward
##              Value is set on 32 based on the recommendation of Bengio, 2012.
##              https://doi.org/10.48550/arXiv.1206.5533
##
##
MEMORY_CAPACITY = 1000000
BATCH_SIZE      = 32

##------------------------------Constants needed for the agent-------------------------------------------------------##
## EPSILON GENERAL - Epsilon is used for the epsilon greedy strategy in Q-learning. Epsilon is a variable used for
##                   determining the exploration rate of the agent. Depending on the value of the exploration rate
##                   the agent decides whether to explore or to exploits its environment. When exploring the agent
##                   will do a random action in it's environment to search for other rewards than already known. When
##                   exploiting the environment the agent will only do actions based on the Q-values it has found.
##                   By decaying the value of Epsilon over time the agent will at first explore alot, when Epsilon
##                   reaches lower and lower values the agent will explore less and less.
##
##
## TO SELF -> https://arxiv.org/pdf/1910.13701.pdf -> Reward based epsilon decay?
AMOUNT_OF_SAMPLES                      = 60
AMOUNT_OF_EPISODES_FOR_TRAINING        = 500
AMOUNT_EPISODES_TEST_CONVERGENCE       = 30
EPSILON_DECAY_EXP                      = 0.0085
EPSILON_END_EXP                        = 0.01         #The value for epsilon when we want to stop exploring
EPSILON_DECAY_LINEAR                   = 0.0025
EPSILON_END_LINEAR                     = 0.05
AMOUNT_EPISODES_BEFORE_TARGET_UPDATE   = 20
AMOUNT_EPISODES_BEFORE_POLICY_NET_EVAL = 1
EPSILON_START                          = 1            #The epsilon value we want to start with indicating exploration

##------------------------------Constants for hebbian calculations-------------------------------------##
## #TO-DO
##
##
##
##
HEBB_LEARNING_RATE = 0.5
HEBB_DECAY = 0.005
HEBB_START = 1
HEBB_END = 0

##------------------------------Constants for plotting-------------------------------------##
## #TO-DO
##
##
##
##
LINEAR_HIST_COLOR = "#2E6171"
EXPONENTIAL_HIST_COLOR = "#FF9505"
SUPERVISED_HIST_COLOR = "#37000A" #"#7E007B"
OVERLAP_HIST_COLOR = "#4A4A48"
AXIS_FONT_SIZE = 13
LEGEND_FONT_SIZE = 11