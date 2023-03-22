import torch
import constants as c
import random
import plots
import copy
import Hebbian_functions as hebb
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import os.path
import gym
from collections import deque
##------------------------------------------trainer.py------------------------------------------------##
## GENERAL - The trainer.py file facilitates training functions which can be used to let te agent explore
##           it's environment and compose a neural network fitting the solution to the environment.
##
## Variables:
##
##
def unique_file(basename, ext):
    actualname = "%s.%s" % (basename, ext)
    c = itertools.count()
    while os.path.exists(actualname):
        actualname = "%s (%d).%s" % (basename, next(c), ext)
    return actualname

def target_net_update_needed(episode_num):
    return not (episode_num % c.AMOUNT_EPISODES_BEFORE_TARGET_UPDATE)  # not is included here since % returns a 0 when update needed.

def policy_net_evaluation_needed(episode_num):
    return not (episode_num % c.AMOUNT_EPISODES_BEFORE_POLICY_NET_EVAL)  # not is included here since % returns a 0 when update needed.

def test_for_convergence_needed(episode_num):
    return not (episode_num % c.AMOUNT_EPISODES_TEST_CONVERGENCE)  # not is included here since % returns a 0 when update needed.

def copy_policyNet_to_TargetNet(policy_net, target_net):
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

activations_layer_1 = deque([0])
activations_layer_2 = deque([0])

def get_layer_output_layer_1(module, input, output):
    out = output.detach().clone().to(c.DEVICE)
    activations_layer_1.pop()
    activations_layer_1.append(out)
    return

def get_layer_output_layer_2(module, input, output):
    out = output.detach().clone().to(c.DEVICE)
    activations_layer_2.pop()
    activations_layer_2.append(out)
    return

def get_best_weights(weight_1, weight_2, network, network_layer, amount_test_rounds):

    score_weight_1 = 0
    score_weight_2 = 0
    environment = gym.make(c.ENVIRONMENT)

    #test for 10 episodes, the weights that achieve the highest score do achieve the best on average.
    hebb.install_new_weights_in_hebb_layer(network_layer, weight_1)
    for i in range(amount_test_rounds):
        score_weight_1 += test_performance_of_network(environment, network)

    hebb.install_new_weights_in_hebb_layer(network_layer, weight_2)
    for i in range(amount_test_rounds):
        score_weight_2 += test_performance_of_network(environment, network)

    if score_weight_1 > score_weight_2:
        #print("weight 1 was the best")
        environment.close()
        return weight_1

    elif score_weight_1 < score_weight_2:
        #print("weight 2 was the best")
        environment.close()
        return weight_2

    elif score_weight_1 == score_weight_2:
        #print("result was equal")
        environment.close()
        return weight_1

    else:
        print("something went wrong comparing the networks")


def get_best_weights_multilayer(weight_1_1, weight_1_2, weight_2_1, weight_2_2, network, amount_test_rounds):
    score_weight_1 = 0
    score_weight_2 = 0
    environment = gym.make(c.ENVIRONMENT)

    # test for 10 episodes, the weights that achieve the highest score do achieve the best on average.
    hebb.install_new_weights_in_hebb_layer(network.layer_1, weight_1_1)
    hebb.install_new_weights_in_hebb_layer(network.layer_2, weight_1_2)

    for i in range(amount_test_rounds):
        score_weight_1 += test_performance_of_network(environment, network)

    hebb.install_new_weights_in_hebb_layer(network.layer_1, weight_2_1)
    hebb.install_new_weights_in_hebb_layer(network.layer_2, weight_2_2)

    for i in range(amount_test_rounds):
        score_weight_2 += test_performance_of_network(environment, network)

    if score_weight_1 > score_weight_2:
        # print("weight 1 was the best")
        environment.close()
        return weight_1_1, weight_1_2

    elif score_weight_1 < score_weight_2:
        # print("weight 2 was the best")
        environment.close()
        return weight_2_1, weight_2_2

    elif score_weight_1 == score_weight_2:
        # print("result was equal")
        environment.close()
        return weight_1_1, weight_1_2

    else:
        print("something went wrong comparing the networks")
        return

def train_basic_DQN_policyNet_targetNet_compare(agent, environment, replay_memory, policy_network, target_network):

    scores = []
    epsilon_history = []
    previous_net = copy.deepcopy(policy_network)

    for episode in range(c.AMOUNT_OF_EPISODES_FOR_TRAINING):

        # reset the agent and environment, also put rewards to zero
        state = torch.tensor(environment.reset())
        done = False
        if episode >= 1:
            print("Current episode: {}, score: {}".format(episode, scores[episode-1]))
        score = 0

        if episode > 100:
            if policy_net_evaluation_needed(episode):
                policy_net_score, previous_net_score = compare_networks(environment, policy_network, previous_net)

                if policy_net_score < previous_net_score:
                    policy_network.load_state_dict(previous_net.state_dict())

            previous_net = copy.deepcopy(policy_network)

        if target_net_update_needed(episode):
            print("updating target net!")
            copy_policyNet_to_TargetNet(policy_network, target_network)

        while not done:

            #Let the agent select an action, do this action and observe the reward and new state
            action = agent.select_action(episode, state)
            new_state, reward, done, info = environment.step(action)

            replay_memory.store_experience(state, new_state, reward, action, done)
            #environment.render()

            replay_memory.update_networks_using_replay_memory(policy_network, target_network)

            score += reward # Keep track of the recieved reward during the episode to plot later
            state = torch.tensor(new_state)

            if done:
                break

        scores.append(score)
        epsilon_history.append(agent.exploration_rate)

    environment.close()
    plots.plot_learning_curve(c.AMOUNT_OF_EPISODES_FOR_TRAINING, scores, epsilon_history, 'Learning_curve_BG.png')
    print("Training of DQN(Basal Ganglia) network done!!")

def train_basic_DQN_policyNet_targetNet(agent, environment, replay_memory, policy_network, target_network):

    scores = []
    epsilon_history = []
    convergence = False
    episodes_till_convergence = 0

    for episode in range(c.AMOUNT_OF_EPISODES_FOR_TRAINING):

        # reset the agent and environment, also put rewards to zero
        state = torch.tensor(environment.reset())
        done = False
        if episode >= 1:
            print("Current episode: {}, score: {}".format(episode, scores[episode - 1]))
        score = 0

        while not done:

            #Let the agent select an action, do this action and observe the reward and new state
            action = agent.select_action(episode, state)
            new_state, reward, done, info = environment.step(action)

            replay_memory.store_experience(state, new_state, reward, action, done)
            #environment.render()

            replay_memory.update_networks_using_replay_memory(policy_network, target_network)

            score += reward # Keep track of the recieved reward during the episode to plot later
            state = torch.tensor(new_state)

            if done:
                break
        score = test_performance_of_network(environment, policy_network)
        scores.append(score)
        epsilon_history.append(agent.exploration_rate)

        if target_net_update_needed(episode):
            print("updating target net!")
            copy_policyNet_to_TargetNet(policy_network, target_network)

        if test_for_convergence_needed(episode):

            for i in range(10):
                reward = test_performance_of_network(environment, policy_network, 1, True)

                if reward < c.MAXIMUM_REWARD_ENVIRONMENT:
                    convergence = False
                    break

                convergence = True

            environment.close()

            if convergence == True:
                print("Convergence reached, environment solved by network!")
                print("Episodes until convergence: {}".format(episode))
                episodes_till_convergence = episode + 1
                break

        if convergence == True:
            break

    environment.close()
    filename = "DQNplot"
    plots.plot_learning_curve_BG_full_supervision_paper(scores, episodes_till_convergence, filename, "Learning curve DQN")
    plots.plot_learning_curve(episodes_till_convergence, scores, epsilon_history, 'Learning_curve_BG.png')
    print("Training of DQN(Basal Ganglia) network done!!")
    print("Episodes until convergence: {}".format(episodes_till_convergence))

def test_performance_of_network(environment, network, episode_amount=1, render=False):

    rewards_current_episode = 0

    for episode in range(episode_amount):
        state = torch.tensor(environment.reset())
        rewards_current_episode = 0
        done = False

        if render == True:
            print("Current show-off episode: %d" % episode)

        while not done:

            # Let the agent select an action, do this action and observe the reward and new state
            action = network.forward(state).softmax(dim=0).argmax().item()
            new_state, reward, done, info = environment.step(action)

            if render == True:
                environment.render()

            rewards_current_episode += reward
            state = torch.tensor(new_state)

            if done:
                break

        if render == True:
            print("Reward for episode: {}".format(rewards_current_episode))

    return rewards_current_episode

def train_Hebb_with_annealing_BG_input(environment, BG_network, Hebb_network, Hebb_network_layer):

    scores = []
    weight_avg_history = []

    for episode in range(c.AMOUNT_OF_EPISODES_FOR_TRAINING):

        state = torch.tensor(environment.reset())
        done = False
        if episode >= 1:
            print("Current episode: {}, score: {}".format(episode, scores[episode - 1]))
        score = 0

        while not done:

            with torch.no_grad(): #Prevent pytorch from calculating gradients, we calculate our own weights here.

                output_BG = BG_network.forward(state)
                output_HEBB = Hebb_network.forward(state)

                combined_ouput = hebb.combine_outputs_bg_hebb_weighted_avg(output_BG,output_HEBB,episode)

                action = output_BG.argmax().item()
                new_state, reward, done, info = environment.step(action)
                #environment.render()

                #Update the weights of the Hebbian layer

                output_BG *=  0.01
                #print("output bg: {}".format(output_BG))
                #state *= 0.1


                new_weights_hebb_layer = hebb.hebbian_weightcalc_oja_direct_mapping(output_BG,
                                                                                    Hebb_network_layer.weight,
                                                                                    state)
                #print("Output BG: {}".format(output_BG))
                #print("Output Hebb: {}".format(output_HEBB))
                #print("combined_ouput: {}".format(combined_ouput))
                #print("Input state: {}".format(state))
                #print("New_weights: {}".format(new_weights_hebb_layer))
                '''
                declined_flag = False
                for i in range(c.ACTION_SPACE):
                    for k in range(c.OBSERVATION_SPACE):
                        if new_weights_hebb_layer[i][k] > 1000:
                            new_weights_hebb_layer = new_weights_hebb_layer.mul(0.001)
                            #print('declined the weights')
                            declined_flag = True
                            break
                    if declined_flag == True:
                        declined_flag = False
                        break
                '''
                hebb.install_new_weights_in_hebb_layer(Hebb_network_layer, new_weights_hebb_layer)
                #print("new_weights_hebb_layer: {}".format(new_weights_hebb_layer))
                score += reward  # Keep track of the recieved reward during the episode to plot later
                state = torch.tensor(new_state)

        if test_for_convergence_needed(episode):
            test_performance_of_network(environment, Hebb_network, render=True)

        score = test_performance_of_network(environment, Hebb_network)
        if score >= 500.0:
            torch.save(Hebb_network.state_dict(), unique_file("Hebbian_net_500", "pt"))

        #print("weights of hebb layer: {}".format(Hebb_network_layer.weight))
        scores.append(score)
        weight_avg_history.append(hebb.calculate_weighted_avg_num_linear_decay(episode))

    environment.close()
    plots.plot_learning_curve_hebb(c.AMOUNT_OF_EPISODES_FOR_TRAINING, scores, weight_avg_history, 'Learning_curve_HEBB.png')
    print("Training of hebbian network done!!")

def train_Hebb_with_BG_full_supervision(environment, BG_network, Hebb_network, Hebb_network_layer):

    print("training with full supervision")
    scores = []
    weight_avg_history = []
    convergence = False
    episodes_till_convergence = 0

    for episode in range(c.AMOUNT_OF_EPISODES_FOR_TRAINING):

        state = torch.tensor(environment.reset())
        done = False
        if episode >= 1:
            print("Current episode: {}, score: {}".format(episode, scores[episode - 1]))
        score = 0

        while not done:

            with torch.no_grad(): #Prevent pytorch from calculating gradients, we calculate our own weights here.
                output_BG = BG_network.forward(state).softmax(dim=0)
                output_HEBB = Hebb_network.forward(state).softmax(dim=0)

                action = output_BG.argmax().item()
                new_state, reward, done, info = environment.step(action)
                #environment.render()

                #Update the weights of the Hebbian layer
                new_weights_hebb_layer = hebb.hebbian_weightcalc_direct_mapping_weighted_lr(output_BG,
                                                                                    Hebb_network_layer.weight,
                                                                                    state,output_HEBB)
                #print("Output BG: {}".format(output_BG))
                #print("Output Hebb: {}".format(output_HEBB))
                #print("combined_ouput: {}".format(combined_ouput))
                #print("Input state: {}".format(state))
                #print("New_weights: {}".format(new_weights_hebb_layer))

                hebb.install_new_weights_in_hebb_layer(Hebb_network_layer, new_weights_hebb_layer)

                score += reward  # Keep track of the recieved reward during the episode to plot later
                state = torch.tensor(new_state)
        #print("score achieved by BG: {}".format(score))

        score = test_performance_of_network(environment, Hebb_network)
        if test_for_convergence_needed(episode) or score >= 500.0:
            rewards = []
            for i in range(50):
                reward = test_performance_of_network(environment, Hebb_network, 1, False)
                rewards.append(reward)

                if reward < 500.0:
                    convergence = False
                    print("Rewards during test: {}".format(rewards))
                    print("Convergence not yet reached")
                    #print(Hebb_network_layer.weight)
                    break

                convergence = True

            environment.close()

            if convergence == True:
                print("Convergence reached, environment solved by network!")
                print("Episodes until convergence: {}".format(episode))
                episodes_till_convergence = episode
                break

        if convergence == True:
            break

        #score = test_performance_of_network(environment, Hebb_network)
        #if score >= 500.0:
         #   torch.save(Hebb_network.state_dict(), unique_file("Hebbian_net_500_softmax_relu", "pt"))
        #print("weights of hebb layer: {}".format(Hebb_network_layer.weight))
        scores.append(score)
        weight_avg_history.append(100)

    environment.close()
    if convergence == True:
        plots.plot_learning_curve_hebb(episodes_till_convergence, scores, weight_avg_history, 'Learning_curve_HEBB_no_relu_lr0001_weighted')
        print("Training of hebbian network done!!")
        return
    plots.plot_learning_curve_hebb(c.AMOUNT_OF_EPISODES_FOR_TRAINING, scores, weight_avg_history, 'Learning_curve_HEBB_no_relu_lr0001_weighted')
    print("Training of hebbian network done!!")

def train_Hebb_with_BG_keep_best(environment, BG_network, Hebb_network, Hebb_network_layer):

    scores = []
    weight_avg_history = []

    for episode in range(c.AMOUNT_OF_EPISODES_FOR_TRAINING):

        state = torch.tensor(environment.reset())
        done = False
        if episode >= 1:
            print("Current episode: {}, score: {}".format(episode, scores[episode - 1]))
        score = 0

        weights_prev_episode = Hebb_network_layer.weight.detach().clone().to(c.DEVICE)

        while not done:

            with torch.no_grad(): #Prevent pytorch from calculating gradients, we calculate our own weights here.

                output_BG = BG_network.forward(state)
                output_HEBB = Hebb_network.forward(state)

                combined_ouput = hebb.combine_outputs_bg_hebb_weighted_avg(output_BG,output_HEBB,episode)

                action = output_BG.argmax().item()
                new_state, reward, done, info = environment.step(action)
                #environment.render()

                #Update the weights of the Hebbian layer

                #output_BG = F.normalize(output_BG, dim=0) * 10
                #state = F.normalize(state, dim=0)

                new_weights_hebb_layer = hebb.hebbian_weightcalc_direct_mapping(output_BG,
                                                                                    Hebb_network_layer.weight,
                                                                                    state)
                #print("Output BG: {}".format(output_BG))
                #print("Output Hebb: {}".format(output_HEBB))
                #print("combined_ouput: {}".format(combined_ouput))
                #print("Input state: {}".format(state))
                #print("New_weights: {}".format(new_weights_hebb_layer))

                #new_weights_hebb_layer = F.normalize(new_weights_hebb_layer) * 100
                weights_prev_step = Hebb_network_layer.weight.detach().clone().to(c.DEVICE)
                hebb.install_new_weights_in_hebb_layer(Hebb_network_layer, new_weights_hebb_layer)
                state = torch.tensor(new_state)

                weights_to_save = get_best_weights(weights_prev_step, new_weights_hebb_layer, Hebb_network,
                                                   Hebb_network_layer, 10)
                hebb.install_new_weights_in_hebb_layer(Hebb_network_layer, weights_to_save)

        #Test if performance is better after training episode
        #print("Weights previous episode: {}".format(weights_prev_episode))
        #print("Weights after training episode: {}".format(weights_to_save))
        best_weights = get_best_weights(weights_prev_episode, weights_to_save, Hebb_network,
                                           Hebb_network_layer, 10)
        hebb.install_new_weights_in_hebb_layer(Hebb_network_layer, best_weights)

        #avg_reward = 0
        #for i in range(10):
        #    avg_reward += test_performance_of_network(environment, Hebb_network, render=False)
        #print("average reward after episode training: {}".format(avg_reward / 10))

        if test_for_convergence_needed(episode):
            test_performance_of_network(environment, Hebb_network, render=True)
            print("Current weights: {}".format(Hebb_network_layer.weight))


        score = test_performance_of_network(environment, Hebb_network)
        if score >= 500.0:
            torch.save(Hebb_network.state_dict(), unique_file("Hebbian_net_500", "pt"))

        #print("weights of hebb layer: {}".format(Hebb_network_layer.weight))
        scores.append(score)
        weight_avg_history.append(hebb.calculate_weighted_avg_num_linear_decay(episode))

    environment.close()
    plots.plot_learning_curve_hebb(c.AMOUNT_OF_EPISODES_FOR_TRAINING, scores, weight_avg_history, 'Learning_curve_HEBB.png')
    print("Training of hebbian network done!!")

def train_multilayer_Hebb_with_annealing_BG_input(environment, BG_network, Hebb_network):

    scores = []
    weight_avg_history = []

    Hebb_network.layer_1.register_forward_hook(get_layer_output_layer_1)

    for episode in range(c.AMOUNT_OF_EPISODES_FOR_TRAINING):

        state = torch.tensor(environment.reset())
        done = False
        if episode >= 1:
            print("Current episode: {}, score: {}".format(episode, scores[episode - 1]))
        score = 0

        while not done:

            with torch.no_grad(): #Prevent pytorch from calculating gradients, we calculate our own weights here.

                output_BG = BG_network.forward(state)
                output_HEBB = Hebb_network.forward(state)

                combined_ouput = hebb.combine_outputs_bg_hebb_weighted_avg(output_BG,output_HEBB,episode)

                action = output_BG.argmax().item()
                new_state, reward, done, info = environment.step(action)
                #environment.render()

                #Update the weights of the Hebbian layer

                #print("output bg: {}".format(output_BG))
                #state *= 0.1

                new_weights_hebb_layer_1 = hebb.hebbian_weightcalc_direct_mapping_multilayer(activations_layer_1[0],
                                                                                  Hebb_network.layer_1.weight,
                                                                                  state)
                new_weights_hebb_layer_2 = hebb.hebbian_weightcalc_direct_mapping_multilayer(output_BG,
                                                                                  Hebb_network.layer_2.weight,
                                                                                  activations_layer_1[0])

                hebb.install_new_weights_in_hebb_layer(Hebb_network.layer_1, new_weights_hebb_layer_1)
                hebb.install_new_weights_in_hebb_layer(Hebb_network.layer_2, new_weights_hebb_layer_2)


                #print("Output BG: {}".format(output_BG))
                #print("Output Hebb: {}".format(output_HEBB))
                #print("combined_ouput: {}".format(combined_ouput))
                #print("Input state: {}".format(state))
                #print("New_weights: {}".format(new_weights_hebb_layer))

                #print("new_weights_hebb_layer: {}".format(new_weights_hebb_layer))
                score += reward  # Keep track of the recieved reward during the episode to plot later
                state = torch.tensor(new_state)
        #print('Current Weights: ')
        #print(Hebb_network.layer_1.weight)
        #print(Hebb_network.layer_2.weight)

        if test_for_convergence_needed(episode):
            test_performance_of_network(environment, Hebb_network, render=True)

        score = test_performance_of_network(environment, Hebb_network)
        if score >= 500.0:
            torch.save(Hebb_network.state_dict(), unique_file("Hebbian_net_500", "pt"))

        #print("weights of hebb layer: {}".format(Hebb_network_layer.weight))
        scores.append(score)
        weight_avg_history.append(hebb.calculate_weighted_avg_num_linear_decay(episode))

    environment.close()
    plots.plot_learning_curve_hebb(c.AMOUNT_OF_EPISODES_FOR_TRAINING, scores, weight_avg_history, 'Learning_curve_HEBB.png')
    print("Training of hebbian network done!!")

def train_Hebb_with_BG_keep_best_multilayer(environment, BG_network, Hebb_network):

    scores = []
    weight_avg_history = []
    Hebb_network.layer_1.register_forward_hook(get_layer_output_layer_1)


    for episode in range(c.AMOUNT_OF_EPISODES_FOR_TRAINING):

        state = torch.tensor(environment.reset())
        done = False
        if episode >= 1:
            print("Current episode: {}, score: {}".format(episode, scores[episode - 1]))
            print("Current weights layer 1: {}".format(Hebb_network.layer_1.weight))
            print("Current weights layer 2: {}".format(Hebb_network.layer_2.weight))
        score = 0

        previous_episode_weight_layer_1 = Hebb_network.layer_1.weight.detach().clone().to(c.DEVICE)
        previous_episode_weight_layer_2 = Hebb_network.layer_2.weight.detach().clone().to(c.DEVICE)

        first_step = True

        while not done:

            with torch.no_grad(): #Prevent pytorch from calculating gradients, we calculate our own weights here.

                output_BG = BG_network.forward(state)
                output_HEBB = Hebb_network.forward(state)

                combined_ouput = hebb.combine_outputs_bg_hebb_weighted_avg(output_BG,output_HEBB,episode)

                action = output_BG.argmax().item()
                new_state, reward, done, info = environment.step(action)
                #environment.render()

                #Update the weights of the Hebbian layer

                #output_BG = F.normalize(output_BG, dim=0) * 10
                #state = F.normalize(state, dim=0)


                previous_weight_layer_1 = Hebb_network.layer_1.weight.detach().clone().to(c.DEVICE)
                previous_weight_layer_2 = Hebb_network.layer_2.weight.detach().clone().to(c.DEVICE)

                new_weights_hebb_layer_1 = hebb.hebbian_weightcalc_oja_direct_mapping(activations_layer_1[0],
                                                                                             Hebb_network.layer_1.weight,
                                                                                             state)
                new_weights_hebb_layer_2 = hebb.hebbian_weightcalc_oja_direct_mapping(output_BG,
                                                                                             Hebb_network.layer_2.weight,
                                                                                             activations_layer_1[0])

                #print("Output BG: {}".format(output_BG))
                #print("Output Hebb: {}".format(output_HEBB))
                #print("combined_ouput: {}".format(combined_ouput))
                #print("Input state: {}".format(state))
                #print("New_weights: {}".format(new_weights_hebb_layer))


                #new_weights_hebb_layer = F.normalize(new_weights_hebb_layer) * 100

                if first_step == True:

                    weights_to_save_1, weights_to_save_2 = get_best_weights_multilayer(previous_episode_weight_layer_1,
                                                                                       previous_episode_weight_layer_2,
                                                                                       new_weights_hebb_layer_1,
                                                                                       new_weights_hebb_layer_2,
                                                                                       Hebb_network, 10)
                    first_step = False

                elif first_step == False:
                    weights_to_save_1, weights_to_save_2 = get_best_weights_multilayer(previous_weight_layer_1,
                                                                                       previous_weight_layer_2,
                                                                                       new_weights_hebb_layer_1,
                                                                                       new_weights_hebb_layer_2,
                                                                                       Hebb_network, 10)

                hebb.install_new_weights_in_hebb_layer(Hebb_network.layer_1, weights_to_save_1)
                hebb.install_new_weights_in_hebb_layer(Hebb_network.layer_2, weights_to_save_2)

                state = torch.tensor(new_state)

        #Test if performance is better after training episode
        #print("Weights previous episode: {}".format(weights_prev_episode))
        #print("Weights after training episode: {}".format(weights_to_save))

        #avg_reward = 0
        #for i in range(10):
        #    avg_reward += test_performance_of_network(environment, Hebb_network, render=False)
        #print("average reward after episode training: {}".format(avg_reward / 10))
        weights_to_save_1, weights_to_save_2 = get_best_weights_multilayer(previous_episode_weight_layer_1,
                                                                           previous_episode_weight_layer_2,
                                                                           weights_to_save_1,
                                                                           weights_to_save_2,
                                                                           Hebb_network, 10)

        hebb.install_new_weights_in_hebb_layer(Hebb_network.layer_1, weights_to_save_1)
        hebb.install_new_weights_in_hebb_layer(Hebb_network.layer_2, weights_to_save_2)

        if test_for_convergence_needed(episode):
            test_performance_of_network(environment, Hebb_network, render=True)

        score = test_performance_of_network(environment, Hebb_network)

        if score >= 500.0:
            torch.save(Hebb_network.state_dict(), unique_file("Hebbian_net_500", "pt"))

        #print("weights of hebb layer: {}".format(Hebb_network_layer.weight))
        scores.append(score)
        weight_avg_history.append(hebb.calculate_weighted_avg_num_linear_decay(episode))

    environment.close()
    plots.plot_learning_curve_hebb(c.AMOUNT_OF_EPISODES_FOR_TRAINING, scores, weight_avg_history, 'Lr_0000001_HEBB_multilayer_Oja_keep_best.png')
    print("Training of hebbian network done!!")

def train_Hebb_with_BG_full_supervision_paper(environment, BG_network, Hebb_network, Hebb_network_layer):

    print("training with full supervision of BG")
    scores = []
    weight_avg_history = []
    convergence = False
    convergence_episode_noted = False
    episodes_till_convergence = 0
    sigmoid = nn.Sigmoid()

    for episode in range(c.AMOUNT_OF_EPISODES_FOR_TRAINING):

        state = torch.tensor(environment.reset())
        done = False
        if episode >= 1:
            print("Current episode: {}, score: {}".format(episode, scores[episode - 1]))

        while not done:

            with torch.no_grad(): #Prevent pytorch from calculating gradients, we calculate our own weights here.
                output_BG = BG_network.forward(state).softmax(dim=0)
                output_HEBB = Hebb_network.forward(state).softmax(dim=0)

                action = output_BG.argmax().item()
                new_state, reward, done, info = environment.step(action)
                #environment.render()

                #Update the weights of the Hebbian layer
                new_weights_hebb_layer = hebb.hebbian_weightcalc_direct_mapping_inverse_sigmoid_weighted_lr(output_BG,
                                                                                                            Hebb_network_layer.weight,
                                                                                                            state,output_HEBB)
                hebb.install_new_weights_in_hebb_layer(Hebb_network_layer, new_weights_hebb_layer)
                state = torch.tensor(new_state)

        score = test_performance_of_network(environment, Hebb_network)

        if (test_for_convergence_needed(episode) or score >= 500.0) and convergence == False:
            rewards = []
            for i in range(50):
                reward = test_performance_of_network(environment, Hebb_network, 1, False)
                rewards.append(reward)

                if reward < 500.0:
                    convergence = False
                    break

                convergence = True

            environment.close()

            if convergence == True and convergence_episode_noted == False:
                print("Convergence reached, environment solved by network!")
                print("Episodes until convergence: {}".format(episode))
                episodes_till_convergence = episode
                convergence_episode_noted = True

        scores.append(score)
        weight_avg_history.append(100)

    environment.close()

    #plots.plot_learning_curve_hebb(c.AMOUNT_OF_EPISODES_FOR_TRAINING, scores, weight_avg_history, 'Learning_curve_HEBB')
    print("Training of hebbian network done!!")
    return convergence, episodes_till_convergence, scores

def train_Hebb_with_annealing_BG_paper(environment, BG_network, Hebb_network, Hebb_network_layer, anneal_method = "linear"):

    print("Training Hebb network  - Action = p*h + q*t ")
    print("Annealing method: {}".format(anneal_method))
    scores = []
    weight_avg_BG_history = []
    weight_avg_Hebb_history = []
    episode_scores = []

    convergence = False
    convergence_episode_noted = False
    episodes_till_convergence = 0

    for episode in range(c.AMOUNT_OF_EPISODES_FOR_TRAINING):

        state = torch.tensor(environment.reset())
        done = False
        if episode >= 1:
            print("Current episode: {}, score: {}".format(episode, scores[episode - 1]))
        episode_score = 0
        while not done:

            with torch.no_grad(): #Prevent pytorch from calculating gradients, we calculate our own weights here.
                output_BG = BG_network.forward(state).softmax(dim=0)
                output_HEBB = Hebb_network.forward(state).softmax(dim=0)
                weighted_action, weighted_average_BG, weighted_average_Hebb = hebb.combine_outputs_bg_hebb_weighted_avg(output_BG,
                                                                                                                        output_HEBB,
                                                                                                                        episode,
                                                                                                                        anneal_method)

                action = weighted_action.argmax().item()
                new_state, reward, done, info = environment.step(action)
                #environment.render()

                #Update the weights of the Hebbian layer
                new_weights_hebb_layer = hebb.hebbian_weightcalc_direct_mapping_inverse_sigmoid_weighted_lr(output_BG,
                                                                                                            Hebb_network_layer.weight,
                                                                                                            state,output_HEBB)
                hebb.install_new_weights_in_hebb_layer(Hebb_network_layer, new_weights_hebb_layer)
                state = torch.tensor(new_state)
                episode_score += reward

        score = test_performance_of_network(environment, Hebb_network, render = False)

        if (test_for_convergence_needed(episode) or score >= 500.0) and convergence == False:
            rewards = []
            for i in range(50):
                reward = test_performance_of_network(environment, Hebb_network, 1, False)
                rewards.append(reward)

                if reward < 500.0:
                    convergence = False
                    break

                convergence = True

            environment.close()

            if convergence == True and convergence_episode_noted == False:
                print("Convergence reached, environment solved by network!")
                print("Episodes until convergence: {}".format(episode))
                episodes_till_convergence = episode
                convergence_episode_noted = True

        episode_scores.append(episode_score)
        scores.append(score)
        weight_avg_BG_history.append(weighted_average_BG*100)
        weight_avg_Hebb_history.append(weighted_average_Hebb*100)

    environment.close()

    #plots.plot_learning_curve_hebb(c.AMOUNT_OF_EPISODES_FOR_TRAINING, scores, weight_avg_BG_history, 'Learning_curve_HEBB')
    print("Training of hebbian network done!!")
    print("weight hebb in training function: {}".format(weight_avg_Hebb_history))
    return convergence, episodes_till_convergence, scores, weight_avg_BG_history, weight_avg_Hebb_history, episode_scores