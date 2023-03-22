import constants as c
import torch
import torch.nn as nn
import math
import numpy as np
from scipy.special import logit
#papers on hebbian learning:
#
# 1. https://doi.org/10.48550/arXiv.2007.02686 - > https://github.com/enajx/HebbianMetaLearning/blob/master/evolution_strategy_hebb.py
#
#
#

#Questions: Does backpropagation use a


#A function for retrieving the activation of the output neurons

#A function for retrieving the activation of the input neurons

#Calculate the new weights

# -> What variables are needed to update the weights?
# -> Which variables are in Oja's rule?
# -> How to retrieve the correct outputs of a full network?
# -> How to link the output of the BG_network to the hebbian network?

# -> how to reshape the tensors such that the network can learn and how to plug them back in?
# -> Retrieve the shapes and play with it in Google Colab and see how to feed them back in

# TO-Do:
#3 days
# 1. Make a neural network for the Hebbian
# 3. Learn all about hebbian and understand how to code it.
# 2. Code up Oja's rule -> Toy with it and make sure it calculates correct value
# 3. Code up Oja's rule such that it can use the shapes of the network

# 3 - days
# 4. Retrieve the output of the neurons in the Hebbian network
# 5. Retrieve the output of the neurons in the Hebbian network in which the BG network  is involved

# Long days
# 6. Link Oja's rule to the output including the BG network
# 7. If all working then see how to down-tune the involvement of the BG_network
# 8. Plot all the curves
# 9. Look at the performance and try to understand why it will or won't work.
# Why could this be? : 1. Learning rule of Oja not sufficiënt



#A function to calculate hebbian weights according to Oja's rule.
#Note! This is only for direct mapping! So no deep network

def caluclate_weighted_learning_rate(target_output, hebb_output):
    learning_rate = c.HEBB_LEARNING_RATE * ((target_output - hebb_output)**2)

    return learning_rate

def hebbian_weightcalc_oja_direct_mapping(outputs, weights, inputs):

    new_weights = torch.zeros_like(weights)

    #The tensors need to be detached and cloned, because the tensors are like 'pointers'
    output_tens = outputs.detach().clone().to(c.DEVICE)
    input_tens = inputs.detach().clone().to(c.DEVICE)

    #print("previous weights: {}".format(weights))

    for i in range(len(outputs)):
        for k in range(len(inputs)):

            #print("---------------------------------------")
            #print("step by step:")
            #print("previous weight: {}".format(weights[i][k]))
            #print("output number: {}".format(output_tens[i]))
            #print("input number: {}".format(input_tens[k]))
            #print("Learning rate: {}".format(c.HEBB_LEARNING_RATE))

            new_weights[i][k] = weights[i][k] + ((c.HEBB_LEARNING_RATE * output_tens[i]) * (input_tens[k] - (weights[i][k] * output_tens[i])))

            #print("new weights: {}".format(new_weights[i][k]))
    #print("-----------------------------------------------")
    #print("learning rate: {}".format(c.HEBB_LEARNING_RATE))
    #print("outputs: {}".format(outputs))
    #print("inputs: {}".format(inputs))
    #print("new weights: {}".format(new_weights))



    return new_weights

def hebbian_weightcalc_direct_mapping(outputs, weights, inputs):

    new_weights = torch.zeros_like(weights)

    #The tensors need to be detached and cloned, because the tensors are like 'pointers'
    output_tens = outputs.detach().clone().to(c.DEVICE)
    input_tens = inputs.detach().clone().to(c.DEVICE)

    for i in range(len(outputs)):
        for k in range(len(inputs)):
            new_weights[i][k] = weights[i][k] + (c.HEBB_LEARNING_RATE*(input_tens[k] * output_tens[i]))

    return new_weights

def hebbian_weightcalc_direct_mapping_inverse_sigmoid_weighted_lr(target_output, weights, inputs, hebb_output):

    new_weights = torch.zeros_like(weights)

    #The tensors need to be detached and cloned, because the tensors are like 'pointers'
    output_tens = target_output.detach().clone().to(c.DEVICE)
    input_tens = inputs.detach().clone().to(c.DEVICE)
    hebb_tens = hebb_output.detach().clone().to(c.DEVICE)


    for i in range(len(output_tens)):

        learning_rate = caluclate_weighted_learning_rate(output_tens[i], hebb_tens[i])

        #hardcode the inverse sigmoid function
        output_value = output_tens[i]
        if output_tens[i] == 1:
            output_value = 0.99999999999
        if output_tens[i] == 0:
            output_value = 0.00000000001

        for k in range(len(inputs)):
            new_weights[i][k] = weights[i][k] + (learning_rate*(input_tens[k] * logit(output_value)))
    #print(new_weights)
    return new_weights

def hebbian_weightcalc_direct_mapping_inverse_sigmoid(target_output, weights, inputs, hebb_output):

    new_weights = torch.zeros_like(weights)

    #The tensors need to be detached and cloned, because the tensors are like 'pointers'
    output_tens = target_output.detach().clone().to(c.DEVICE)
    input_tens = inputs.detach().clone().to(c.DEVICE)
    hebb_tens = hebb_output.detach().clone().to(c.DEVICE)


    for i in range(len(output_tens)):
        #hardcode the inverse sigmoid function
        output_value = output_tens[i]
        if output_tens[i] == 1:
            output_value = 0.99999999999
        if output_tens[i] == 0:
            output_value = 0.00000000001

        for k in range(len(inputs)):
            new_weights[i][k] = weights[i][k] + (c.HEBB_LEARNING_RATE*(input_tens[k] * logit(output_value)))
    #print(new_weights)
    return new_weights

def hebbian_weightcalc_direct_mapping_weighted_lr(target_output, weights, inputs, hebb_output):

    new_weights = torch.zeros_like(weights)

    #The tensors need to be detached and cloned, because the tensors are like 'pointers'
    output_tens = target_output.detach().clone().to(c.DEVICE)
    input_tens = inputs.detach().clone().to(c.DEVICE)
    hebb_tens = hebb_output.detach().clone().to(c.DEVICE)


    for i in range(len(output_tens)):
        learning_rate = caluclate_weighted_learning_rate(output_tens[i], hebb_tens[i])
        for k in range(len(inputs)):
            new_weights[i][k] = weights[i][k] + (learning_rate*(input_tens[k] * output_tens[i]))
    return new_weights




def hebbian_weightcalc_direct_mapping_multilayer(outputs, weights, inputs):

    new_weights = torch.zeros_like(weights)

    #The tensors need to be detached and cloned, because the tensors are like 'pointers'
    output_tens = outputs
    input_tens = inputs
    #print(weights)

    for i in range(len(output_tens)):
        for k in range(len(input_tens)):
            new_weights[i][k] = weights[i][k] + (c.HEBB_LEARNING_RATE*(input_tens[k] * output_tens[i]))

    return new_weights

def calculate_weighted_avg_num_reward_based(reward, reward_threshold, times_threshold_reached):

    if reward >= reward_threshold:
        times_threshold_reached += 1

    weighted_avg_num = c.HEBB_END + (c.HEBB_START - (c.HEBB_DECAY * times_threshold_reached))

    return weighted_avg_num

def calculate_weighted_avg_num_exponential_decay(episode):

    weighted_avg_num = c.EPSILON_END_EXP + (c.EPSILON_START - c.EPSILON_END_EXP) * math.exp(-1. * episode * c.EPSILON_DECAY_EXP)

    return weighted_avg_num

def calculate_weighted_avg_num_linear_decay(episode):

    #print("Input episode: {}".format(episode))
    weighted_avg_num = c.HEBB_START - (episode * c.HEBB_DECAY)

    if weighted_avg_num < c.HEBB_END:
        weighted_avg_num = c.HEBB_END

    #print("weighted avg num: {}".format(weighted_avg_num))
    return weighted_avg_num

def combine_outputs_bg_hebb_weighted_avg(output_BG, output_hebb, episode, method = "linear"):

    output_hebb_detached = output_hebb.detach().clone().to(c.DEVICE)
    output_BG_detached = output_BG.detach().clone().to(c.DEVICE)

    if method == "linear":
        weighted_average_BG = calculate_weighted_avg_num_linear_decay(episode)
    elif method == "exp":
        weighted_average_BG = calculate_weighted_avg_num_exponential_decay(episode)

    weighted_average_Hebb = 1 - weighted_average_BG
    weighted_average_output = (weighted_average_BG * output_BG_detached) + (weighted_average_Hebb * output_hebb_detached)

    return weighted_average_output, weighted_average_BG, weighted_average_Hebb

def install_new_weights_in_hebb_layer(layer_name, weights):

    with torch.no_grad():
        layer_name.weight = nn.Parameter(weights, requires_grad = False)

    return