#import libraries
import copy
import os.path
import gym
import csv
import constants as c
import replayMemory
import agents
import neuralNetworks
import torch
import trainer
import plots
import itertools
import copy
import statistics as stats
# Does a new BG-model need to be trained and generated?
train_new_BG = True #-> Should be entered by user
train_new_Hebb = True
show_off_BG = False
BG_model_exists = os.path.isfile('BG_network.pt')
Hebb_model_exists = os.path.isfile('Hebbian_net.pt')
if BG_model_exists:
    print("BG model exists!")
if Hebb_model_exists:
    print("Hebb model exists!")
# Create environment based on environment given by user in constants file
environment = gym.make(c.ENVIRONMENT)

print("Training environment: {}, Reward threshold: {}".format(c.ENVIRONMENT, c.MAXIMUM_REWARD_ENVIRONMENT))
#-----------------------BASAL GANGLIA NETWORK------------------------------------#
# Create neural network, agent and replay memory in order to play the environment
policy_network = neuralNetworks.basic_DQN_Network().to(c.DEVICE)
target_network = copy.deepcopy(policy_network).to(c.DEVICE) # The target network is the same as the policy network
agent          = agents.DQN_policyNet_targetNet_Agent(policy_network)
replay_memory  = replayMemory.replayMemoryDQN()

# Because the target network will only be used to update the q-values of the policy network to, we set it on eval mode.
# Eval mode
target_network.eval()

#Train aa DQN representing the Basal Ganglia if it doesn't exist yet.
if train_new_BG == True or BG_model_exists == False:
    trainer.train_basic_DQN_policyNet_targetNet(agent, environment, replay_memory, policy_network, target_network)
    torch.save(policy_network.state_dict(),'BG_network_softmax.pt')
    print("saved the new neural network as BG_network")

#If a BG_network file does exist already, then load it and show off to the user.

BG_network = neuralNetworks.basic_DQN_Network().to(c.DEVICE)# Make a dummy network in which the file can be copied in
BG_network.load_state_dict(torch.load('BG_network.pt'))
BG_network.eval()

if show_off_BG == True:
    trainer.test_performance_of_network(environment, BG_network, episode_amount=10, render=True)

#----------------------------
def unique_file(basename, ext):
    actualname = "%s.%s" % (basename, ext)
    c = itertools.count()
    while os.path.exists(actualname):
        actualname = "%s (%d).%s" % (basename, next(c), ext)
    return actualname

#-------------------------HEBBIAN NETWORK------------------------------#

#Train the hebbian network based on the BG_network if it does not yet exists

amount_of_runs = c.AMOUNT_OF_SAMPLES
print("obs_space: {}".format(c.OBSERVATION_SPACE))
print("action space: {}".format(c.ACTION_SPACE))
if train_new_Hebb == True or Hebb_model_exists == False:

    for i in range(1):

        scores = []
        episodes_till_convergence_exp = []
        performance_list = []
        for i in range(amount_of_runs):

            # Train with exponential decay.

            print("Training a Hebbian network with exponential decay")
            hebb_training_network = neuralNetworks.hebbian_network_1_layer().to(c.DEVICE)
            convergence, converge_episode, hebb_performance, bg_decay, hebb_decay, train_performance = trainer.train_Hebb_with_annealing_BG_paper(environment,
                                                                                                     BG_network,
                                                                                                     hebb_training_network,
                                                                                                     hebb_training_network.hebb_layer,
                                                                                                     anneal_method = "exp")
            if convergence == False:
                print("The Hebbian network did not converge after training, is something wrong?")

            # ['name', 'subsample', 'convergence_episodes', 'training_performance','performance_hebb', 'hebb_decay', 'BG_decay']
            #data_to_save = ['exp', '1', convergence_episode, ]
            #stats.save_data_to_file()


            scores.append(hebb_performance)
            episodes_till_convergence_exp.append(converge_episode)
            performance_list.append(train_performance)
            #torch.save(hebb_training_network.state_dict(), unique_file("Hebbian_net_sigmoid","pt"))
            print("saved the new neural network as HEBB_network.pt")
        title = "Learning curve "+"Exponential " + r'$\varphi$' + "-decay"
        hebb_mean_exp, train_mean_exp = plots.plot_learning_curve_performance_and_annealing_BG_paper(scores, episodes_till_convergence_exp, hebb_decay, bg_decay, performance_list, filename= "Learning curve - annealing - exp - perf", title=title)
        hebb_scores_exp = copy.deepcopy(scores)
        performance_scores_exp = copy.deepcopy(performance_list)


        scores = []
        episodes_till_convergence_linear = []
        performance_list = []
        for i in range(amount_of_runs):
            print("Training a new Hebbian network with linear decay")
            hebb_training_network = neuralNetworks.hebbian_network_1_layer().to(c.DEVICE)
            convergence, converge_episode, score, weight_bg, weight_hebb, performance = trainer.train_Hebb_with_annealing_BG_paper(environment,
                                                                                                     BG_network,
                                                                                                     hebb_training_network,
                                                                                                     hebb_training_network.hebb_layer,
                                                                                                     anneal_method = "linear")
            if convergence == False:
                print("The Hebbian network did not converge after training, is something wrong?")

            scores.append(score)
            episodes_till_convergence_linear.append(converge_episode)
            performance_list.append(performance)

            #torch.save(hebb_training_network.state_dict(), unique_file("Hebbian_net_sigmoid","pt"))
            print("saved the new neural network as HEBB_network.pt")

        title = "Learning curve " + "Linear " + r'$\varphi$' + "-decay"
        hebb_mean_lin, train_mean_lin = plots.plot_learning_curve_performance_and_annealing_BG_paper(scores, episodes_till_convergence_linear, weight_hebb, weight_bg, performance_list, filename= "Learning curve - annealing - linear - perf", title=title)
        hebb_scores_lin = copy.deepcopy(scores)
        performance_scores_lin = copy.deepcopy(performance_list)


        scores = []
        episodes_till_convergence_supervised = []
        for i in range(amount_of_runs):
            print("Training a new Hebbian network with full supervision")
            hebb_training_network = neuralNetworks.hebbian_network_1_layer().to(c.DEVICE)
            convergence, converge_episode, score = trainer.train_Hebb_with_BG_full_supervision_paper(environment,
                                                                                                     BG_network,
                                                                                                     hebb_training_network,
                                                                                                     hebb_training_network.hebb_layer)
            if convergence == False:
                print("The Hebbian network did not converge after training, is something wrong?")

            scores.append(score)
            episodes_till_convergence_supervised.append(converge_episode)

            #torch.save(hebb_training_network.state_dict(), unique_file("Hebbian_net_sigmoid","pt"))
            print("saved the new neural network as HEBB_network.pt")

        title = "Learning curve Supervised"
        hebb_mean_supervised = plots.plot_learning_curve_BG_full_supervision_paper(scores, episodes_till_convergence_supervised, filename= "Learning curve - supervision", title=title)
        hebb_scores_supervised = copy.deepcopy(scores)
    #perform t-tests
    stats.paired_samples_t_test_convergence(episodes_till_convergence_exp, episodes_till_convergence_linear, distr_1_name="Exponential decay", distr_2_name = "Linear decay")
    stats.paired_samples_t_test_convergence(episodes_till_convergence_exp, episodes_till_convergence_supervised, distr_1_name="Exponential decay", distr_2_name = "Supervised")
    stats.paired_samples_t_test_convergence(episodes_till_convergence_linear, episodes_till_convergence_supervised, distr_1_name="Linear decay", distr_2_name = "Supervised")

    #make histograms
    #plot_histograms_2_distributions_overlap(distribution_1, distribution_2, distr_1_name = "Distribution 1", distr_2_name="Distribution 2"):

    plots.plot_histograms_2_distributions_overlap(episodes_till_convergence_exp, episodes_till_convergence_linear,
                                                  distr_1_name = "Exponential " + r'$\varphi$' + "-decay",
                                                  distr_2_name = "Linear " + r'$\varphi$' + "-decay",
                                                  colors = [c.EXPONENTIAL_HIST_COLOR, c.LINEAR_HIST_COLOR])
    plots.plot_histograms_2_distributions_overlap(episodes_till_convergence_exp, episodes_till_convergence_supervised,
                                                  distr_1_name="Exponential " + r'$\varphi$' + "-decay",
                                                  distr_2_name="Supervised",
                                                  colors=[c.EXPONENTIAL_HIST_COLOR, c.SUPERVISED_HIST_COLOR])
    plots.plot_histograms_2_distributions_overlap(episodes_till_convergence_linear, episodes_till_convergence_supervised,
                                                  distr_1_name="Linear " + r'$\varphi$' + "-decay",
                                                  distr_2_name="Supervised",
                                                  colors=[c.LINEAR_HIST_COLOR, c.SUPERVISED_HIST_COLOR])

    plots.plot_1_distribution(episodes_till_convergence_exp,"Exponential " + r'$\varphi$' + "-decay", c.EXPONENTIAL_HIST_COLOR)
    plots.plot_1_distribution(episodes_till_convergence_linear, "Linear " + r'$\varphi$' + "-decay",
                              c.LINEAR_HIST_COLOR)
    plots.plot_1_distribution(episodes_till_convergence_supervised, "Supervised",
                              c.SUPERVISED_HIST_COLOR)
    #print the used dataset so the user can sive it
    print("-------------------------------------------")
    print("Data used for dataset")
    print("-------------------------------------------")
    print("Sample of mean episodes until convergence exponential decay")
    print(episodes_till_convergence_exp)
    print("-------------------------------------------")
    print("Sample of mean episodes until convergence linear decay")
    print(episodes_till_convergence_linear)
    print("-------------------------------------------")
    print("Sample of mean episodes until convergence supervised")
    print(episodes_till_convergence_supervised)
    print("-------------------------------------------")
    print("Scores of hebbian network in exponential decay")
    print(hebb_scores_exp)
    print("-------------------------------------------")
    print("Scores of training performance in exponential decay")
    print(performance_scores_exp)
    print("Scores of hebbian network in linear decay")
    print(hebb_scores_lin)
    print("-------------------------------------------")
    print("Scores of training performance in linear decay")
    print(performance_scores_exp)
    print("-------------------------------------------")
    print("Scores of hebbian network in supervised")
    print(hebb_scores_supervised)

    #save the data to a csv file
    filename = unique_file("data",".csv")

    # Open the file in write mode and write the data
    with open(filename, mode='w', newline='') as data_file:
        data_writer = csv.writer(data_file)
        data_writer.writerow(["Episodes Till Convergence Exp", "Episodes Till Convergence Linear",
                              "Episodes Till Convergence Supervised", "Hebb Scores Exp", "Performance Scores Exp",
                              "Hebb Scores Linear"])
        for i in range(len(episodes_till_convergence_exp)):
            data_writer.writerow([episodes_till_convergence_exp, episodes_till_convergence_linear,
                                  episodes_till_convergence_supervised[i], hebb_scores_exp,
                                  performance_scores_exp[i], hebb_scores_lin[i]])

    filename = unique_file("data",".txt")
    # Open the file in write mode and save the variables to it
    with open(filename, "w") as f:
        f.write("episodes_till_convergence_exp: " + str(episodes_till_convergence_exp) + "\n")
        f.write("episodes_till_convergence_linear: " + str(episodes_till_convergence_linear) + "\n")
        f.write("episodes_till_convergence_supervised: " + str(episodes_till_convergence_supervised) + "\n")
        f.write("hebb_scores_exp: " + str(hebb_scores_exp) + "\n")
        f.write("performance_scores_exp: " + str(performance_scores_exp) + "\n")
        f.write("hebb_scores_lin: " + str(hebb_scores_lin) + "\n")
        f.write("performance_scores_lin: " + str(performance_scores_lin) + "\n")
        f.write("hebb_scores_supervised: " + str(hebb_scores_supervised) + "\n")

'''
#If a BG_network file does exist already, then load it and show off to the user.
hebb_network = neuralNetworks.hebbian_network_1_layer().to(c.DEVICE)# Make a dummy network in which the file can be copied in
hebb_network.load_state_dict(torch.load('Hebbian_net_sigmoid (0).pt'))
print(hebb_network.hebb_layer.weight)
hebb_network.eval()
trainer.test_performance_of_network(environment, hebb_network, episode_amount = 10, render=True)

#train a BG model
#save the BG model


#if a BG model is already trained, then use that model to train the hebbian model

#train the hebbian model if needed

#plot a curve of the performance of the hebbian mode
'''