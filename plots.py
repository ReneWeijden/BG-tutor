import matplotlib as mpl
import matplotlib.font_manager as font_manager

mpl.rcParams['font.family']='serif'
cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
mpl.rcParams['font.serif']=cmfont.get_name()
mpl.rcParams['mathtext.fontset']='cm'
mpl.rcParams['axes.unicode_minus']=False
import matplotlib.pyplot as plt

import numpy as np
import itertools
import os.path
import constants as c
#Big thanks to machine learning with phil: https://www.youtube.com/watch?v=wc-FxNENg9U&t=728s
def plot_learning_curve(num_episodes, scores, epsilons, filename, lines=None):
    x = [i+1 for i in range(num_episodes)]
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Game", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1", s=2)
    #ax2.xaxis.tick_top()
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    #ax2.set_xlabel('x label 2', color="C1")
    ax2.set_ylabel('Score', color="C1")
    #ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    #ax2.tick_params(axis='x', colors="C1")
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)
    plt.close()

def unique_file(basename, ext):
    actualname = "%s.%s" % (basename, ext)
    c = itertools.count()
    while os.path.exists(actualname):
        actualname = "%s (%d).%s" % (basename, next(c), ext)
    return actualname

def calculate_descriptive_statistics(values):
    mean = np.mean(values, axis = 0)
    std_dev = np.std(values, axis = 0)

    #Standard deviation capped for the maximum reward of the environment. The agent can't achieve higher than max
    uncorrected_upper_std_dev = mean + std_dev
    upper_std_dev = [500.0 if (x > 500.0) else x for x in uncorrected_upper_std_dev]

    #Lower standard deviation
    lower_std_dev = mean - std_dev

    return mean, upper_std_dev, lower_std_dev

def plot_learning_curve_hebb(num_episodes, scores, weighted_avg, filename, lines=None):
    x = [i+1 for i in range(num_episodes)]
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, weighted_avg, color="C0")
    ax.set_xlabel("Game", color="C0")
    ax.set_ylabel("Weighted average BG", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1", s=1)
    #ax2.xaxis.tick_top()
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    #ax2.set_xlabel('x label 2', color="C1")
    ax2.set_ylabel('Score', color="C1")
    #ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    #ax2.tick_params(axis='x', colors="C1")
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(unique_file(filename, "png"))
    plt.savefig(unique_file(filename, "svg"), format="svg", dpi=1200)

def plot_learning_curve_BG_full_supervision_paper(train_scores, episodes_till_convergence, filename, title):

    print("train scores: {}".format(train_scores))

    train_mean, train_upper_std_dev, train_lower_std_dev =  calculate_descriptive_statistics(train_scores)
    # Create means and standard deviations of training set scores
    train_std = np.std(train_scores, axis = 0)
    train_sizes = list(range(len(train_mean)))
    convergence_mean = round(np.mean(episodes_till_convergence, dtype=np.float64).item(),1)

    #create new figure
    plt.figure()

    # Draw lines
    plt.plot(train_sizes, train_mean, color="#111111", label="Performance Hebbian Network")

    # Draw standard deviation bands
    plt.fill_between(train_sizes, train_lower_std_dev, train_upper_std_dev, color="#DDDDDD")

    # Create plot
    plt.title(title)
    plt.xlabel("Episodes", fontsize = c.AXIS_FONT_SIZE)
    plt.ylabel("Score", fontsize=c.AXIS_FONT_SIZE)
    plt.legend(title_fontsize=c.LEGEND_FONT_SIZE, fontsize = c.LEGEND_FONT_SIZE,loc="lower right", title=("Average episodes until HN convergence: " + str(convergence_mean)))
    plt.tight_layout()
    plt.grid()
    plt.savefig(unique_file(filename, "png"))
    plt.savefig(unique_file(filename, "svg"), format="svg", dpi=1200)
    plt.close()

def plot_learning_curve_annealing_BG_paper(train_scores,episodes_till_convergence, weight_hebb, weight_BG, filename):

    # Create means and standard deviations of training set scores
    train_mean = np.mean(train_scores, axis = 0)
    train_std = np.std(train_scores, axis = 0)
    train_sizes = list(range(len(train_mean)))
    convergence_mean = round(np.mean(episodes_till_convergence, dtype=np.float64).item(),1)

    #create new figure
    fig = plt.figure()

    ax = fig.add_subplot(111, label="1")
    ax.yaxis.tick_left()
    ax.set_ylabel('Score')
    ax.yaxis.set_label_position('left')

    ax2 = fig.add_subplot(111, label="2", frame_on=False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Weighted output (%)')
    ax2.yaxis.set_label_position('right')

    # Draw lines
    ax.plot(train_sizes, train_mean, color="#111111", label="Performance Hebbian Network")
    ax2.plot(train_sizes, weight_hebb, color="#006992", label="Weight Hebbian Network")
    ax2.plot(train_sizes, weight_BG, color="#ECA400", label="Weight Basal Ganglia network")

    # Draw standard deviation bands
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")

    # Create plot
    plt.title("Learning Curve")
    plt.xlabel("Episodes")
    plt.grid()
    fig.legend(title_fontsize=c.LEGEND_FONT_SIZE, fontsize = c.LEGEND_FONT_SIZE,bbox_to_anchor=(1, 0.5), loc="center left", title=("Average episodes till convergence: " + str(convergence_mean)))
    #fig.tight_layout(rect=[0, 0, 0.75, 1])
    fig.savefig(unique_file(filename, "png"), bbox_inches="tight")
    fig.savefig(unique_file(filename, "svg"), format="svg", dpi=1200, bbox_inches="tight")
    plt.close()

def plot_learning_curve_performance_and_annealing_BG_paper(hebbian_scores, episodes_till_convergence, weight_hebb, weight_BG, train_performance, filename, title = "Learning curve"):

    # Create means and standard deviations of training set scores

    convergence_mean = round(np.mean(episodes_till_convergence, dtype=np.float64).item(), 1)

    # Statistics Hebbian network
    hebb_mean, hebb_upper_std_dev, hebb_lower_std_dev = calculate_descriptive_statistics(hebbian_scores)

    #Statistics training performance
    train_mean, train_upper_std_dev, train_lower_std_dev = calculate_descriptive_statistics(train_performance)

    #Create size of x-axis based on the size of the mean lists
    size_x_axis = list(range(len(train_mean)))

    #create new figure
    fig = plt.figure()

    # Create axis for scores
    ax = fig.add_subplot(111, label="1")
    ax.yaxis.tick_left()
    ax.set_ylabel('Score', fontsize = c.AXIS_FONT_SIZE)
    ax.yaxis.set_label_position('left')

    #crete axis for weighted output of BG and Hebb
    ax2 = fig.add_subplot(111, label="2", frame_on=False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel(r'$\varphi$'+"-Decay value", fontsize = c.AXIS_FONT_SIZE)
    ax2.yaxis.set_label_position('right')

    print(hebbian_scores)
    print(size_x_axis)
    print(hebb_mean)

    # Draw lines
    ax.plot(size_x_axis, hebb_mean, color="#111111", label="Hebbian Network performance")
    ax.plot(size_x_axis, train_mean, color="#E54B4B", label="Training performance - " + r'$argmax(\varphi\mathbf{y}+(1-\varphi)\mathbf{x})$')
    ax2.plot(size_x_axis, weight_hebb, color="#006992", label= r'$\varphi$' + "-Decay value Hebbian Network")
    ax2.plot(size_x_axis, weight_BG, color="#ECA400", label= r'$\varphi$' + "-Decay value DQN")

    # Draw standard deviation bands
    ax.fill_between(size_x_axis, hebb_lower_std_dev, hebb_upper_std_dev, color="#DDDDDD")
    ax.fill_between(size_x_axis, train_lower_std_dev, train_upper_std_dev, color="#E54B4B", alpha=0.2)

    # Create plot
    plt.title(title)
    plt.xlabel("Episodes", fontsize = c.AXIS_FONT_SIZE)
    plt.grid()
    fig.legend(title_fontsize=c.LEGEND_FONT_SIZE, fontsize = c.LEGEND_FONT_SIZE, bbox_to_anchor=(1, 0.5), loc="center left", title=("Average amount of episodes until HN convergence: " + str(convergence_mean)))
    fig.savefig(unique_file(filename, "png"), bbox_inches="tight")
    fig.savefig(unique_file(filename, "svg"), format="svg", dpi=1200, bbox_inches="tight")
    plt.close()

    return hebb_mean, train_mean

def plot_learning_curve_BG_performance_and_full_supervision_paper(hebb_scores, train_performance, episodes_till_convergence, filename):

    # Create means and standard deviations of training set scores
    hebb_mean, hebb_upper_std, hebb_lower_std = calculate_descriptive_statistics(hebb_scores)

    train_mean, train_upper_std, train_lower_std = calculate_descriptive_statistics(train_performance)

    train_sizes = list(range(len(train_mean)))
    convergence_mean = round(np.mean(episodes_till_convergence, dtype=np.float64).item(),1)

    #create new figure
    plt.figure()

    # Draw lines
    plt.plot(train_sizes, hebb_mean, color="#111111", label="Performance Hebbian Network")
    plt.plot(train_sizes, train_mean, color="#E54B4B", label="Training performance")

    # Draw standard deviation bands
    plt.fill_between(train_sizes, hebb_lower_std, hebb_upper_std, color="#DDDDDD")
    plt.fill_between(train_sizes, train_lower_std, train_upper_std, color="#E54B4B", alpha=0.2)

    # Create plot
    plt.title("Learning Curve")
    plt.xlabel("Episodes", fontsize= c.AXIS_FONT_SIZE)
    plt.ylabel("Score", fontsize =c.AXIS_FONT_SIZE)
    plt.legend(loc="lower right", fontsize = c.LEGEND_FONT_SIZE, title=("Average amount of episodes until HN convergence: " + str(convergence_mean)))
    plt.tight_layout()
    plt.grid()
    plt.legend(title_fontsize=c.LEGEND_FONT_SIZE, fontsize = c.LEGEND_FONT_SIZE, bbox_to_anchor=(1, 0.5), loc="center left",
               title=("Average amount of episodes until HN convergence: " + str(convergence_mean)))
    plt.savefig(unique_file(filename, "png"))
    plt.savefig(unique_file(filename, "svg"), format="svg", dpi=1200)
    plt.close()

def plot_histograms_2_distributions_overlap(distribution_1, distribution_2, distr_1_name = "Distribution 1", distr_2_name="Distribution 2", colors = ["#FF9505", "#2E6171"]):

    # calc overlap
    overlap = []
    for num in set(distribution_1 + distribution_2):
        freq_1 = distribution_1.count(num)
        freq_2 = distribution_2.count(num)
        if freq_1 > 0 and freq_2 > 0:
            overlap.extend([num] * min(freq_1, freq_2))

    plt.figure()
    plt.hist(distribution_1, bins=np.arange(min(distribution_1), max(distribution_1) + 1.1, 1), histtype='bar', alpha=1,
             color=colors[0], label=distr_1_name)
    plt.hist(distribution_2, bins=np.arange(min(distribution_2), max(distribution_2) + 1.1, 1), histtype='bar', alpha=1,
             color=colors[1], label=distr_2_name)
    plt.hist(overlap, bins=np.arange(min(distribution_2), max(distribution_2) + 1.1, 1), histtype='bar', alpha=1,
             color=c.OVERLAP_HIST_COLOR, label="Overlap of histograms")
    plt.title("Histogram of {} and {}".format(distr_1_name, distr_2_name))
    plt.xlabel("Convergence episode", fontsize = c.AXIS_FONT_SIZE)
    plt.ylabel("Frequency", fontsize = c.AXIS_FONT_SIZE)
    plt.xticks(np.arange(min(min(distribution_1), min(distribution_2)), max(max(distribution_1), max(distribution_2)) + 2, 1.0))
    plt.yticks(np.arange(0, max(max(np.bincount(distribution_1)), max(np.bincount(distribution_2))) + 1, 1.0))
    plt.grid(color='#000000')
    plt.tight_layout()
    plt.legend(fontsize = c.LEGEND_FONT_SIZE, bbox_to_anchor=(1, 0.5), loc="center left")
    filename = "Histogram of" + distr_1_name + "and" + distr_2_name
    plt.savefig(unique_file(filename, "png"))
    plt.savefig(unique_file(filename, "svg"), format="svg", dpi=1200)
    plt.show()

def plot_1_distribution(distribution, distr_name = "Distribution 1", graph_color = "#2E6171", bin_adjust=1):
    plt.figure()
    plt.hist(distribution, bins=np.arange(min(distribution), max(distribution)+2, 1), histtype='bar', alpha=0.5, color=graph_color, label=distr_name, edgecolor = (np.array(mpl.colors.hex2color(graph_color))*0.8).tolist())
    plt.title("Histogram of {}".format(distr_name))
    plt.xlabel("Convergence episode", fontsize=c.AXIS_FONT_SIZE)
    plt.ylabel("Frequency", fontsize = c.AXIS_FONT_SIZE)
    plt.xticks(np.arange(min(distribution), max(distribution) + bin_adjust, bin_adjust))
    plt.yticks(range(0, int(max(np.bincount(distribution)))+1, 1))
    #plt.grid(color= '#000000', axis="y")
    plt.tight_layout()
    plt.legend(fontsize = c.LEGEND_FONT_SIZE, bbox_to_anchor=(1, 0.5), loc="center left")
    filename = "Histogram of" + distr_name
    plt.savefig(unique_file(filename, "png"))
    plt.savefig(unique_file(filename, "svg"), format="svg", dpi=1200)
    plt.show()

def latest_plot_histograms_2_distributions_overlap(distribution_1, distribution_2, distr_1_name = "Distribution 1", distr_2_name="Distribution 2", colors = ["#FF9505", "#2E6171"], bin_adjust = 1):

    # calc overlap
    overlap = []
    for num in set(distribution_1 + distribution_2):
        freq_1 = distribution_1.count(num)
        freq_2 = distribution_2.count(num)
        if freq_1 > 0 and freq_2 > 0:
            overlap.extend([num] * min(freq_1, freq_2))

    bin_width = 1
    x_min = min(min(distribution_1), min(distribution_2))
    x_max = max(max(distribution_1), max(distribution_2))
    x_ticks = np.arange(x_min - bin_width/2, x_max + bin_width, bin_width)

    plt.figure()
    plt.hist(distribution_1, bins=x_ticks, histtype='bar', alpha=0.5,
             color=colors[0], label=distr_1_name, edgecolor = (np.array(mpl.colors.hex2color(colors[0]))*0.8).tolist())
    plt.hist(distribution_2, bins=x_ticks, histtype='bar', alpha=0.5             ,
             color=colors[1], label=distr_2_name, edgecolor = (np.array(mpl.colors.hex2color(colors[1]))*0.8).tolist())
    #plt.hist(overlap, bins=x_ticks, histtype='bar', alpha=1,
             #color=c.OVERLAP_HIST_COLOR, label=distr_2_name, edgecolor=(np.array(mpl.colors.hex2color(c.OVERLAP_HIST_COLOR                                                          )) * 0.8).tolist())
    plt.title("Histogram of {} and {}".format(distr_1_name, distr_2_name))
    plt.xlabel("Convergence episode", fontsize = c.AXIS_FONT_SIZE)
    plt.ylabel("Frequency", fontsize = c.AXIS_FONT_SIZE)
    plt.xticks(np.arange(min(min(distribution_1), min(distribution_2)), max(max(distribution_1), max(distribution_2)) + bin_adjust, bin_adjust))
    plt.yticks(np.arange(0, max(max(np.bincount(distribution_1)), max(np.bincount(distribution_2))) + 1, 1.0))
    plt.tight_layout()
    plt.legend(fontsize = c.LEGEND_FONT_SIZE, bbox_to_anchor=(1, 0.5), loc="center left")
    filename = "Histogram of" + distr_1_name + "and" + distr_2_name
    plt.savefig(unique_file(filename, "png"))
    plt.savefig(unique_file(filename, "svg"), format="svg", dpi=1200)
    plt.show()