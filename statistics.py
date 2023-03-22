from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import itertools
import os.path
import csv
import ast
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from scipy import stats
import plots
from collections import Iterable
import pingouin as pg
import pandas as pd

def flatten(lis): # thanks to https://stackoverflow.com/questions/17485747/how-to-convert-a-nested-list-into-a-one-dimensional-list-in-python
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item

def paired_samples_t_test_convergence(distribution_1, distribution_2, distr_1_name='Distribution 1',
                                      distr_2_name='Distribution 2'):
    descriptive_stats_list = []
    name_list = [distr_1_name, distr_2_name]

    # Calculate descriptive stats
    descriptive_stats_list.append(stats.describe(distribution_1))
    descriptive_stats_list.append(stats.describe(distribution_2))

    # print descriptive stats
    for i in range(len(name_list)):
        print("-------------------------------------------")
        print("Descriptive statistics {}".format(name_list[i]))
        print("")
        print("Number of observations: {}".format(descriptive_stats_list[i][0]))
        print("Minimum: {}".format(descriptive_stats_list[i][1][0]))
        print("Maximum: {}".format(descriptive_stats_list[i][1][1]))
        print("Mean: {}".format(descriptive_stats_list[i][2]))
        print("Variance: {}".format(descriptive_stats_list[i][3]))
        print("Skewness: {}".format(descriptive_stats_list[i][4]))
        print("Kurtosis: {}".format(descriptive_stats_list[i][5]))
        print("-------------------------------------------")

    # Make a histogram of both distributions
    #plots.plot_histograms_2_distributions_overlap(distribution_1, distribution_2, distr_1_name=name_list[0],
#                                                  distr_2_name=name_list[1])
    #make seperate plots
    #for i in range(len(name_list)):
     #   plots.plot_1_distribution(distribution_list[i], name_list[i], graph_color=color_list[i])

    # Check for normality
    if (stats.shapiro(distribution_1)[1] < 0.05) or (stats.shapiro(distribution_2)[1] < 0.05):
        print("-------------------------------------------")
        print("Shapiro-Wilk test for normality turned out significant.")
        print("Assumption for normality violated, one should not continue this statistic analysis")
        print("Proceeding anyways")
        print("-------------------------------------------")

    # conduct the paired samples t-test
    paired_t_test = stats.ttest_rel(distribution_1, distribution_2)
    print("-------------------------------------------")
    print("Paired sample t-test")
    print("Comparing {} against {}".format(distr_1_name, distr_2_name))
    print("T-statistic: {}".format(paired_t_test[0]))
    print("p-value: {}".format(paired_t_test[1]))

    if paired_t_test[1] <= 0.05:
        print("Statistic significant!")
    else:
        print("Statistic not significant")

    namefile = distr_1_name + "vs" + distr_2_name
    filename = unique_file(namefile, "txt")
    write_t_test_to_txt(distribution_1, distribution_2, distr_1_name,
                                      distr_2_name,filename)

    return

def write_t_test_to_txt(distribution_1, distribution_2, distr_1_name='Distribution 1',
                                      distr_2_name='Distribution 2',
                                      file_name='t-test output.txt'):

    descriptive_stats_list = []
    name_list = [distr_1_name, distr_2_name]

    # Calculate descriptive stats
    descriptive_stats_list.append(stats.describe(distribution_1))
    descriptive_stats_list.append(stats.describe(distribution_2))

    # write descriptive stats to file
    with open(file_name, 'w') as f:
        for i in range(len(name_list)):
            f.write("-------------------------------------------\n")
            f.write("Descriptive statistics {}\n".format(name_list[i]))
            f.write("\n")
            f.write("Number of observations: {}\n".format(descriptive_stats_list[i][0]))
            f.write("Minimum: {}\n".format(descriptive_stats_list[i][1][0]))
            f.write("Maximum: {}\n".format(descriptive_stats_list[i][1][1]))
            f.write("Mean: {}\n".format(descriptive_stats_list[i][2]))
            f.write("Variance: {}\n".format(descriptive_stats_list[i][3]))
            f.write("Skewness: {}\n".format(descriptive_stats_list[i][4]))
            f.write("Kurtosis: {}\n".format(descriptive_stats_list[i][5]))
            f.write("-------------------------------------------\n")

    # Check for normality
    if (stats.shapiro(distribution_1)[1] < 0.05) or (stats.shapiro(distribution_2)[1] < 0.05):
        with open(file_name, 'a') as f:
            f.write("-------------------------------------------\n")
            f.write("Shapiro-Wilk test for normality turned out significant.\n")
            f.write("Assumption for normality violated, one should not continue this statistic analysis\n")
            f.write("Proceeding anyways\n")
            f.write("-------------------------------------------\n")

    # conduct the paired samples t-test
    paired_t_test = stats.ttest_rel(distribution_1, distribution_2)
    with open(file_name, 'a') as f:
        f.write("-------------------------------------------\n")
        f.write("Paired sample t-test\n")
        f.write("Comparing {} against {}\n".format(distr_1_name, distr_2_name))
        f.write("T-statistic: {}\n".format(paired_t_test[0]))
        f.write("p-value: {}\n".format(paired_t_test[1]))

        if paired_t_test[1] <= 0.05:
            f.write("Statistic significant!\n")
        else:
            f.write("Statistic not significant\n")

    return


def read_and_ready_data_from_file(file_name, data_structure, subsets=False, subset_num=0):
    if data_structure == "t-test":

        convergence_episodes = []

        with open(file_name, 'r') as data_file:
            csv_reader = csv.DictReader(data_file, delimiter='\t')

            if subsets == True:
                print("collecting data from {} for t-test from subsample {}".format(file_name, subset_num))

                for line in csv_reader:

                    if int(line['subsample']) == subset_num:
                        convergence_episodes.append(int(line['convergence_episodes']))

                return convergence_episodes

            print("collecting data from {} for t-test from whole file".format(file_name))
            for line in csv_reader:
                convergence_episodes.append(int(line['convergence_episodes']))

            return convergence_episodes

    if data_structure == "full_supervision_plot":

        train_score = []
        hebb_score = []
        convergence_episodes = []

        with open(file_name, 'r') as data_file:
            csv_reader = csv.DictReader(data_file, delimiter='\t')

            if subsets == True:
                print(
                    "collecting data from {} for full supervision plot from subsample {}".format(file_name, subset_num))

                for line in csv_reader:

                    if int(line['subsample']) == subset_num:
                        train_score.append(ast.literal_eval(line['training_performance']))  # provides nested list
                        hebb_score.append(ast.literal_eval(line['training_performance']))  # provides nested list
                        convergence_episodes.append(int(line['convergence_episodes']))  # provides a list with convergence distribution

                return train_score, hebb_score, convergence_episodes

            print("collecting data from {} for full supervision plot from whole file".format(file_name))
            for line in csv_reader:
                train_score.append(ast.literal_eval(line['training_performance']))  # provides nested list
                hebb_score.append(ast.literal_eval(line['training_performance']))  # provides nested list
                convergence_episodes.append(int(line['convergence_episodes']))  # provides a list with convergence distribution

            return train_score, hebb_score, convergence_episodes

    if data_structure == "decay_plot":

        train_score = []
        hebb_score = []
        convergence_episodes = []
        BG_decay = []
        hebb_decay = []

        with open(file_name, 'r') as data_file:
            csv_reader = csv.DictReader(data_file, delimiter='\t')

            if subsets == True:
                print(
                    "collecting data from {} for full supervision plot from subsample {}".format(file_name, subset_num))

                for line in csv_reader:

                    if int(line['subsample']) == subset_num:
                        train_score.append(ast.literal_eval(line['training_performance']))  # provides nested list
                        hebb_score.append(ast.literal_eval(line['training_performance']))  # provides nested list
                        convergence_episodes.append(int(line['convergence_episodes']))  # provides a list with convergence distribution
                        BG_decay = ast.literal_eval(line['BG_decay'])
                        hebb_decay = ast.literal_eval(line['hebb_decay'])

            return train_score, hebb_score, convergence_episodes, BG_decay, hebb_decay

    if data_structure == "combined":
        _, supervised_performance, _ = read_and_ready_data_from_file("exponential_data_file.csv",
                                                                     "full_supervision_plot", subsets=subsets,
                                                                     subset_num=subset_num)
        _, exp_performance, _, _, _ = read_and_ready_data_from_file("exponential_data_file.csv", "decay_plot",
                                                                    subsets=subsets, subset_num=subset_num)
        _, lin_performance, _, _, _ = read_and_ready_data_from_file("exponential_data_file.csv", "decay_plot",
                                                                    subsets=subsets, subset_num=subset_num)

        return supervised_performance, exp_performance, lin_performance

    raise Exception("No data_structure given to read function. No data gathered. Proces terminated")

def save_data_to_file(data, filetype="None"):
    if filetype == "exp":

        fieldnames = ['name', 'subsample', 'convergence_episodes', 'mean_convergence', 'training_performance', \
                      'performance_hebb', 'hebb_decay', 'BG_decay']

        if os.path.exists('exponential_data_file.csv') == False:
            with open('exp_data_file.csv', 'w') as data_file:
                csv_writer = csv.writer(data_file, delimiter='\t')
                csv_writer.writerow(fieldnames)

                if len(data) != len(fieldnames):
                    raise Exception("Error in saving data in exponential datafile, the list of data does not match length of fieldnames. Data will be stored incorrect, terminated process. Make sure the datastructure is given correctly.")

                csv_writer.writerow(data)

        elif os.path.exists('exponential_data_file.csv') == True:
            with open('exp_data_file.csv', 'a') as data_file:
                if len(data) != len(fieldnames):
                    raise Exception("Error in saving data in exponential datafile, the list of data does not match length of fieldnames. Data will be stored incorrect, terminated process. Make sure the datastructure is given correctly.")

                csv_writer = csv.writer(data_file, delimiter='\t')
                csv_writer.writerow(data)

    if filetype == "lin":

        fieldnames = ['name', 'subsample', 'convergence_episodes', 'training_performance', \
                      'performance_hebb', 'hebb_decay', 'BG_decay']

        if os.path.exists('linear_data_file.csv') == False:
            with open('linear_data_file.csv', 'w') as data_file:

                csv_writer = csv.writer(data_file, delimiter='\t')
                csv_writer.writerow(fieldnames)

                if len(data) != len(fieldnames):
                    raise Exception("Error in saving data in linear data file, the list of data does not match length of fieldnames. Data will be stored incorrect, terminated process. Make sure the datastructure is given correctly.")
                csv_writer.writerow(data)

        elif os.path.exists('linear_data_file.csv') == True:
            with open('linear_data_file.csv', 'a') as data_file:

                if len(data) != len(fieldnames):
                    raise Exception("Error in saving data in linar data file, the list of data does not match length of fieldnames. Data will be stored incorrect, terminated process. Make sure the datastructure is given correctly.")

                csv_writer = csv.writer(data_file, delimiter='\t')
                csv_writer.writerow(data)

    if filetype == "supervised":
        if os.path.exists('supervised_data_file.csv') == False:
            with open('supervised_data_file.csv', 'w') as data_file:

                fieldnames = ['name', 'subsample', 'convergence_episodes', 'training_performance', \
                              'performance_hebb',]

                csv_writer = csv.writer(data_file, delimiter='\t')
                csv_writer.writerow(fieldnames)

                csv_writer.writerow(data)

        elif os.path.exists('supervised_data_file.csv') == True:
            with open('supervised_data_file.csv', 'a') as data_file:
                csv_writer = csv.writer(data_file, delimiter='\t')
                csv_writer.writerow(data)

    if filetype == "None":
        print("No filetype given! Saving data in a seperate random file")

        if os.path.exists('NoneType_data_file.csv') == False:
            with open('NoneType_data_file.csv', 'w') as data_file:

                csv_writer = csv.writer(data_file, delimiter='\t')
                csv_writer.writerow(data)

        elif os.path.exists('NoneType_data_file.csv') == True:
            with open('NoneType_data_file.csv', 'a') as data_file:
                csv_writer = csv.writer(data_file, delimiter='\t')
                csv_writer.writerow(data)

    return

def t_test_kolmogorov(distribution_1, distribution_2, distr_1_name='Distribution 1',
                                      distr_2_name='Distribution 2',
                                      file_name='t-test output.txt'):

    descriptive_stats_list = []
    name_list = [distr_1_name, distr_2_name]

    # Calculate descriptive stats
    descriptive_stats_list.append(stats.describe(distribution_1))
    descriptive_stats_list.append(stats.describe(distribution_2))

    # write descriptive stats to file
    with open(file_name, 'w') as f:
        for i in range(len(name_list)):
            f.write("-------------------------------------------\n")
            f.write("Descriptive statistics {}\n".format(name_list[i]))
            f.write("\n")
            f.write("Number of observations: {}\n".format(descriptive_stats_list[i][0]))
            f.write("Minimum: {}\n".format(descriptive_stats_list[i][1][0]))
            f.write("Maximum: {}\n".format(descriptive_stats_list[i][1][1]))
            f.write("Mean: {}\n".format(descriptive_stats_list[i][2]))
            f.write("Variance: {}\n".format(descriptive_stats_list[i][3]))
            f.write("Skewness: {}\n".format(descriptive_stats_list[i][4]))
            f.write("Kurtosis: {}\n".format(descriptive_stats_list[i][5]))
            f.write("-------------------------------------------\n")

    # Check for normality
    ks_stat1, ks_p1 = stats.kstest(distribution_1, 'norm')
    ks_stat2, ks_p2 = stats.kstest(distribution_2, 'norm')

    if (ks_p1 <= 0.05) or (ks_p2 <= 0.05):
        with open(file_name, 'a') as f:
            f.write("-------------------------------------------\n")
            f.write("Kolmogorov smirnov test for normality turned out significant\n")
            if ks_p1 <= 0.05:
                f.write("{} distribution not normally distributed\n".format(distr_1_name))
            if ks_p2 <= 0.05:
                f.write("{} distribution not normally distributed\n".format(distr_2_name))
            f.write("KS statistic distribution 1: {}, distribution 2: {}\n".format(ks_stat1, ks_stat2))
            f.write("P-value distribution 1: {}, distribution 2: {}\n".format(ks_p1, ks_p2))
            f.write("Assumption for normality violated, continuing with wilcoxon signed rank test\n")
            test_stat, p_val = stats.wilcoxon(distribution_1, distribution_2)
            f.write("-------------------------------------------\n")
            f.write("Wilcoxon signed rank test\n")
            f.write("Comparing {} against {}\n".format(distr_1_name, distr_2_name))
            f.write("Wilcoxon statistic: {}\n".format(test_stat))
            f.write("p-value: {}\n".format(p_val))
            if p_val <= 0.05:
                f.write("Wilcoxon Statistic significant!\n")
            else:
                f.write("Wilcoxon Statistic not significant\n")
            f.write("-------------------------------------------\n")

            return

    # conduct the paired samples t-test
    paired_t_test = stats.ttest_rel(distribution_1, distribution_2)
    with open(file_name, 'a') as f:
        f.write("-------------------------------------------\n")
        f.write("Paired sample t-test\n")
        f.write("Comparing {} against {}\n".format(distr_1_name, distr_2_name))
        f.write("T-statistic: {}\n".format(paired_t_test[0]))
        f.write("p-value: {}\n".format(paired_t_test[1]))

        if paired_t_test[1] <= 0.05:
            f.write("Paired sample t-test Statistic significant!\n")
        else:
            f.write("Paired sample t-test Statistic not significant\n")

    return

def welch_ANOVA(distr_1,distr_2,distr_3, distr_1_name = "Distribution 1", distr_2_name="Distribution 2", distr_3_name="Distribution 2", file_name="Welch_ANOVA.txt"):
    # Check for equality of variances using bartlett

    bartlett_stat, bartlett_p_value = stats.bartlett(distr_1, distr_2, distr_3)

    if (bartlett_p_value <= 0.05):
        print("bartlett stat significant!")
        with open(file_name, 'a') as f:
            f.write("-------------------------------------------\n")
            f.write("Bartlett test for equal variances turned out significant\n")
            f.write("Bartlett statistic: {}\n".format(round(bartlett_stat,3)))
            f.write("Bartlett p_value: {}\n".format(round(bartlett_p_value,3)))
            f.write("Welch ANOVA can be performed\n")

            df = pd.DataFrame({'Value': np.concatenate((distr_1, distr_2, distr_3)), 'Group': np.repeat([distr_1_name, distr_2_name, distr_3_name], int(len(distr_1)))})
            welch_anova = pg.welch_anova(dv='Value', between='Group', data=df)
            f.write('\n\nWelch ANOVA:\n')
            f.write(welch_anova.to_string())

            # perform Welch's ANOVA
    print("bartlet statt non-sig: {}".format(bartlett_p_value))
    return







# Thanks to: http://www.jtrive.com/determining-histogram-bin-width-using-the-freedman-diaconis-rule.html
def freedman_bins_calculator(data):
    data = np.asarray(data, dtype=np.float_)
    IQR = stats.iqr(data, rng=(25, 75), scale="raw", nan_policy="omit")
    N = data.size
    bw = ((2 * IQR) / np.power(N, 1 / 3))

    datmin, datmax = data.min(), data.max()
    datrng = datmax - datmin
    result = int(datrng / bw) + 1

    return result

def unique_file(basename, ext):
    actualname = "%s.%s" % (basename, ext)
    c = itertools.count()
    while os.path.exists(actualname):
        actualname = "%s (%d).%s" % (basename, next(c), ext)
    return actualname
