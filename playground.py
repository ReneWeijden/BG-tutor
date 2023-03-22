import statistics as s
import plots as p
import constants as c



distr_linear = [10, 9, 26, 12, 13, 11, 16, 14, 9, 10, 12, 9, 13, 8, 11, 8, 11, 6, 9, 11, 6, 6, 11, 10, 13, 10, 7, 20, 16, 11, 14, 16, 8, 18, 14, 14, 3, 13, 8, 9, 13, 12, 16, 11, 7, 7, 7, 8, 14, 15, 7, 7, 12, 25, 21, 8, 10, 10, 7, 10]
distr_supervised = [35, 18, 13, 31, 18, 81, 13, 30, 12, 32, 14, 29, 41, 13, 20, 30, 21, 59, 35, 17, 15, 20, 38, 14, 53, 20, 22, 7, 8, 18, 51, 9, 18, 10, 66, 19, 61, 12, 21, 11, 12, 8, 21, 35, 23, 56, 15, 27, 5, 15, 13, 19, 25, 23, 23, 36, 13, 31, 23, 33]
distr_exponential = [13, 13, 16, 14, 24, 9, 6, 43, 11, 26, 13, 6, 12, 12, 9, 20, 12, 7, 6, 7, 6, 7, 44, 7, 6, 13, 12, 6, 16, 7, 8, 11, 10, 9, 39, 10, 8, 5, 5, 24, 10, 9, 8, 6, 8, 8, 7, 30, 13, 7, 10, 8, 6, 3, 23, 7, 8, 10, 7, 10]

s.welch_ANOVA(distr_linear,distr_supervised,distr_exponential, distr_1_name = "Linear", distr_2_name="Supervised", distr_3_name="Exponential", file_name="Welch_ANOVA.txt")

s.t_test_kolmogorov(distr_linear, distr_supervised, distr_1_name='Linear decay',
                                    distr_2_name='Supervised',
                                   file_name='linear vs supervised.txt')

s.t_test_kolmogorov(distr_exponential, distr_supervised, distr_1_name='exponential',
                                    distr_2_name='supervised',
                                   file_name='exponential vs supervised.txt')
s.t_test_kolmogorov(distr_linear, distr_exponential, distr_1_name='Linear decay',
                                    distr_2_name='Exponential decay',
                                   file_name='linear vs exponental.txt')

#plot
p.latest_plot_histograms_2_distributions_overlap(distr_linear, distr_supervised,
                                              distr_1_name="Linear " + r'$\varphi$' + "-decay",
                                              distr_2_name="No " + r'$\varphi$' + "-decay",
                                              colors=[c.LINEAR_HIST_COLOR, c.SUPERVISED_HIST_COLOR],bin_adjust = 3)
p.latest_plot_histograms_2_distributions_overlap(distr_exponential, distr_linear,
                                              distr_2_name="Linear " + r'$\varphi$' + "-decay",
                                              distr_1_name="Exponential " + r'$\varphi$' + "-decay",
                                              colors=[c.EXPONENTIAL_HIST_COLOR, c.LINEAR_HIST_COLOR], bin_adjust = 2)
p.latest_plot_histograms_2_distributions_overlap(distr_exponential, distr_supervised,
                                              distr_1_name="Exponential " + r'$\varphi$' + "-decay",
                                              distr_2_name="No " + r'$\varphi$' + "-decay",
                                              colors=[c.EXPONENTIAL_HIST_COLOR, c.SUPERVISED_HIST_COLOR],bin_adjust = 3)
# def plot_1_distribution(distribution, distr_name = "Distribution 1", graph_color = "#2E6171"):
p.plot_1_distribution(distr_linear, "Linear " + r'$\varphi$' + "-decay", c.LINEAR_HIST_COLOR)
p.plot_1_distribution(distr_exponential, "Exponential " + r'$\varphi$' + "-decay", c.EXPONENTIAL_HIST_COLOR, bin_adjust=2)
p.plot_1_distribution(distr_supervised, "No " + r'$\varphi$' + "-decay", c.SUPERVISED_HIST_COLOR, bin_adjust=3)
#latest_plot_histograms_2_distributions_overlap(distribution_1, distribution_2, distr_1_name = "Distribution 1", distr_2_name="Distribution 2", colors = ["#FF9505", "#2E6171"]):