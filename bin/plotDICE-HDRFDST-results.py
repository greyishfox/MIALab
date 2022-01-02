# import pdb
import matplotlib.axis
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd


def plot_metrics(data_vec, label_vec, rf_param_vec, fix_nbr, limits, rf_depth_or_num, x_label, end_idx, len_label):

    # Adapt plot settings
    font = {'weight': 'normal',
            'size': 14}
    matplotlib.rc('font', **font)

    # Prepare two line styles ( for multi-label and single-label graphs )
    line_style = ['-', '--']

    # Prepare plot
    plt.subplots(2, 2, figsize=(12, 6))

    # Iterate through data collect: [ random forest (rf_parameter) -  that can be depth or tree number (the changing
    # variable), label, dice value, HDRFDST value]
    for i in range(end_idx):
        # Prepare variables
        metrics_table = []
        res = []
        # Get all multi or single labels data
        data_s_m = data_vec[i * len(label_vec):(i + 1) * len(label_vec)]
        # Set different line style for single or multi label
        line_temp = line_style[i]
        for data, rf_param, label in zip(data_s_m, rf_param_vec, label_vec):
            res_subset = data.loc[data['LABEL'] == label]
            res_subset = res_subset[~res_subset['SUBJECT'].str.endswith('-PP')]  # Remove PP results
            label_dice = res_subset['DICE'].mean()
            label_HDRFDST = res_subset['HDRFDST'].mean()
            metrics_table.append([rf_param, label, label_dice, label_HDRFDST])

        # Get results of each label
        metrics_table = pd.DataFrame(metrics_table, columns=[rf_depth_or_num, 'LABEL', 'DICE', 'HDRFDST'])
        for label in label_vec:
            res.append(metrics_table[metrics_table['LABEL'] == label])

        # Plot DICE graph
        plt.subplot(1, 2, 1)
        plt.gca().set_prop_cycle(None)
        # Plot DICE lines
        for ii in range(len_label):
            plt.plot(res[ii][rf_depth_or_num], res[ii]['DICE'], marker='o', mfc='k', ls=line_temp)

        # Plot HDRFDST graph
        plt.subplot(1, 2, 2)
        plt.gca().set_prop_cycle(None)
        # Plot HDRFDST lines
        for ii in range(len_label):
            plt.plot(res[ii][rf_depth_or_num], res[ii]['HDRFDST'], marker='o', mfc='k', ls=line_temp)

    # Prepare labels_names (legend plot)
    labels_name = []

    for label in label_vec[:len_label]:
        labels_name.append('ML-' + label)

    if end_idx == 2:
        for label in label_vec[len_label:2*len_label]:
            labels_name.append('SL-' + label)

    # Set DICE graph layout
    plt.subplot(1, 2, 1)
    plt.xlim(limits[0][0][0], limits[0][0][1])
    plt.ylim(limits[0][1][0], limits[0][1][1])
    plt.xlabel(x_label)
    plt.ylabel("DICE")
    plt.legend(labels_name, loc='lower right', fontsize=12)
    plt.xticks(rf_param_vec)
    plt.grid(True)

    # Set HDRFDST graph layout
    plt.subplot(1, 2, 2)
    plt.xlim(limits[1][0][0], limits[1][0][1])
    # plt.ylim(limits[1][1][0], limits[1][1][1])  # set to automatic for most plots
    plt.xlabel(x_label)
    plt.ylabel("Hausdorff Distance")
    plt.legend(labels_name, fontsize=12)
    plt.xticks(rf_param_vec)
    plt.grid(True)

    # Depending on fix tree or depth number plot correct subtitles
    if rf_depth_or_num == 'RF_DEPTH':
        plt.suptitle("Random Forest with number of trees = " + str(fix_nbr))
    else:
        plt.suptitle("Random Forest with tree depth = " + str(fix_nbr))

    # Show plot
    plt.draw()  # show()


def main():

    # When you run this script, plots are created and saved directly to the
    # PlotResults folder. The result is two images with two graphs, one
    # representing the DICE coefficient and the other representing the Hausdorff
    # distance. Once a graph with fixed tree number and variable tree depth is
    # shown while the other time the tree number varies and the tree depth remains
    # fixed.
    #
    # The settings at the beginning of the main can be customized as desired.
    #
    # It is possible to output the multi label plot separately or combined with the
    # single label plots. For the combined plot, leave the variable
    # compareInSameGraph set to true.
    #
    # In the graph all or only the desired labels can be displayed which can be
    # selected with the variable choose_labels

    ####################################################################################################################
    # Settings (can be adapted)

    # load data from folder
    result_folder = 'mia-result'  # 'run_1'

    # Compare single and multilayer if not only multi layers will be plot
    compareInSameGraph = True

    # Plot series one, with fix tree number and variable tree depth
    # Adapt the parameter first series
    tree_nbr_fix = 10
    tree_depth_var = [5, 10, 20, 40, 80]  # default: [5, 10, 20, 40, 80]
    # Plot limits DICE (first row and Hausdorff second row)
    plot_limits1 = [[[0, 90, 10], [0.0, 1.0, 0.2]],
                    [[0, 90, 10], [0.0, 25.0, 5]]]

    # Plot series two, with fix tree depth and variable tree number
    # Adapt the parameter second series
    tree_depth_fix = 40
    tree_nbr_var = [1, 5, 10, 20, 50]  # default: [1, 5, 10, 20, 50]
    # Plot limits DICE (first row and Hausdorff second row)
    plot_limits2 = [[[0, 60, 10], [0.0, 1.0, 0.2]],
                    [[0, 60, 10], [0, 90, 10]]]

    # Set labels: 'WhiteMatter', 'GreyMatter', 'Hippocampus', 'Amygdala', 'Thalamus'
    choose_labels = [True, True, False, True, False]  # Often used: [False, False, True, False, True],

    ####################################################################################################################
    # Other variables
    label_vec = ['WhiteMatter', 'GreyMatter', 'Hippocampus', 'Amygdala', 'Thalamus']
    label_vec = np.where(choose_labels, label_vec, '')

    # Remove empty strings !
    label_vec = ' '.join(label_vec).split()
    label_str = '-'.join(label_vec)

    # Exit if settings are wrong and not working with existing code
    if np.array([tree_depth_var]).shape != np.array([tree_nbr_var]).shape or \
            np.array([tree_nbr_fix]).shape != (1,) or np.array([tree_depth_fix]).shape != (1,):
        exit()

    # Prepare all vectors to correctly plot points in graphs
    len_label = len(label_vec)
    len_var = len(tree_depth_var)
    label_vec *= len_var
    tree_depth_var = np.repeat(tree_depth_var, len_label)
    tree_nbr_var = np.repeat(tree_nbr_var, len_label)

    label_ids = []
    label_ids_m = ['all'] * len_var * len_label
    label_ids_s = label_vec

    res_vec_depths = []
    res_vec_trees = []

    # Prepare filename
    f_fileName = lambda x, y, z: 'TreeD-' + str(x).zfill(3) + '-TreeN-' + str(y).zfill(3) + '-Label-' + z

    # Load csv files$
    end_idx = 1
    if compareInSameGraph:
        end_idx = 2

    # Load all needed files
    for i in range(end_idx):
        if i == 0:
            label_ids = label_ids_m
        if i:
            label_ids = label_ids_s

        for label, tree_depth, tree_nbr in zip(label_ids, tree_depth_var, tree_nbr_var):
            path = os.path.join(result_folder, f_fileName(tree_depth, tree_nbr_fix, label), 'results.csv')
            res_vec_depths.append(pd.read_csv(path, sep=';'))
            path = os.path.join(result_folder, f_fileName(tree_depth_fix, tree_nbr, label), 'results.csv')
            res_vec_trees.append(pd.read_csv(path, sep=';'))

    # Set and create save path
    result_dir = os.path.join(os.getcwd(), 'PlotResults')
    os.makedirs(result_dir, exist_ok=True)

    # Plot and save results
    # First series, with fix tree number and variable tree depth
    plot_metrics(res_vec_depths, label_vec, tree_depth_var, tree_nbr_fix, plot_limits1, 'RF_DEPTH', "Tree Depth",
                 end_idx, len_label)
    plt.savefig(os.path.join(result_dir, 'RF_DEPTH' + '_DICE_&_HDRFDST_Result_' + label_str + '_' +
                             str(end_idx) + '.png'))
    plt.close('all')

    # First series, with fix tree depth and variable tree number
    plot_metrics(res_vec_trees, label_vec, tree_nbr_var, tree_depth_fix, plot_limits2, 'RF_NUM', "Tree Number", end_idx,
                 len_label)
    plt.savefig(os.path.join(result_dir, 'RF_NUM' + '_DICE_&_HDRFDST_Result_' + label_str + '_' +
                             str(end_idx) + '.png'))
    plt.close('all')


if __name__ == '__main__':
    main()
