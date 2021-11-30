# import pdb
import pdb

import matplotlib.axis
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from babel._compat import force_text


def plot_metrics(data_vec, label_vec, rf_param_vec, fix_nbr, limits, flag, x_label, end_idx, x_labels):

    line_style = []
    line_style.append('-')
    line_style.append('--')

    plt.subplots(2, 2, figsize=(12, 6))

    # Iterate through data collect: [ random forest - depth or tree number (changing variable) , label, dice value,
    # HDRFDST value]
    for i in range(end_idx):
        metrics_table = []
        res = []
        data_s_m = data_vec[i*len(label_vec):(i+1)*len(label_vec)]
        line_temp = line_style[i]
        for data, rf_param, label in zip(data_s_m, rf_param_vec, label_vec):
            res_subset = data.loc[data['LABEL'] == label]
            label_dice = res_subset['DICE'].mean()
            label_HDRFDST = res_subset['HDRFDST'].mean()
            metrics_table.append([rf_param, label, label_dice, label_HDRFDST])

        # Get results of each label
        metrics_table = pd.DataFrame(metrics_table, columns=[flag, 'LABEL', 'DICE', 'HDRFDST'])
        for label in label_vec:
            res.append(metrics_table[metrics_table['LABEL'] == label])

        # Plot DICE graph
        plt.subplot(1, 2, 1)
        plt.gca().set_prop_cycle(None)
        # Plot DICE lines
        for ii in range(x_labels):
            plt.plot(res[ii][flag], res[ii]['DICE'], marker='o', mfc='k',ls=line_temp)

        # Plot HDRFDST graph
        plt.subplot(1, 2, 2)
        plt.gca().set_prop_cycle(None)
        # Plot HDRFDST lines
        for ii in range(x_labels):
            plt.plot(res[ii][flag], res[ii]['HDRFDST'], marker='o', mfc='k',ls=line_temp)

    # Set DICE graph layout
    plt.subplot(1, 2, 1)
    plt.xlim(limits[0][0][0], limits[0][0][1])
    plt.ylim(limits[0][1][0], limits[0][1][1])
    plt.xlabel(x_label)
    plt.ylabel("DICE")
    plt.legend(label_vec, loc='lower right')
    plt.xticks(rf_param_vec)
    plt.grid(True)

    # Set HDRFDST graph layout
    plt.subplot(1, 2, 2)
    plt.xlim(limits[1][0][0], limits[1][0][1])
    plt.ylim(limits[1][1][0], limits[1][1][1])
    plt.xlabel(x_label)
    plt.ylabel("Hausdorff Distance")
    plt.legend(label_vec)
    plt.xticks(rf_param_vec)
    plt.grid(True)

    # Depending on fix tree or depth number plot correct subtitles
    if flag == 'RF_DEPTH':
        plt.suptitle("Random Forest with number of trees = " + str(fix_nbr))
    else:
        plt.suptitle("Random Forest with tree depth = " + str(fix_nbr))

    # Show plot
    plt.draw() # show()

def main():
    # todo: load the "results.csv" file from the mia-results directory
    # todo: read the data into a list
    # todo: plot the Dice coefficients per label (i.e. white matter, gray matter, hippocampus, amygdala, thalamus)
    #  in a boxplot

    # alternative: instead of manually loading/reading the csv file you could also use the pandas package
    # but you will need to install it first ('pip install pandas') and import it to this file ('import pandas as pd')
    # pass is just a placeholder if there is no other code

    # Settings
    result_folder = 'run1'
    compareInSameGraph = True

    tree_nbr_fix = 10
    tree_depth_var = [5, 10, 20, 40, 80]
    plot_limits1 = [[[0, 90, 10], [0.0, 1.0, 0.2]],
                    [[0, 90, 10], [0.0, 50.0, 10]]]
    tree_depth_fix = 40
    tree_nbr_var = [1, 5, 10, 20, 50]
    plot_limits2 = [[[0, 60, 10], [0.0, 1.0, 0.2]],
                    [[0, 60, 10], [0, 60, 10]]]

    # Prepare variables
    label_vec = ['WhiteMatter', 'GreyMatter', 'Hippocampus', 'Amygdala', 'Thalamus']
    choose_labels = [True, True, False, True, False]
    label_vec = np.where(choose_labels, label_vec, '')

    # Remvoe empty strings !
    label_vec = ' '.join(label_vec).split()
    label_str = '-'.join(label_vec)

    # Exit if settings are wrong and not working with existing code
    if np.array([tree_depth_var]).shape != np.array([tree_nbr_var]).shape or \
            np.array([tree_nbr_fix]).shape != (1,) or np.array([tree_depth_fix]).shape != (1,):
        exit()

    len_label = len(label_vec)
    len_var = len(tree_depth_var)
    label_vec *= len_var
    tree_depth_var = np.repeat(tree_depth_var, len_label)
    tree_nbr_var = np.repeat(tree_nbr_var, len_label)

    label_ids = []
    label_ids_m = ['all']*len_var*len_label
    label_ids_s = label_vec


    res_vec_depths = []
    res_vec_trees = []

    # Prepare filename
    f_fileName = lambda x, y, z: 'TreeD-' + str(x).zfill(3) + '-TreeN-' + str(y).zfill(3) + '-Label-' + z

    # Load csv files$
    end_idx = 1
    if compareInSameGraph:
        end_idx = 2

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

    result_dir = os.path.join(os.getcwd(), 'PlotResults')
    os.makedirs(result_dir, exist_ok=True)

    # Plot results
    plot_metrics(res_vec_depths, label_vec, tree_depth_var, tree_nbr_fix, plot_limits1, 'RF_DEPTH', "Tree Depth", end_idx, len_label)
    plt.savefig(os.path.join(result_dir, 'RF_DEPTH' + '_DICE_&_HDRFDST_Result_' + label_str + '.png'))
    plt.close('all')

    plot_metrics(res_vec_trees, label_vec, tree_nbr_var, tree_depth_fix, plot_limits2, 'RF_NUM', "Tree Number", end_idx, len_label)
    plt.savefig(os.path.join(result_dir, 'RF_NUM' + '_DICE_&_HDRFDST_Result_' + label_str + '.png'))
    plt.close('all')


if __name__ == '__main__':
    main()
