# import pdb
import pdb

import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd


def plot_metrics(data_vec, label_vec, rf_param_vec, fix_nbr, limits, flag, x_label):
    metrics_table = []
    res = []

    # Iterate through data collect: [ random forest - depth or tree number (changing variable) , label, dice value,
    # HDRFDST value]
    for data, rf_param, label in zip(data_vec, rf_param_vec, label_vec):
        res_subset = data.loc[data['LABEL'] == label]
        label_dice = res_subset['DICE'].mean()
        label_HDRFDST = res_subset['HDRFDST'].mean()
        metrics_table.append([rf_param, label, label_dice, label_HDRFDST])

    # Get results of each label
    metrics_table = pd.DataFrame(metrics_table, columns=[flag, 'LABEL', 'DICE', 'HDRFDST'])
    for label in label_vec:
        res.append(metrics_table[metrics_table['LABEL'] == label])

    # Plot DICE graph
    plt.subplots(2, 2, figsize=(12, 6))
    plt.subplot(1, 2, 1)

    # Plot DICE lines
    plt.plot(res[0][flag], res[0]['DICE'], res[1][flag], res[1]['DICE'],
             res[2][flag], res[2]['DICE'], res[3][flag], res[3]['DICE'],
             res[4][flag], res[4]['DICE'], marker='o', mfc='k')

    # Set DICE graph layout
    plt.xlim(limits[0][0][0], limits[0][0][1])
    plt.ylim(limits[0][1][0], limits[0][1][1])
    plt.xlabel(x_label)
    plt.ylabel("DICE")
    plt.legend(label_vec, loc='lower right')
    plt.grid(True)

    # Plot HDRFDST graph
    plt.subplot(1, 2, 2)

    # Plot HDRFDST lines
    plt.plot(res[0][flag], res[0]['HDRFDST'], res[1][flag], res[1]['HDRFDST'],
             res[2][flag], res[2]['HDRFDST'], res[3][flag], res[3]['HDRFDST'],
             res[4][flag], res[4]['HDRFDST'], marker='o', mfc='k')

    # Set HDRFDST graph layout
    plt.xlim(limits[1][0][0], limits[1][0][1])
    plt.ylim(limits[1][1][0], limits[1][1][1])
    plt.xlabel(x_label)
    plt.ylabel("Hausdorff Distance")
    plt.legend(label_vec)
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
    single_label_flag = False
    result_folder = 'run1'

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
    choose_labels = [True, True, True, True, True]
    label_vec = np.where(choose_labels, label_vec, '')

    # Remvoe empty strings !
    label_vec = ' '.join(label_vec).split()

    # Exit if settings are wrong and not working with existing code
    if np.array([tree_depth_var]).shape != np.array([tree_nbr_var]).shape or \
            np.array([tree_nbr_fix]).shape != (1,) or np.array([tree_depth_fix]).shape != (1,):
        exit()

    len_label = len(label_vec)
    len_var = len(tree_depth_var)
    label_vec *= len_var
    tree_depth_var = np.repeat(tree_depth_var, len_label)
    tree_nbr_var = np.repeat(tree_nbr_var, len_label)

    if single_label_flag:
        label_ids = label_vec
        save_text = 'single label'
    else:
        label_ids = ['all']*len_var*len_label
        save_text = 'multi label'

    res_vec_depths = []
    res_vec_trees = []

    # Prepare filename
    f_fileName = lambda x, y, z: 'TreeD-' + str(x).zfill(3) + '-TreeN-' + str(y).zfill(3) + '-Label-' + z

    # Load csv files
    for label, tree_depth, tree_nbr in zip(label_vec, tree_depth_var, tree_nbr_var):
        path = os.path.join(result_folder, f_fileName(tree_depth, tree_nbr_fix, label), 'results.csv')
        res_vec_depths.append(pd.read_csv(path, sep=';'))
        path = os.path.join(result_folder, f_fileName(tree_depth_fix, tree_nbr, label), 'results.csv')
        res_vec_trees.append(pd.read_csv(path, sep=';'))

    # pdb.set_trace()
    result_dir = os.path.join(os.getcwd(), 'PlotResults')
    os.makedirs(result_dir, exist_ok=True)


    # Plot results
    plot_metrics(res_vec_depths, label_vec, tree_depth_var, tree_nbr_fix, plot_limits1, 'RF_DEPTH', "Tree Depth")
    plt.savefig(os.path.join(result_dir, save_text + '_' + 'RF_DEPTH' + '_DICE_&_HDRFDST_Result + label_vec + .png'))
    plt.close('all')

    plot_metrics(res_vec_trees, label_vec, tree_nbr_var, tree_depth_fix, plot_limits2, 'RF_NUM', "Tree Number")
    plt.savefig(os.path.join(result_dir, save_text + '_' + 'RF_NUM' + '_DICE_&_HDRFDST_Result + label_vec + .png'))
    plt.close('all')


if __name__ == '__main__':
    main()
