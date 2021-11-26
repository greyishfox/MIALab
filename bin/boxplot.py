import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd

def showBoxPlot(data_vec, label_vec):
    res_table = []
    for i in range(len(label_vec)):
        res_subset = data_vec[3].loc[data_vec[3]['LABEL'] == label_vec[i]]
        res_table.append(pd.DataFrame(res_subset, columns=['SUBJECT', 'LABEL', 'DICE', 'HDRFDST']))

    #print(res_table)
    W_res = res_table[0]
    G_res = res_table[1]
    H_res = res_table[2]
    A_res = res_table[3]
    T_res = res_table[4]
    print(W_res)
    W_res.boxplot(by='LABEL', column='DICE', figsize=(10, 10))
    plt.ylim(0, 1)

    plt.show()

def plotMetrics(data_vec, label_vec, rf_param_vec, flag):
    metrics_table = []
    for i in range(5):
        for j in range(5):
            res_subset = data_vec[i].loc[data_vec[i]['LABEL'] == label_vec[j]]
            label_dice = res_subset['DICE'].mean()
            label_HDRFDST = res_subset['HDRFDST'].mean()
            metrics_table.append([rf_param_vec[i], label_vec[j], label_dice, label_HDRFDST])


    if flag == 0:
        metrics_table = pd.DataFrame(metrics_table, columns=['RF_DEPTH', 'LABEL', 'DICE', 'HDRFDST'])

        w_res = metrics_table[metrics_table['LABEL'] == label_vec[0]]
        g_res = metrics_table[metrics_table['LABEL'] == label_vec[1]]
        h_res = metrics_table[metrics_table['LABEL'] == label_vec[2]]
        a_res = metrics_table[metrics_table['LABEL'] == label_vec[3]]
        t_res = metrics_table[metrics_table['LABEL'] == label_vec[4]]

        plt.subplot(1, 2, 1)
        plt.xlim(0, 90, 10)
        plt.ylim(0.0, 1.0, 0.2)
        plt.xlabel("Tree Depth")
        plt.ylabel("DICE")
        plt.plot(w_res['RF_DEPTH'], w_res['DICE'], g_res['RF_DEPTH'], g_res['DICE'],
                 h_res['RF_DEPTH'], h_res['DICE'], a_res['RF_DEPTH'], a_res['DICE'],
                 t_res['RF_DEPTH'], t_res['DICE'], marker='o', mfc='k')

        plt.legend(label_vec, loc='lower right')

        plt.subplot(1, 2, 2)
        plt.xlim(0, 90, 10)
        plt.ylim(0.0, 50.0, 10)
        plt.xlabel("Tree Depth")
        plt.ylabel("Hausdorff Distance")
        plt.plot(w_res['RF_DEPTH'], w_res['HDRFDST'], g_res['RF_DEPTH'], g_res['HDRFDST'],
                 h_res['RF_DEPTH'], h_res['HDRFDST'], a_res['RF_DEPTH'], a_res['HDRFDST'],
                 t_res['RF_DEPTH'], t_res['HDRFDST'], marker='o', mfc='k')

        plt.legend(label_vec)
        plt.suptitle("Random Forest with nTrees = 10")
        plt.figure(1)
        plt.show()

    else:
        metrics_table = pd.DataFrame(metrics_table, columns=['RF_NUM', 'LABEL', 'DICE', 'HDRFDST'])

        w_res = metrics_table[metrics_table['LABEL'] == label_vec[0]]
        g_res = metrics_table[metrics_table['LABEL'] == label_vec[1]]
        h_res = metrics_table[metrics_table['LABEL'] == label_vec[2]]
        a_res = metrics_table[metrics_table['LABEL'] == label_vec[3]]
        t_res = metrics_table[metrics_table['LABEL'] == label_vec[4]]

        plt.subplot(1, 2, 1)
        plt.xlim(0, 60, 10)
        plt.ylim(0.0, 1.0, 0.2)
        plt.xlabel("Tree Number")
        plt.ylabel("DICE")
        plt.plot(w_res['RF_NUM'], w_res['DICE'], g_res['RF_NUM'], g_res['DICE'],
                 h_res['RF_NUM'], h_res['DICE'], a_res['RF_NUM'], a_res['DICE'],
                 t_res['RF_NUM'], t_res['DICE'], marker='o', mfc='k')

        plt.legend(label_vec)

        plt.subplot(1, 2, 2)
        plt.xlim(0, 60, 10)
        plt.ylim(0.0, 50.0, 10)
        plt.xlabel("Tree Number")
        plt.ylabel("Hausdorff Distance")
        plt.plot(w_res['RF_NUM'], w_res['HDRFDST'], g_res['RF_NUM'], g_res['HDRFDST'],
                 h_res['RF_NUM'], h_res['HDRFDST'], a_res['RF_NUM'], a_res['HDRFDST'],
                 t_res['RF_NUM'], t_res['HDRFDST'], marker='o', mfc='k')

        plt.legend(label_vec)
        plt.suptitle("Random Forest with Tree_d = 20")
        plt.figure(2)
        plt.show()

def main():
    # todo: load the "results.csv" file from the mia-results directory
    # todo: read the data into a list
    # todo: plot the Dice coefficients per label (i.e. white matter, gray matter, hippocampus, amygdala, thalamus)
    #  in a boxplot

    #file = open('../bin/mia-result/2021-10-10-14-14-12/results.csv')
    #type(file)

    # choose the correct delimiter to separate the data
    #csvreader = csv.reader(file, delimiter=';')

    # create a header list from the first row in the file
    #header = next(csvreader)

    # read the data from the second row (first row is the header) to the last row into the list "data"
    # the function "next" from above already jumped to the second row
    #dataSet = []
    #for data in csvreader:
    #    dataSet.append(data)

    #file.close()

    # alternative: instead of manually loading/reading the csv file you could also use the pandas package
    # but you will need to install it first ('pip install pandas') and import it to this file ('import pandas as pd')
    # pass is just a placeholder if there is no other code

    # Fixed tree number (ntrees = 10) but varying tree depth (5, 10, 20, 40, 80)
    df_N10_D05 = pd.read_csv('mia-result/TreeN-10-TreeD-05/results.csv', sep=';')
    df_N10_D10 = pd.read_csv('mia-result/TreeN-10-TreeD-10/results.csv', sep=';')
    df_N10_D20 = pd.read_csv('mia-result/TreeN-10-TreeD-20/results.csv', sep=';')
    df_N10_D40 = pd.read_csv('mia-result/TreeN-10-TreeD-40/results.csv', sep=';')
    df_N10_D80 = pd.read_csv('mia-result/TreeN-10-TreeD-80/results.csv', sep=';')

    # Fixed tree depth (tree_d = 20) but varying tree number (1, 5, 10, 20, 50)
    df_D40_N01 = pd.read_csv('mia-result/TreeD-40-TreeN-01/results.csv', sep=';')
    df_D40_N05 = pd.read_csv('mia-result/TreeD-40-TreeN-05/results.csv', sep=';')
    df_D40_N10 = pd.read_csv('mia-result/TreeD-40-TreeN-10/results.csv', sep=';')
    df_D40_N20 = pd.read_csv('mia-result/TreeD-40-TreeN-20/results.csv', sep=';')
    df_D40_N50 = pd.read_csv('mia-result/TreeD-40-TreeN-50/results.csv', sep=';')

    #df_N10_D05.boxplot(by='LABEL', column='DICE', grid=False)
    label_vec = ['WhiteMatter', 'GreyMatter', 'Hippocampus', 'Amygdala', 'Thalamus']

    # Define vectors for change in tree depth
    res_vec = [df_N10_D05, df_N10_D10, df_N10_D20, df_N10_D40, df_N10_D80]
    rf_depth_vec = [5, 10, 20, 40, 80]
    flag = 0
    #plotMetrics(res_vec, label_vec, rf_depth_vec, flag)

    # Define vectors for change in tree number
    res_vec_d = [df_D40_N01, df_D40_N05, df_D40_N10, df_D40_N20, df_D40_N50]
    rf_num_vec = [1, 5, 10, 20, 50]
    flag = 1
    #plotMetrics(res_vec_d, label_vec, rf_num_vec, flag)

    showBoxPlot(res_vec_d, label_vec)

if __name__ == '__main__':
    main()
