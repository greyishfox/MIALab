import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
import os

def showBoxPlot(s_lbl_train, m_lbl, label_vec):
    sgl_res_table = []

    for i in range(len(label_vec)):
        res_subset = s_lbl_train[i].loc[s_lbl_train[i]['LABEL'] == label_vec[i]]
        sgl_res_table.append(pd.DataFrame(res_subset, columns=['SUBJECT', 'LABEL', 'DICE', 'HDRFDST']))

    for i in range(len(label_vec)):
        sgl_res_table[i].boxplot(by='LABEL', column='DICE', figsize=(10, 10))
        plt.ylim(0, 1)

    plt.show()


def main():

    # Single-labels (1=WiteMatter, 2=GreyMatter, 3=Hippocampus, 4=Amygdala, 5=Thalamus)
    sigl_lbl_1 = pd.read_csv('run1/TreeD-040-TreeN-020-Label-1/results.csv', sep=';')
    sigl_lbl_2 = pd.read_csv('run1/TreeD-040-TreeN-020-Label-2/results.csv', sep=';')
    sigl_lbl_3 = pd.read_csv('run1/TreeD-040-TreeN-020-Label-3/results.csv', sep=';')
    sigl_lbl_4 = pd.read_csv('run1/TreeD-040-TreeN-020-Label-4/results.csv', sep=';')
    sigl_lbl_5 = pd.read_csv('run1/TreeD-040-TreeN-020-Label-5/results.csv', sep=';')

    # Multi-label
    mult_lbl = pd.read_csv('mia-result/TreeD-40-TreeN-20/results.csv', sep=';')


    # Create label vector
    label_train = ['WhiteMatter', 'GreyMatter', 'Hippocampus', 'Amygdala', 'Thalamus']

    # Create single-label vector
    sigl_lbl_train = [sigl_lbl_1, sigl_lbl_2, sigl_lbl_3, sigl_lbl_4, sigl_lbl_5]

    # run box plot method
    showBoxPlot(sigl_lbl_train, mult_lbl, label_train)


if __name__ == '__main__':
    main()
