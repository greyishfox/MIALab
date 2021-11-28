import matplotlib.pyplot as plt
import matplotlib.patches as pltpat
import numpy as np
import csv
import pandas as pd
import os
import seaborn as sns

def advBoxPlot(single_label_train, multi_label, label_names):
    # Prepare label data for boxplot
    wm_multi_lst = pd.DataFrame(multi_label.loc[multi_label['LABEL'] == label_names[0]])

    wm_multi_lst = wm_multi_lst['DICE'].tolist()
    print(wm_multi_lst)





    # Define names on the x-axis
    xAxis_text = ['WhiteMatter_Single', 'WhiteMatter_Multi', 'GreyMatter_Single', 'GreyMatter_Multi',
                  'Hippocampus_Single', 'Hippocampus_Multi', 'Amygdala_Single', 'Amygdala_Multi',
                  'Thalamus_Single', 'Thalamus_Multi']
    N = 500

    norm = np.random.normal(1, 1, N)
    logn = np.random.lognormal(1, 1, N)
    expo = np.random.exponential(1, N)
    gumb = np.random.gumbel(6, 4, N)
    tria = np.random.triangular(2, 9, 11, N)

    # Generate some random indices that we'll use to resample the original data
    # arrays. For code brevity, just use the same random indices for each array
    bootstrap_indices = np.random.randint(0, N, N)
    data = [
        norm, wm_multi_lst,
        logn, logn[bootstrap_indices],
        expo, expo[bootstrap_indices],
        gumb, gumb[bootstrap_indices],
        tria, tria[bootstrap_indices],
    ]

    print('Number of rows: ' + str(len(data[0])))
    print('Number of cols: ' + str(len(data)))

    fig, ax1 = plt.subplots(figsize=(8, 8))
    fig.canvas.manager.set_window_title('Boxplot of Single-label vs. Multi-label')
    fig.subplots_adjust(left=0.08, right=0.95, top=0.9, bottom=0.25)
    # fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    bp = ax1.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')

    # Add horizontal grid
    ax1.yaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.5)

    ax1.set(
        axisbelow=True,  # Hide the grid behind plot objects
        title='Comparison Of Label-Specific VS. Multi-Label Random Forest Classifiers',
        xlabel='Labels',
        ylabel='DICE',
    )

    # Now fill the boxes with desired colors
    box_colors = ['skyblue', 'orange']
    num_boxes = len(data)
    medians = np.empty(num_boxes)
    for i in range(num_boxes):
        box = bp['boxes'][i]
        box_x = []
        box_y = []
        for j in range(5):
            box_x.append(box.get_xdata()[j])
            box_y.append(box.get_ydata()[j])
        box_coords = np.column_stack([box_x, box_y])
        # Alternate between Dark Khaki and Royal Blue
        ax1.add_patch(pltpat.Polygon(box_coords, facecolor=box_colors[i % 2]))
        # Now draw the median lines back over what we just filled in
        med = bp['medians'][i]
        median_x = []
        median_y = []
        for j in range(2):
            median_x.append(med.get_xdata()[j])
            median_y.append(med.get_ydata()[j])
            ax1.plot(median_x, median_y, 'k')
        medians[i] = median_y[0]
        # Finally, overplot the sample averages, with horizontal alignment
        # in the center of each box
        ax1.plot(np.average(med.get_xdata()), np.average(data[i]),
                 color='w', marker='o', markeredgecolor='k')

    # Set the axes ranges and axes labels
    ax1.set_xlim(0.5, num_boxes + 0.5)
    top = 40
    bottom = -5
    ax1.set_ylim(bottom, top)
    ax1.set_xticklabels(xAxis_text, rotation=45, fontsize=8)

    # Due to the Y-axis scale being different across samples, it can be
    # hard to compare differences in medians across the samples. Add upper
    # X-axis tick labels with the sample medians to aid in comparison
    # (just use two decimal places of precision)
    pos = np.arange(num_boxes) + 1
    upper_labels = [str(round(s, 2)) for s in medians]
    weights = ['bold', 'bold']
    for tick, label in zip(range(num_boxes), ax1.get_xticklabels()):
        k = tick % 2
        ax1.text(pos[tick], .95, upper_labels[tick],
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', size='x-small',
                 weight=weights[k], color=box_colors[k])

    # Finally, add a basic legend
    fig.text(0.80, 0.08, 'Single-Labels', backgroundcolor=box_colors[0], color='black', weight='roman', size='x-small')
    fig.text(0.80, 0.045, 'Multi-Labels  ', backgroundcolor=box_colors[1], color='black', weight='roman', size='x-small')
    fig.text(0.80, 0.013, 'o', color='black', weight='normal', size='medium')
    fig.text(0.818, 0.013, 'Mean Value', color='black', weight='roman', size='x-small')

    plt.show()




def showBoxPlot(s_lbl_train, m_lbl, label_vec):
    # Add additional column 'TYPE' to both DataFrames
    label_type = ['MULTI', 'SINGLE']

    m_lbl['TYPE'] = label_type[0]

    for i in range(len(s_lbl_train)):
        s_lbl_train[i]['TYPE'] = label_type[1]

    single_labels = pd.DataFrame(s_lbl_train[0],
                                 columns=['SUBJECT', 'LABEL', 'DICE', 'HDRFDST', 'TYPE']).assign(Location=1)

    multi_label = pd.DataFrame(m_lbl[m_lbl['LABEL'] == label_vec[0]],
                               columns=['SUBJECT', 'LABEL', 'DICE', 'HDRFDST', 'TYPE']).assign(Location=2)

    # for i in range(len(label_vec)):
    #     single_labels[i].boxplot(by='LABEL', column='DICE', figsize=(10, 10))
    #     plt.ylim(0, 1)

    # multi_label.boxplot(by='LABEL', column='DICE', figsize=(10, 10))

    mult_sgl_concat = pd.DataFrame(pd.concat([single_labels, multi_label]))
    mult_sgl_concat.boxplot(by='TYPE', column='DICE')
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
    #showBoxPlot(sigl_lbl_train, mult_lbl, label_train)

    advBoxPlot(sigl_lbl_train, mult_lbl, label_train)


if __name__ == '__main__':
    main()
