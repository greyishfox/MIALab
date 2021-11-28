import matplotlib.pyplot as plt
import matplotlib.patches as pltpat
import numpy as np
import pandas as pd


# The following boxplot design was adapted from: https://matplotlib.org/stable/gallery/statistics/boxplot_demo.html
def advBoxPlot(single_label_train, multi_label, label_names):
    # Prepare label data for boxplot:
    # Convert pandas DataFrame to list (required for matplotlib as written below)

    # Multi-label list
    multi_label_list = []
    for i in range(len(label_names)):
        temp = pd.DataFrame(multi_label.loc[multi_label['LABEL'] == label_names[i]])
        multi_label_list.append(temp['DICE'].tolist())

    # Single-label list
    single_label_list = []
    for i in range(len(label_names)):
        single_label_list.append(single_label_train[i]['DICE'].tolist())

    # Define names on the x-axis
    xAxis_text = ['WhiteMatter_Single', 'WhiteMatter_Multi', 'GreyMatter_Single', 'GreyMatter_Multi',
                  'Hippocampus_Single', 'Hippocampus_Multi', 'Amygdala_Single', 'Amygdala_Multi',
                  'Thalamus_Single', 'Thalamus_Multi']

    # Create list of label lists (single-label and multi-label)
    data = [
        single_label_list[0], multi_label_list[0],
        single_label_list[1], multi_label_list[1],
        single_label_list[2], multi_label_list[2],
        single_label_list[3], multi_label_list[3],
        single_label_list[4], multi_label_list[4],
    ]

    # Define figure characteristics
    fig, ax1 = plt.subplots(figsize=(8, 6))
    fig.canvas.manager.set_window_title('Boxplot of Single-label vs. Multi-label')
    fig.subplots_adjust(left=0.08, right=0.95, top=0.9, bottom=0.25)

    # Define boxplot design
    bp = ax1.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')

    # Add horizontal grid
    ax1.yaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.5)

    # Add title and label for xAxis & yAxis
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
    top = 1.0
    bottom = 0.0
    ax1.set_ylim(bottom, top)
    ax1.set_xticklabels(xAxis_text, rotation=45, fontsize=8)

    # Due to the Y-axis scale being different across samples, it can be
    # hard to compare differences in medians across the samples. Add upper
    # X-axis tick labels with the sample medians to aid in comparison
    # (just use two decimal places of precision)
    pos = np.arange(num_boxes) + 1
    upper_labels = [str(round(s, 2)) for s in medians]

    for tick, label in zip(range(num_boxes), ax1.get_xticklabels()):
        k = tick % 2
        ax1.text(pos[tick], .95, upper_labels[tick],
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', size='x-small',
                 weight='bold', color=box_colors[k])

    # Finally, add a basic legend
    fig.text(0.82, 0.350, 'Single-Labels', backgroundcolor=box_colors[0], color='black', weight='roman', size='x-small')
    fig.text(0.82, 0.325, 'Multi-Labels  ', backgroundcolor=box_colors[1], color='black', weight='roman', size='x-small')
    fig.text(0.82, 0.297, 'o', color='black', weight='normal', size='medium')
    fig.text(0.838, 0.297, 'Mean Value', color='black', weight='roman', size='x-small')

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
    advBoxPlot(sigl_lbl_train, mult_lbl, label_train)


if __name__ == '__main__':
    main()
