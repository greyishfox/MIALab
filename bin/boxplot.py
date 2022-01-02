import matplotlib.pyplot as plt
import matplotlib.patches as pltpat
import numpy as np
import pandas as pd


def dataFrameToList(single_label_train, multi_label, label_names, metric):
    # Prepare label data for boxplot:
    # Convert pandas DataFrame to list (required for matplotlib as written below)

    # There is no post-processing implemented yet...therefore, do not take results from "-PP"-file into account
    multi_label = multi_label[~multi_label['SUBJECT'].str.endswith('-PP')]

    single_label_train_tmp = []
    for i in range(len(label_names)):
        single_label_train_tmp.append(single_label_train[i][~single_label_train[i]['SUBJECT'].str.endswith('-PP')])

    single_label_train = single_label_train_tmp

    # Multi-label list
    multi_label_list = []
    for i in range(len(label_names)):
        temp = pd.DataFrame(multi_label.loc[multi_label['LABEL'] == label_names[i]])
        multi_label_list.append(temp['DICE'].tolist())
        multi_label_list.append(temp['HDRFDST'].tolist())

    # Single-label list
    single_label_list = []
    data = []
    for i in range(len(label_names)):
        single_label_list.append(single_label_train[i]['DICE'].tolist())
        single_label_list.append(single_label_train[i]['HDRFDST'].tolist())

    # Create list of label lists (single-label and multi-label)
    if metric == 'DICE':
        data = [
            single_label_list[0], multi_label_list[0],
            single_label_list[2], multi_label_list[2],
            single_label_list[4], multi_label_list[4],
            single_label_list[6], multi_label_list[6],
            single_label_list[8], multi_label_list[8],
        ]
    elif metric == 'HDRFDST':
        data = [
            single_label_list[1], multi_label_list[1],
            single_label_list[3], multi_label_list[3],
            single_label_list[5], multi_label_list[5],
            single_label_list[7], multi_label_list[7],
            single_label_list[9], multi_label_list[9],
        ]
    else:
        print("Error: Unknown metric!")

    return data


# The following boxplot design was adapted from: https://matplotlib.org/stable/gallery/statistics/boxplot_demo.html
def advBoxPlot(data, metric, yLimit):
    # Calculate standard deviation
    std_values = []
    var_values = []
    for i in range(len(data)):
        std_values.append(np.std(data[i]))
        var_values.append(np.var(data[i]))

    # Define names on the x-axis
    xAxis_text = ['WhiteMatter\nSingle', 'WhiteMatter\nMulti', 'GreyMatter\nSingle', 'GreyMatter\nMulti',
                  'Hippocampus\nSingle', 'Hippocampus\nMulti', 'Amygdala\nSingle', 'Amygdala\nMulti',
                  'Thalamus\nSingle', 'Thalamus\nMulti']

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
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

    # Add title and label for xAxis & yAxis
    ax1.set(
        axisbelow=True,  # Hide the grid behind plot objects
        title='Comparison Of Label-Specific VS. Multi-Label Random Forest Classifiers',
        #xlabel='Labels',
        #ylabel=metric,
    )
    ax1.set_xlabel(xlabel='Labels', fontsize=10, fontweight="bold")
    ax1.set_ylabel(ylabel=metric, fontsize=10, fontweight="bold")

    # Now fill the boxes with desired colors
    box_colors = ['lightblue', 'gold']
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
        # Alternate between skyblue and orange
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
    top = yLimit[0]
    bottom = yLimit[1]
    ax1.set_ylim(bottom, top)
    ax1.set_xticklabels(xAxis_text, rotation=60, fontsize=10)

    # Due to the Y-axis scale being different across samples, it can be
    # hard to compare differences in medians across the samples. Add upper
    # X-axis tick labels with the sample medians to aid in comparison
    # (just use two decimal places of precision)
    pos = np.arange(num_boxes) + 1
    upper_labels = [str(round(s, 2)) for s in medians]

    ax1.text(pos[0], .96, 'Median:', transform=ax1.get_xaxis_transform(), horizontalalignment='center', size='x-small',
                 weight='bold', color='black')

    for tick, label in zip(range(num_boxes), ax1.get_xticklabels()):
        k = tick % 2
        ax1.text(pos[tick], .93, upper_labels[tick],
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', size='x-small',
                 weight='bold', color='black')  # color=box_colors[k])

    # STD and VAR
    #pos = np.arange(num_boxes) + 1
    upper_labels = [str(round(s, 3)) for s in std_values]

    ax1.text(pos[0], .90, 'STD:', transform=ax1.get_xaxis_transform(), horizontalalignment='center', size='x-small',
                 weight='bold', color='black')

    for tick, label in zip(range(num_boxes), ax1.get_xticklabels()):
        k = tick % 2
        ax1.text(pos[tick], .87, upper_labels[tick],
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', size='x-small',
                 weight='bold', color='black')  # color=box_colors[k])

    # Finally, add a basic legend
    if metric == 'DICE':
        fig.text(0.82, 0.350, 'Single-Labels', backgroundcolor=box_colors[0], color='black', weight='roman', size='x-small')
        fig.text(0.82, 0.325, 'Multi-Labels  ', backgroundcolor=box_colors[1], color='black', weight='roman', size='x-small')
        fig.text(0.82, 0.297, 'o', color='black', weight='normal', size='medium')
        fig.text(0.838, 0.297, 'Mean Value', color='black', weight='roman', size='x-small')
    elif metric == 'HDRFDST':
        fig.text(0.84, 0.320, 'Single-Labels', backgroundcolor=box_colors[0], color='black', weight='roman', size='x-small')
        fig.text(0.84, 0.295, 'Multi-Labels  ', backgroundcolor=box_colors[1], color='black', weight='roman', size='x-small')
        fig.text(0.84, 0.267, 'o', color='black', weight='normal', size='medium')
        fig.text(0.858, 0.267, 'Mean Value', color='black', weight='roman', size='x-small')
    else:
        print("Error: Unknown metric!")

    plt.show()


def main():

    # Single-labels (1=WiteMatter, 2=GreyMatter, 3=Hippocampus, 4=Amygdala, 5=Thalamus)
    sigl_lbl_1 = pd.read_csv('run_1/TreeD-040-TreeN-020-Label-WhiteMatter/results.csv', sep=';')
    sigl_lbl_2 = pd.read_csv('run_1/TreeD-040-TreeN-020-Label-GreyMatter/results.csv', sep=';')
    sigl_lbl_3 = pd.read_csv('run_1/TreeD-040-TreeN-020-Label-Hippocampus/results.csv', sep=';')
    sigl_lbl_4 = pd.read_csv('run_1/TreeD-040-TreeN-020-Label-Amygdala/results.csv', sep=';')
    sigl_lbl_5 = pd.read_csv('run_1/TreeD-040-TreeN-020-Label-Thalamus/results.csv', sep=';')

    # Multi-label
    mult_lbl = pd.read_csv('run_1/TreeD-040-TreeN-020-Label-all/results.csv', sep=';')


    # Create label vector
    label_train = ['WhiteMatter', 'GreyMatter', 'Hippocampus', 'Amygdala', 'Thalamus']
    # Create metric vector
    metricVec = ['DICE', 'HDRFDST']

    # Define ylimits for DICE and HDRFDST -> [DICE_top, DICE_bottom, HDRFDST_top, HDRFDST_bottom]
    yLimit_DICE = [1.1, 0.0]
    yLimit_HDRFDST = [20.0, 0.0]


    # Create single-label vector
    sigl_lbl_train = [sigl_lbl_1, sigl_lbl_2, sigl_lbl_3, sigl_lbl_4, sigl_lbl_5]

    # Convert dataFrame to list
    DICE_data = dataFrameToList(sigl_lbl_train, mult_lbl, label_train, metricVec[0])
    HDRFDST_data = dataFrameToList(sigl_lbl_train, mult_lbl, label_train, metricVec[1])

    # Run boxplot method
    advBoxPlot(DICE_data, metricVec[0], yLimit_DICE)
    advBoxPlot(HDRFDST_data, metricVec[1], yLimit_HDRFDST)


if __name__ == '__main__':
    main()
