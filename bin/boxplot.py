import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd

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

    df = pd.read_csv('../bin/mia-result/2021-10-10-14-14-12/results.csv', sep=';')
    print(df)

    df.boxplot(by='LABEL', column='DICE', grid=False)
    plt.show()



if __name__ == '__main__':
    main()
