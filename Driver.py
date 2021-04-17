import data_preparation
import decision_tree
import random_forest
import naive_bayes
import kNN
import numpy as np
import matplotlib.pyplot as plt

def main():
    df = data_preparation.prepare_data(["Data/players_15.csv", "Data/players_16.csv", "Data/players_17.csv",
                                        "Data/players_18.csv", "Data/players_19.csv", "Data/players_20.csv",
                                        "Data/players_21.csv"])

    seed = 10

    max_depth_list = [5, 7, 9, 10, 12, 14]
    dt_accuracy_list = decision_tree.decision_tree(df, seed, max_depth_list=max_depth_list)
    plot_bar_accuracy("Decision Tree Accuracy", "Depth", "Accuracy", dt_accuracy_list, max_depth_list)

    '''num_of_trees_list = [10, 15, 20, 25, 30, 35]
    rf_accuracy_list = random_forest.random_forest(df, seed, num_of_trees_list=num_of_trees_list)
    plot_bar_accuracy("Random Forest Accuracy", "Trees", "Accuracy", rf_accuracy_list, num_of_trees_list)'''

    '''nb_accuracy = naive_bayes.naive_bayes(df, seed)
    plot_bar_accuracy("Naive Bayes Accuracy", "Smoothing", "Accuracy", [0, 0, nb_accuracy, 0, 0], ["", "", 1.0, "", ""])'''

    '''neighbors_list = [10, 15, 25, 150, 210, 250, 300]
    kNN_accuracy_list = kNN.k_nearest_neighbors(df, seed, neighbors_list=neighbors_list)
    plot_bar_accuracy("K Nearest Neighbors Accuracy", "Neighbors", "Accuracy", kNN_accuracy_list, neighbors_list)'''

    print("")


# This function will create a bar plot for each list given to it
def plot_bar_accuracy(title, x_label, y_label, data, label):
    # this is for plotting purpose
    index = np.arange(len(label))
    plt.figure(figsize=(20,10))
    plt.bar(index, data)
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    plt.xticks(index, label, fontsize=20, rotation=30)
    plt.title(title, fontsize=20)
    for index, value in enumerate(data):
        plt.text(index, value, "{0:.4f}".format(value), fontsize=10)
    plt.show()


main()
