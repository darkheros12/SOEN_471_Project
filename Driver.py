import data_preparation
import decision_tree
import random_forest
import naive_bayes
import kNN
import numpy as np
import matplotlib.pyplot as plt

def main():
    df = data_preparation.prepare_data(["players_15.csv", "players_16.csv", "players_17.csv", "players_18.csv",
                                        "players_19.csv", "players_20.csv", "players_21.csv"])

    seed = 10

    dt_accuracy1 = decision_tree.decision_tree(df, seed, max_depth=5)
    dt_accuracy2 = decision_tree.decision_tree(df, seed, max_depth=25)
    dt_accuracy3 = decision_tree.decision_tree(df, seed, max_depth=30)
    # ("Decision Tree Accuracy = " + str(dt_accuracy))
    plot_bar_accuracy("Decision Tree Accuracy", "Depth", "Accuracy", [dt_accuracy1, dt_accuracy2, dt_accuracy3], [5, 25, 30])

    rf_accuracy = random_forest.random_forest(df, seed, num_of_trees=10)
    print("Random Forest Accuracy = " + str(rf_accuracy))

    nb_accuracy = naive_bayes.naive_bayes(df, seed)
    print("Naive Bayes Accuracy = " + str(nb_accuracy))

    kNN_accuracy = kNN.k_nearest_neighbors(df, seed, neighbors=50)
    print("K Nearest Neighbors Accuracy = " + str(kNN_accuracy))

#This function will create a bar plot for each list given to it
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
        plt.text(index, value, str(value), fontsize=15)
    plt.show()


main()
