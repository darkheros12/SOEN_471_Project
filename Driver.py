import data_preparation
import decision_tree
import random_forest
import naive_bayes
import kNN

def main():
    df = data_preparation.prepare_data("players_21.csv")

    seed = 500

    dt_accuracy = decision_tree.decision_tree(df, seed)
    print("Decision Tree Accuracy = " + str(dt_accuracy))

    rf_accuracy = random_forest.random_forest(df, seed)
    print("Random Forest Accuracy = " + str(rf_accuracy))

    nb_accuracy = naive_bayes.naive_bayes(df, seed)
    print("Naive Bayes Accuracy = " + str(nb_accuracy))

    kNN_accuracy = kNN.k_nearest_neighbors(df, seed)
    print("K Nearest Neighbors Accuracy = " + str(kNN_accuracy))


main()
