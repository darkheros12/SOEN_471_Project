import data_preparation
import decision_tree
import random_forest

def main():
    df = data_preparation.prepare_data("players_21.csv")

    dt_accuracy = decision_tree.decision_tree(df)
    print("Decision Tree Accuracy = " + str(dt_accuracy))

    rf_accuracy = random_forest.random_forest(df)
    print("Random Forest Accuracy = " + str(dt_accuracy))


main()
