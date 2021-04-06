import data_preparation
import decision_tree

def main():
    df = data_preparation.prepare_data("players_21.csv")
    decision_tree.decision_tree(df)


main()