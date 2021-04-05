import data_preparation
import decision_tree

def main():
    df = data_preparation.prepare_data()
    number_of_classes = df.select("team_position").distinct().count()
    # decision_tree.decision_tree(df, number_of_classes)


main()