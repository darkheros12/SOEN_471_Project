from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

'''
Parameters: 
df: The dataframe
'''
def k_nearest_neighbors(df, seed, neighbors):
    # Drop preferred_foot because it's the only categorical column, the others are all numerical
    # Use preferred_foot if we have time to implement it
    df = df.drop("preferred_foot")

    (train_data, test_data) = df.randomSplit([0.8, 0.2], seed)
    x_train = train_data.drop("team_position").collect()
    y_train = train_data.select("team_position").rdd.map(lambda row: row[0]).collect()

    x_test = test_data.drop("team_position").collect()
    y_test = test_data.select("team_position").rdd.map(lambda row: row[0]).collect()

    neighbors = KNeighborsClassifier(n_neighbors=neighbors)
    neighbors.fit(x_train, y_train)
    prediction_labels = neighbors.predict(x_test)
    accuracy = neighbors.score(x_test, y_test)
    print("Classification report and confusion matrix for kNN with " + str(neighbors) + " neighbors:")
    print(classification_report(y_test, prediction_labels))
    cm = confusion_matrix(y_test, prediction_labels)
    # print(cm)

    confusion_matrix_corrected = [[cm[1][1], cm[1][2], cm[1][0]], [cm[2][1], cm[2][2], cm[2][0]], [cm[0][1], cm[0][2], cm[0][0]]]
    print("")
    print(confusion_matrix_corrected[0])
    print(confusion_matrix_corrected[1])
    print(confusion_matrix_corrected[2])

    return accuracy
