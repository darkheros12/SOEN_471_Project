from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler
from sklearn.metrics import classification_report, confusion_matrix

'''
Parameters: 
df: The dataframe
'''
def random_forest(df, seed, num_of_trees):
    # Drop preferred_foot because it's the only categorical column, the others are all numerical
    # Use preferred_foot if we have time to implement it
    df = df.drop("preferred_foot")

    # Create a new column for the team_position label that is numerical instead of categorical
    labelIndexer = StringIndexer(inputCol="team_position", outputCol="indexed_label").fit(df)
    df = labelIndexer.transform(df)

    list_of_features = df.drop("team_position").drop("indexed_label").columns  # Get list of all features
    assembler = VectorAssembler(inputCols=list_of_features, outputCol="indexed_features")
    df = assembler.transform(df)

    (training_data, testing_data) = df.randomSplit([0.8, 0.2], seed)  # Split the training and testing data

    random_forest = RandomForestClassifier(labelCol="indexed_label", featuresCol="indexed_features", numTrees=num_of_trees)
    model = random_forest.fit(training_data)

    predictions = model.transform(testing_data)

    evaluator = MulticlassClassificationEvaluator(labelCol="indexed_label", predictionCol="prediction",
                                                  metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)

    y_true = predictions.select(['indexed_label']).collect()
    y_pred = predictions.select(['prediction']).collect()

    print("Classification report and confusion matrix for Random Forest with " + str(num_of_trees) + " trees:")
    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    confusion_matrix_corrected = [[cm[1][1], cm[1][2], cm[1][0]], [cm[2][1], cm[2][2], cm[2][0]],
                                  [cm[0][1], cm[0][2], cm[0][0]]]
    print("")
    print(confusion_matrix_corrected[0])
    print(confusion_matrix_corrected[1])
    print(confusion_matrix_corrected[2])

    return accuracy
