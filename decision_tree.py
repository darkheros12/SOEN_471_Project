from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler

'''
Parameters: 
df: The dataframe
'''
def decision_tree(df, seed):
    # Drop preferred_foot because it's the only categorical column, the others are all numerical
    # Use preferred_foot if we have time to implement it
    df = df.drop("preferred_foot")

    labelIndexer = StringIndexer(inputCol="team_position", outputCol="indexed_label").fit(df)
    df = labelIndexer.transform(df)

    list_of_features = df.drop("team_position").drop("indexed_label").columns  # Get list of all features
    assembler = VectorAssembler(inputCols=list_of_features, outputCol="indexed_features")
    df = assembler.transform(df)

    (training_data, testing_data) = df.randomSplit([0.8, 0.2], seed)  # Split the training and testing data

    d_tree = DecisionTreeClassifier(labelCol="indexed_label", featuresCol="indexed_features", impurity="entropy", maxDepth=5)
    model = d_tree.fit(training_data)

    # todo: Try with gini instead of entropy and compare

    # Prediction happens here
    predictions = model.transform(testing_data)

    evaluator = MulticlassClassificationEvaluator(labelCol="indexed_label", predictionCol="prediction",
                                                  metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    return accuracy
