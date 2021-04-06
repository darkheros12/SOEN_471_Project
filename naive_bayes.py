from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler

'''
Parameters: 
df: The dataframe
'''
def naive_bayes(df):
    # Drop preferred_foot because it's the only categorical column, the others are all numerical
    # Use preferred_foot if we have time to implement it
    df = df.drop("preferred_foot")

    labelIndexer = StringIndexer(inputCol="team_position", outputCol="label").fit(df)
    df = labelIndexer.transform(df)
    df = df.drop("team_position")

    list_of_features = df.drop("label").columns  # Get list of all features
    assembler = VectorAssembler(inputCols=list_of_features, outputCol="features")
    df = assembler.transform(df)

    splits = df.randomSplit([0.8, 0.2])
    train_data = splits[0]
    test_data = splits[1]

    n_bayes = NaiveBayes(smoothing=1.0, modelType="multinomial")

    model = n_bayes.fit(train_data)  # Training happens here

    predictions = model.transform(test_data)

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                                  metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)

    return accuracy
