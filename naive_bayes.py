from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

'''
Parameters: 
df: The dataframe
seed: Used for randomSplit
'''
def naive_bayes(df, seed):
    # Drop preferred_foot because it's the only categorical column, the others are all numerical
    # Use preferred_foot if we have time to implement it
    df = df.drop("preferred_foot")

    labelIndexer = StringIndexer(inputCol="team_position", outputCol="label").fit(df)
    df = labelIndexer.transform(df)
    df = df.drop("team_position")

    list_of_features = df.drop("label").columns  # Get list of all features
    assembler = VectorAssembler(inputCols=list_of_features, outputCol="features")
    df = assembler.transform(df)

    (train_data, test_data) = df.randomSplit([0.8, 0.2], seed)

    n_bayes = NaiveBayes(smoothing=1.0, modelType="multinomial")

    model = n_bayes.fit(train_data)  # Training happens here

    predictions = model.transform(test_data)

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                                  metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)

    y_true = predictions.select(['label']).collect()
    y_pred = predictions.select(['prediction']).collect()

    print("Classification report and confusion matrix for Naive Bayes:")
    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    confusion_matrix_corrected = [[cm[1][1], cm[1][2], cm[1][0]], [cm[2][1], cm[2][2], cm[2][0]],
                                  [cm[0][1], cm[0][2], cm[0][0]]]
    print("")
    print(confusion_matrix_corrected[0])
    print(confusion_matrix_corrected[1])
    print(confusion_matrix_corrected[2])

    cm = np.array([confusion_matrix_corrected[0], confusion_matrix_corrected[1], confusion_matrix_corrected[2]])

    return accuracy, cm
