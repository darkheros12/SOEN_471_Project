from pyspark.mllib.tree import DecisionTree
# todo: pyspark.ml better than pyspark.mllib?

'''
Parameters: 
df: The dataframe
num_of_classes: The total number of distinct classes. I.e there are 28 of st, lw, rw, cdm, etc distinct classes
'''
def decision_tree(df, num_of_classes):
    df = df.drop("preferred_foot")

    # data_rdd = df.rdd # ?

    (training_data, testing_data) = df.randomSplit([0.8, 0.2])  # Split the training and testing data

    # Note: The 'categoricalFeaturesInfo' parameter is a map to show the indexes that have categorical features (as
    # opposed to numerical) and how many categorical features
    # todo: Change to rdd of LabeldPoint
    model = DecisionTree.trainClassifier(training_data.rdd, numClasses=num_of_classes, categoricalFeaturesInfo={}, impurity='entropy', maxDepth=5, maxBins=32)

    # todo: Is maxDepth=5 too low? Apparently maximum value for maxDepth is 30 in spark implementation
    # todo: Try with gini instead of entropy and compare

    # Prediction happens here
    predictions = model.predict(testing_data.drop("team_position").rdd)

    # Zip the actual values and predicted values together for comparison
    labelsAndPredictions = testing_data.rdd.map(lambda row: row.team_position).zip(predictions)

    testErr = labelsAndPredictions.filter(lambda row: row[0] != row[1]).count() / float(testing_data.count())
    print('Test Error = ' + str(testErr))
    print('Learned classification tree model:')
    print(model.toDebugString())
