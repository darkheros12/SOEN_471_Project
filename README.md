# SOEN_471_Project

## Abstract:
This project is a dataset analysis on statistics related to football. More specifically, this dataset contains statistics on over 18 000 players such as dribbling, passing, shooting, defending, and stamina from data collected by sofifa.com. The data collected are stats spanning from the 2015 to 2021 season. The objective of this project is to apply supervised classification techniques to classify player's position based on their statistics and predict which positions they are most likely to play in. Some of the possible positions for players are goalkeeper, defender, midfielder and attacker.

***

## Introduction:
Football is one of the biggest entertainments in the world. With more than 18 000 professional players across the world, it has become very hard to find players that could be a fit in a football club. As a fanatic of football, we pick an Interest of player's performance on certain position. Players tend to perform better in certain position than others.  There are many positions in football such as full-back, centre-back, sweeper, goalkeeper, wing-back, defensive-midfield, central-midfield, attacking-midfield, forward, winger and striker. All those positions have their effect on the game. Assembling a team with different position is essential to make a more fluid gameplay and bring out the best experience for the players. There are multiple factors that might affect the game such as player’s experience, formations, strategies etc. However, for the scope of this project, we will solely focus on positions.

The main objective is to use the player’s stats from 2015 to 2021 to predict potential player's position based on trained stats. This will allow potential coaches or football adept to use players in their best position.

The current problem is coaches have difficult time to pick players based on their abilities. Often, they would put players in their unpreferred position because no one can fill those holes. During transfer session, it is often hard to find an adequate player to buy because of the uncertainty of their preferred position. Thus, this project is intended to solve the problem of future player based on previous trained models. It will predict if the player is fit for a certain position or not. 

There are a few studies using machine learning algorithms to predict game potential outcome such as “A Machine Learning framework for sport result prediction”. They analyzed different model of ML focusing on ANN (Artificial Neural Network) to predict sport results. Another study made by Hengzhi Chen “Neural Network Algorithm in Predicting Football Match Outcome Based on Player Ability Index” is based on player abilities to affect a football game. It will predict the outcome of the game based on player’s abilities. Plenty of previous studies showed that neural network is crucial for predictions in sports. 

***

## Materials and Methods:
The dataset that is chosen for this project includes all the player statistics for all the players from the popular soccer game Fifa. Those statistics include skills such as shooting, passing, dribbling, defending, and much more. In fact, the dataset contains about 100 columns for each player. A very interesting aspect of this dataset is that it is updated every year when a new Fifa game comes out and therefore the dataset includes player statistics from each Fifa game since the collection began with Fifa 15. Moreover, instead of appending new data to the existing dataset, each Fifa game has its own csv file containing the statistics from that year only, meaning we have data pertaining to Fifa 15, 16, 17 and on until Fifa 21, all in separate files. This is something that will facilitate for us to test on individual sets if we wish. Therefore, we can train on individual sets and compare those models with models trained on all the sets combined. Last thing to note about the chosen dataset is that some cleaning and feature engineering will be applied to it. Given the large number of features in the dataset as mentioned earlier, it is clear that some features give us irrelevant information about a player when the goal is to decide their position on the pitch. Therefore, the dataset will be optimized as much as possible before using it to train our models.

Multiple techniques will be used to achieve the objective. The algorithms that will be used to train our models are decision trees, random forest, and kNN (K-nearest neighbors). Decision tree is an algorithm where at each node we split our training set using the feature that gives us the best information gain. For example, when dealing with instances of two classes, we want instances from one class on one side of the split, and instances from another class on the other side, and so we choose the feature that splits these instances best. Random forests are multiple trees where some randomness is introduced so that we end up with a collection of trees that could predict an outcome. These trees then can “vote” for a class and the predicted class will be the one that is chosen by the most number of trees. KNN algorithm, on the other hand, classifies instances by grouping it with neighbors that it has the most in common with. For each algorithm, the hyperparameters will be tuned to find the most optimal model. For example, we will compare the different results obtained by decision tree model by changing the splitting criterion (gini vs. entropy), before comparing our decision tree model with the KNN model. Moreover, another technique that we are interested in is the naïve bayes classifier, which is not seen in this course. Naïve bayes algorithm classifies an instance by calculating a score for each possible class that the instance can belong to and chooses the class with the highest score. We expect all four of these models to have positives and negatives when faced with a dataset like ours. For example, we know that there are some very discriminative features, such as shooting skills, which differentiates strikers from the rest of the players, and therefore decision trees might give accurate classifications. KNN on the other hand might be useful since we know that players of the same position have a few attributes in common which they excel in. Therefore, it will be interesting to analyze each model carefully and determine what makes one model better than the others for our dataset.

Finally, most of our work will be done using Apache Spark and our data will be stored in a spark dataframe. While PyCharm will be our main IDE, Jupyter Notebook will also be used to help visualize our results better.

***

## Discussions

The purpose of this project is to determine the position of any players based on their statistic. As a secondary objective, we want to know which algorithms is better based on the dataset. The predicted position was narrow down to a few general positions for simpler classification. The positions are defender, midfielder, and attacker. As a result, we found out that most of our algorithms have a prediction accuracy of 82.78% and over regardless of the parameters. However, only the Naives Bayes has a 76.85% accuracy which is reasonable as the algorithms is very bias toward features independence assumption. The highest prediction accuracy is 86.75% by Random Forest Algorithm at 15 tries. Very close behind is k Nearest Neighbors with 10 neighbors, it has an accuracy of 86.73%. As the last place is Decision Tree with a max Depth of 10, it has an accuracy of 84.86%.

### Limitations

Some of the limitation was the preprocessing data. We had to filter most features.  




https://www.kaggle.com/stefanoleone992/fifa-21-complete-player-dataset
