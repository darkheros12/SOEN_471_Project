from pyspark.rdd import RDD
from pyspark.sql import Row
from pyspark.sql import DataFrame
from pyspark.sql.functions import desc, size, max, abs, udf, col
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, StringType
from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import numpy as np


# Initialize a spark session.
def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark


def prepare_data(files):
    spark = init_spark()

    # Read the csv files, this df will contain all  ~106 features
    df = None
    for file in files:
        if df is None:
            df = spark.read.csv(file, header=True)
            print(df.count())
        else:
            df = spark.read.csv(file, header=True).union(df)
            print(df.count())

    # Filtering the data (as seen below) will be done in multiple steps for better readability and understandability

    # Select a bunch of important features (this will not yet be the final df that will be used to train the models)
    df = df.select("short_name", "overall", "player_positions", "preferred_foot", "skill_moves", "team_position",
                   "team_jersey_number", "pace", "shooting", "passing", "dribbling", "defending", "physic",
                   "attacking_crossing", "attacking_finishing", "attacking_heading_accuracy", "attacking_short_passing",
                   "attacking_volleys", "skill_dribbling", "skill_curve", "skill_fk_accuracy", "skill_long_passing",
                   "skill_ball_control", "movement_acceleration", "movement_sprint_speed", "movement_agility",
                   "movement_reactions", "movement_balance", "power_shot_power", "power_jumping", "power_stamina",
                   "power_strength", "power_long_shots", "mentality_aggression", "mentality_interceptions",
                   "mentality_positioning", "mentality_vision", "mentality_penalties", "mentality_composure",
                   "defending_standing_tackle", "defending_sliding_tackle", "ls", "st", "rs", "lw", "lf", "cf", "rf",
                   "rw", "lam", "cam", "ram", "lm", "lcm", "cm", "rcm", "rm", "lwb", "ldm", "cdm", "rdm", "rwb", "lb",
                   "lcb", "cb", "rcb", "rb")

    # Here we are filtering out goalkeepers because there are some features which are only specific to goalies, such as
    # "gk_reflexes" and "gk_speed" which have values for only goalies and is "null" for any other type of players, and
    # also features that are only null for goalies such as "pace", "shooting" and "dribbling". Therefore we will
    # remove goalkeepers from the dataset completely and won't try to predict goalkeepers
    df = df.filter(df.team_position != "GK")

    # Drop rows where the player's position is SUB or RES (NOTE: this takes out like 60% of the data.....)
    df = df.filter(df.team_position != "SUB")
    df = df.filter(df.team_position != "RES")

    # The following features will be dropped because their values are not numbers, their form is "92+3", "85+2", etc.
    # And also because these would probably give away the positions way to easily so it doesn't make sense to test based
    # on these features.
    df = df.drop("ls", "st", "rs", "lw", "lf", "cf", "rf", "rw", "lam", "cam", "ram", "lm", "lcm", "cm", "rcm", "rm",
                 "lwb", "ldm", "cdm", "rdm", "rwb", "lb", "lcb", "cb", "rcb", "rb")

    # The following are dropped because it's easy to see that these one are not very discriminative. Players in
    # different positions could have same values for these attributes. This is part of features cleanup (less features =
    # better prediction hopefully)
    df = df.drop("pace", "physic", "skill_curve", "skill_ball_control", "movement_acceleration", "movement_sprint_speed",
                 "movement_balance", "power_jumping", "power_stamina", "power_strength", "power_long_shots")

    # Finally remove the following unneeded features from the previously selected columns
    df = df.drop("short_name", "player_positions", "team_jersey_number", "movement_reactions",
                 "mentality_aggression", "mentality_composure")

    # Drop rows with any null values, we don't want sparse matrix
    df = df.dropna()

    # Here we turn all values to IntegerType because for some reason they are represented as string even though they are
    # numerical (except team_position which is the label/class, and preferred_foot because it's either 'right' or 'left'
    list_of_features = df.drop("team_position").drop("preferred_foot").columns  # Get list of all features that should be numerical
    for feature in list_of_features:
        df = df.withColumn(feature, df[feature].cast(IntegerType()))  # Replace that column with int version of that col

    # Normalizing is being done below
    for feature in list_of_features:
        if feature != "overall":
            df = df.withColumn(feature, df[feature] / df["overall"] * 100.0)  # Normalize using their overall score
    df = df.drop("overall")

    # Change the labels from values such as st, lw, rw, cdm, lm, lb, lwb, cb, to forward, midfielder, and defender
    labels_dict = {'ST': 'Forward', 'CF': 'Forward', 'LF': 'Forward', 'RF': 'Forward', 'LS': 'Forward', 'RS': 'Forward',
                   'LW': 'Forward', 'RW': 'Forward', 'CAM': 'Midfielder', 'LAM': 'Midfielder', 'RAM': 'Midfielder',
                   'CM': 'Midfielder', 'LCM': 'Midfielder', 'RCM': 'Midfielder', 'LM': 'Midfielder', 'RM': 'Midfielder',
                   'CDM': 'Midfielder', 'LDM': 'Midfielder', 'RDM': 'Midfielder', 'CB': 'Defender', 'LCB': 'Defender',
                   'RCB': 'Defender', 'LB': 'Defender', 'RB': 'Defender', 'LWB': 'Defender', 'RWB': 'Defender'}

    def get_label(position):
        return labels_dict[position]

    udf_column = udf(get_label, StringType())
    df = df.withColumn("team_position", udf_column('team_position'))  # Change values to basic labels

    print("Number of columns: " + str(len(df.columns)))
    print("Number of rows: " + str(df.count()))
    print("")

    # printClassCount(df)

    # ----------------------------------------------- Apply SMOTE ------------------------------------------------------
    sm = SMOTE(random_state=42)

    list_of_features = df.drop("team_position").drop("preferred_foot").columns
    X_features = df.drop("team_position").drop("preferred_foot").collect()
    Y_label = df.select("team_position").collect()

    X_sm, y_sm = sm.fit_resample(X_features, Y_label)

    i = 0
    for row in X_sm:
        row.append(y_sm[i])
        i += 1

    list_of_features.append("team_position")
    df = spark.createDataFrame(data=X_sm, schema=list_of_features)

    # ------------------------------------------------------------------------------------------------------------------

    # printClassCount(df)

    # df.show(50)

    return df

    # todo: Try also by changing classes to 'attack', 'midfield', 'defence', 'gk', instead of 'st', 'rw', 'lw', 'cdm'...

def printClassCount(df):
    forward = df.filter(df.team_position == "Forward").count()
    print("Number of forwards: " + str(forward))
    midfielder = df.filter(df.team_position == "Midfielder").count()
    print("Number of midfielders: " + str(midfielder))
    defender = df.filter(df.team_position == "Defender").count()
    print("Number of defenders: " + str(defender))

    total = df.count()
    print("Total number of entries: " + str(total))

    mylabel = ["Forward", "Defender", "Midfielder"]

    y = np.array([(forward / total) * 100, (defender / total) * 100, (midfielder / total) * 100])
    plt.pie(y, labels=mylabel, autopct='%1.2f')
    plt.show()
