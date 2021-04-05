from pyspark.rdd import RDD
from pyspark.sql import Row
from pyspark.sql import DataFrame
from pyspark.sql.functions import desc, size, max, abs
from pyspark.sql import SparkSession


# Initialize a spark session.
def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark


def prepare_data():
    spark = init_spark()
    filename = "players_21.csv"

    # Read the csv files, this df will contain all  ~106 features
    df = spark.read.csv(filename, header=True)

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

    # Drop rows with any null values
    df = df.dropna()

    # Here we are filtering out goalkeepers because there are some features which are only specific to goalies, such as
    # "gk_reflexes" and "gk_speed" which have values for only goalies and is "null" for any other type of players, and
    # also features that are only null for goalies such as "pace", "shooting" and "dribbling". Therefore we will
    # remove goalkeepers from the dataset completely and won't try to predict goalkeepers
    df = df.filter(df.team_position != "GK")

    # The following features will be dropped because their values are not numbers, their form is "92+3", "85+2", etc.
    # And also because these would probably give away the positions way to easily so it doesn't make sense to test based
    # on these features.
    df = df.drop("ls", "st", "rs", "lw", "lf", "cf", "rf", "rw", "lam", "cam", "ram", "lm", "lcm", "cm", "rcm", "rm",
                 "lwb", "ldm", "cdm", "rdm", "rwb", "lb", "lcb", "cb", "rcb", "rb")

    # Finally remove the following unneeded features from the previously selected columns
    df = df.drop("short_name", "player_positions", "overall", "team_jersey_number", "movement_reactions",
                 "mentality_aggression", "mentality_composure")

    print("Number of columns: " + str(len(df.columns)))
    print("Number of rows: " + str(df.count()))
    # df.show(50)

    return df

    # todo: Try also by changing classes to 'attack', 'midfield', 'defence', 'gk', instead of 'st', 'rw', 'lw', 'cdm'...
    # todo: Print a chart showing how many of each classes there are (attack, midfield, defence, etc.)
    # todo: Normalize the ratings for each column?
