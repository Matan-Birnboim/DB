"""
#**Load data**
"""

import findspark
from pyspark.sql.types import StructField, StructType, StringType, DoubleType, IntegerType, DateType, ArrayType

findspark.init()
from pyspark.sql import SparkSession
from pyspark.sql import functions as f


def init_spark(app_name: str):
    spark = SparkSession.builder.appName(app_name).getOrCreate()
    sc = spark.sparkContext
    return spark, sc


def load_users(dataPath):
    users_schema = StructType([
        StructField('user_id', IntegerType(), True),
        StructField('user_location', StringType(), True)
    ])

    users = spark.read.format("csv") \
        .option("header", "true") \
        .schema(users_schema) \
        .load(dataPath)

    return users


def load_tickets(dataPath):
    tickets_schema = StructType([
        StructField('user_id', IntegerType(), True),
        StructField('movie_id', IntegerType(), True),
        StructField('number_of_tickets', IntegerType(), True),
        StructField('city', StringType(), True),
        StructField('cinema_id', IntegerType(), True)
    ])

    tickets = spark.read.format("csv") \
        .option("header", "true") \
        .schema(tickets_schema) \
        .load(dataPath)

    return tickets


def load_credits(dataPath):  # Final script after review load credits data

    credits_schema = StructType([
        StructField('cast', StringType(), True),
        StructField('crew', StringType(), True),
        StructField('id', IntegerType(), True)
    ])

    credits = spark.read.option("Header", True) \
        .option("multiline", True) \
        .option("escape", "\"") \
        .csv(dataPath)
    # (option("quote", "\"")) line above solved the missinterputation of the char " by pySpark

    # --- Phase 1: All cells are Strings ---

    fix_cols = f.udf(lambda x: x.replace('None', '\'None\''))

    credits = credits \
        .withColumn("cast", fix_cols('cast')) \
        .withColumn("crew", fix_cols('crew'))

    # --- Phase 2: Semi Structured to Structured ---
    schema_cast = ArrayType(
        StructType([StructField("cast_id", StringType(), True),
                    StructField("character", StringType(), True),
                    StructField("credit_id", StringType(), True),
                    StructField("gender", StringType(), True),
                    StructField("id", StringType(), True),
                    StructField("name", StringType(), True),
                    StructField("order", StringType(), True),
                    StructField("profile_path", StringType(), True)]))

    schema_crew = ArrayType(
        StructType([StructField("credit_id", StringType(), True),
                    StructField("department", StringType(), True),
                    StructField("gender", StringType(), True),
                    StructField("id", StringType(), True),
                    StructField("job", StringType(), True),
                    StructField("name", StringType(), True),
                    StructField("profile_path", StringType(), True)]))

    # changes the columns containing lists of json to structured cells.
    credits = credits.withColumn("cast", f.from_json(credits.cast, schema_cast)) \
        .withColumn("crew", f.from_json(credits.crew, schema_crew))

    # --- Phase 3: Extract feilds based on the query data ---
    extract_name = f.udf(lambda x: ",".join([elem['name'] for elem in x]) if x is not None else "-", StringType())

    # Extract the director's 'name' only
    extract_director_name = f.udf(
        lambda x: ",".join([elem['name'] for elem in x if elem['job'] == 'Director']) if x is not None else "-",
        StringType())

    credits = credits \
        .withColumn("cast", extract_name('cast')) \
        .withColumn("crew", extract_director_name('crew'))

    # --- Phase 4: Convert Strings representing Arrays to ArrayType(StringType())
    credits = credits \
        .withColumn("cast", f.split('cast', ',')) \
        .withColumn("crew", f.split('crew', ','))

    return credits


def load_queries(dataPath):  # Final script after review load queries data
    queries_schema = StructType([
        StructField('user_id', IntegerType(), True),
        StructField('genres', StringType(), True),
        StructField('lang', StringType(), True),
        StructField('actors', StringType(), True),
        StructField('director', StringType(), True),
        StructField('cities', StringType(), True),
        StructField('country', StringType(), True),
        StructField('from_realese_date', StringType(), True),
        StructField('production_company', StringType(), True),
    ])
    queries = spark.read.option("Header", True) \
        .option("multiline", True) \
        .option("escape", "\"") \
        .csv(dataPath)
    # (option("quote", "\"")) line above solved the missinterputation of the char " by pySpark

    # --- Phase 1: All cells are Strings, Clean unwanted chars and prep for conversion to Array---
    clean_for_array = f.udf(lambda x: ",".join([x[1:-1].replace(' ', '').replace('\'', '')]) if x is not None else "-",
                            StringType())
    clean_for_array_cit = f.udf(lambda x: ",".join(
        [x[1:-1].replace(' ', '').replace('\'', '').replace('TelAviv', 'Tel Aviv')]) if x is not None else "-",
                                StringType())

    queries = queries \
        .withColumn("genres", f.split(clean_for_array('genres'), ',')) \
        .withColumn("lang", f.split(clean_for_array('lang'), ',')) \
        .withColumn("actors", f.split(clean_for_array('actors'), ',')) \
        .withColumn("director", f.split(clean_for_array('director'), ',')) \
        .withColumn("cities", f.split(clean_for_array_cit('cities'), ',')) \
        .withColumn("country", f.split(clean_for_array('country'), ',')) \
        .withColumn("from_realese_date", f.split(clean_for_array('from_realese_date'), ',')) \
        .withColumn("production_company", f.split(clean_for_array('production_company'), ','))

    # --- Phase 2: Conver Array of String to Array of Integers for the Date Data---
    date_to_int = f.udf(lambda x: [int(x[0])], ArrayType(IntegerType()))

    queries = queries.withColumn("from_realese_date", date_to_int('from_realese_date'))
    return queries


def load_movies(dataPath):  # Final script after review load movies data
    movies_schema = StructType([
        StructField('movie_id', IntegerType(), False),
        StructField('genres', StringType(), False),
        StructField('overview', StringType(), False),
        StructField('production_companies', StringType(), False),
        StructField('production_countries', StringType(), False),
        StructField('release_date', StringType(), False),
        StructField('revenue', IntegerType(), False),
        StructField('spoken_languages', StringType(), False),
        StructField('tagline', StringType(), False),
        StructField('title', StringType(), False),
        StructField('cities', StringType(), False)
    ])

    movies = spark.read.option("Header", True) \
        .option("multiline", True) \
        .option("escape", "\"") \
        .csv(dataPath)
    # (option("quote", "\"")) line above solved the missinterputation of the char " by pySpark

    # --- Phase 1: All cells are Strings ---

    # Build structures for empty arrays
    fix_genres = f.udf(lambda x: "[{'id': '-', 'name': '-'}]" if x == '[]' or None else x, StringType())
    fix_prod_comp = f.udf(lambda x: "[{'name': '-', 'id': '-'}]" if x == '[]' or None else x, StringType())
    fix_prod_coun = f.udf(lambda x: "[{'iso_3166_1': '-', 'name': '-'}]" if x == '[]' or None else x, StringType())
    fix_spoken_lang = f.udf(lambda x: "[{'iso_639_1': '-', 'name': '-'}]" if x == '[]' or None else x, StringType())
    fix_null_cities = f.udf(
        lambda x: x[1:-1].replace(' ', '').replace('\'', '').replace('TelAviv', 'Tel Aviv') if x is not None else "[]",
        StringType())
    fix_null_date = f.udf(lambda x: x if x is not None else "00/00/2030", StringType())  # 2030 as none for us.

    movies = movies \
        .withColumn("genres", fix_genres('genres')) \
        .withColumn("production_companies", fix_prod_comp('production_companies')) \
        .withColumn("production_countries", fix_prod_coun('production_countries')) \
        .withColumn("spoken_languages", fix_spoken_lang('spoken_languages')) \
        .withColumn("release_date", fix_null_date('release_date')) \
        .withColumn("cities", fix_null_cities('cities'))

    movies = movies.na.fill('Empty')

    # At this point, all the cells which have another struct in them, will have an empty struct as string, descreption only cells with null- will have 'Empty' instead of null.
    # This is the way we decided to work with the null cells inorder to control the DataFrame and drop any tuples.

    # --- Phase 2: Semi Structured to Structured ---

    schema_genres = ArrayType(
        StructType([StructField("id", StringType(), True),
                    StructField("name", StringType(), True)]))

    schema_production_company = ArrayType(
        StructType([StructField("name", StringType(), True),
                    StructField("id", StringType(), True)]))

    schema_production_countries = ArrayType(
        StructType([StructField("iso_3166_1", StringType(), True),
                    StructField("name", StringType(), True)]))

    schema_spoken_languages = ArrayType(
        StructType([StructField("iso_639_1", StringType(), True),
                    StructField("name", StringType(), True)]))

    # changes the columns containing lists of json to structured cells.
    movies = movies.withColumn("genres", f.from_json(movies.genres, schema_genres)) \
        .withColumn("production_companies", f.from_json(movies.production_companies, schema_production_company)) \
        .withColumn("production_countries", f.from_json(movies.production_countries, schema_production_countries)) \
        .withColumn("spoken_languages", f.from_json(movies.spoken_languages, schema_spoken_languages))

    # --- Phase 3: Extract feilds based on the query data ---

    # Extracts the value of 'name' in each json in the cell of column
    extract_name = f.udf(lambda x: ",".join([elem['name'] for elem in x]) if x is not None else "-", StringType())

    # Extracts year from the date format
    extract_year = f.udf(lambda x: int(x[-4:]) if '-' not in x else int(x[:4]), IntegerType())

    movies = movies \
        .withColumn("genres", extract_name('genres')) \
        .withColumn("production_companies", extract_name('production_companies')) \
        .withColumn("production_countries", extract_name('production_countries')) \
        .withColumn("spoken_languages", extract_name('spoken_languages')) \
        .withColumn("release_date", extract_year('release_date'))

    # --- Phase 4: Convert Strings representing Arrays to ArrayType(StringType()) ---

    # converts columns which need to be array types from string containing arrays of strings
    movies = movies \
        .withColumn("genres", f.split('genres', ',')) \
        .withColumn("production_companies", f.split('production_companies', ',')) \
        .withColumn("production_countries", f.split('production_countries', ',')) \
        .withColumn("spoken_languages", f.split('spoken_languages', ',')) \
        .withColumn("cities", f.split('cities', ','))

    return movies


if __name__ == '__main__':
    """
    Load data as required, 
    Must have pip installed and all the data csvs in the directory to run.
    checked multiple times on colab
    
    !pip install findspark
    !pip install pyspark
    """
    # Create spark project
    spark, sc = init_spark('project_A')

    # Load Data
    movies = load_movies("movies.csv")
    credits = load_credits("credits.csv")
    users = load_users("users.csv")
    queries = load_queries("queries.csv")
    tickets = load_tickets("tickets.csv")
