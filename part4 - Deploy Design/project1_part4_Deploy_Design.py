import os
import findspark
from pyspark.sql.types import StructField, StructType, StringType, DoubleType, IntegerType, DateType, ArrayType
findspark.init()
from pyspark.sql import SparkSession
from pyspark.sql import functions as f


def init_spark(app_name: str):
    spark = SparkSession.builder.appName(app_name).getOrCreate()
    sc = spark.sparkContext
    return spark, sc


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


def haifa(df, dir):
    os.mkdir(dir)  # create new directory for fragments
    print(f"created {dir} directory successfully")

    # part 2 from design PDF- vertical
    haifa_df = df.drop('overview', 'revenue', 'tagline', 'title') \
        .withColumnRenamed('cast', 'actors').withColumnRenamed('crew', 'director')

    # part 3 from design PDF- horizontal
    # filter by release date
    haifa_df = haifa_df.filter(haifa_df.release_date >= 2010)

    # filter by city in movies cities
    haifa_df = haifa_df.select(haifa_df['*'], f.array_contains(haifa_df.cities, 'Haifa')
                               .alias("available_haifa"), f.array_contains(haifa_df.cities, 'Tel Aviv')
                               .alias("available_tlv"), f.array_contains(haifa_df.cities, 'Tiberias')
                               .alias("available_tiberias"))

    haifa_df = haifa_df.filter(
        (haifa_df.available_haifa == True) | (haifa_df.available_tlv == True) | (haifa_df.available_tiberias == True))
    haifa_df = haifa_df.drop('available_haifa', 'available_tlv', 'available_tiberias')

    # add True False columns for the genre and the language constraint
    haifa_df = haifa_df.select(haifa_df['*'], f.array_contains(haifa_df.spoken_languages, 'English')
                               .alias("contains_eng"))

    haifa_frag_1 = haifa_df.filter(haifa_df.contains_eng == False)
    haifa_frag_1 = haifa_frag_1.drop('contains_eng')
    haifa_frag_1 = df_columns_to_str(haifa_frag_1, 1, 1)

    haifa_frag_1.write.option("header", True).option("multiline", True) \
        .option("escape", "\"").csv(f"{dir}/Haifa_fragment_1")
    print('successfully created Haifa fragment 1')

    haifa_frag_2 = haifa_df.filter(haifa_df.contains_eng == True)
    haifa_frag_2 = haifa_frag_2.drop('contains_eng')
    haifa_frag_2 = df_columns_to_str(haifa_frag_2, 1, 1)

    haifa_frag_2.write.option("header", True).option("multiline", True) \
        .option("escape", "\"").csv(f"{dir}/Haifa_fragment_2")
    print('successfully created Haifa fragment 2')
    print()


def tiberias(df, dir):
    os.mkdir(dir)  # create new directory for fragments
    print(f"created {dir} directory successfully")

    # part 2 from design PDF- vertical
    tiberias_df = df.drop('overview', 'revenue', 'tagline', 'title') \
        .withColumnRenamed('cast', 'actors').withColumnRenamed('crew', 'director')

    tiberias_df = tiberias_df.drop('director')

    # part 3 from design PDF- horizontal
    # filter by release date
    tiberias_df = tiberias_df.filter(tiberias_df.release_date >= 1990)

    # filter by city in movies cities
    tiberias_df = tiberias_df.select(tiberias_df['*'], f.array_contains(tiberias_df.cities, 'Tiberias')
                                     .alias("available_tiberias"), f.array_contains(tiberias_df.cities, 'Haifa')
                                     .alias("available_haifa"))

    tiberias_df = tiberias_df.filter((tiberias_df.available_tiberias == True) | (tiberias_df.available_haifa == True))
    tiberias_df = tiberias_df.drop('available_tiberias', 'available_haifa')

    # add True False columns for the genre and the language constraint
    tiberias_df = tiberias_df.select(tiberias_df['*'], f.array_contains(tiberias_df.spoken_languages, 'English')
                                     .alias("contains_eng"))

    tiberias_frag_1 = tiberias_df.filter(tiberias_df.contains_eng == False)
    tiberias_frag_1 = tiberias_frag_1.drop('contains_eng')
    tiberias_frag_1 = df_columns_to_str(tiberias_frag_1, 0, 1)

    tiberias_frag_1.write.option("header", True).option("multiline", True) \
        .option("escape", "\"").csv(f"{dir}/Tiberias_fragment_1")
    print('successfully created Tiberias fragment 1')

    tiberias_frag_2 = tiberias_df.filter(tiberias_df.contains_eng == True)
    tiberias_frag_2 = tiberias_frag_2.drop('contains_eng')
    tiberias_frag_2 = df_columns_to_str(tiberias_frag_2, 0, 1)

    tiberias_frag_2.write.option("header", True).option("multiline", True) \
        .option("escape", "\"").csv(f"{dir}/Tiberias_fragment_2")
    print('successfully created Tiberias fragment 2')
    print()


def jerusalem(df, dir):
    os.mkdir(dir)  # create new directory for fragments
    print(f"created {dir} directory successfully")

    # part 2 from design PDF- vertical
    jr_df = df.drop('overview', 'revenue', 'tagline', 'title') \
        .withColumnRenamed('cast', 'actors').withColumnRenamed('crew', 'director')

    jr_director = jr_df.select('movie_id', 'director', 'release_date')

    jr_df = jr_df.drop('director')

    # part 3 from design PDF- horizontal
    # filter by release date
    jr_df = jr_df.filter(jr_df.release_date >= 1990)

    # filter by city in movies cities
    jr_df = jr_df.select(jr_df['*'], f.array_contains(jr_df.cities, 'Tel Aviv')
                         .alias("available_tlv"), f.array_contains(jr_df.cities, 'Jerusalem')
                         .alias("available_jerusalem"))

    jr_df = jr_df.filter((jr_df.available_tlv == True) | (jr_df.available_jerusalem == True))
    jr_df = jr_df.drop('available_jerusalem', 'available_tlv')

    # add True False columns for the genre and the language constraint
    jr_df = jr_df.select(jr_df['*'], f.array_contains(jr_df.spoken_languages, 'English')
                         .alias("contains_eng"))

    jr_frag_1 = jr_df.filter(jr_df.contains_eng == False)
    jr_frag_1 = jr_frag_1.drop('contains_eng')
    jr_frag_1 = df_columns_to_str(jr_frag_1, 0, 1)

    jr_frag_1.write.option("header", True).option("multiline", True) \
        .option("escape", "\"").csv(f"{dir}/Jerusalem_fragment_1")
    print('successfully created Jerusalem fragment 1')

    jr_frag_2 = jr_df.filter(jr_df.contains_eng == True)
    jr_frag_2 = jr_frag_2.drop('contains_eng')
    jr_frag_2 = df_columns_to_str(jr_frag_2, 0, 1)

    jr_frag_2.write.option("header", True).option("multiline", True) \
        .option("escape", "\"").csv(f"{dir}/Jerusalem_fragment_2")
    print('successfully created Jerusalem fragment 2')

    jr_director = jr_director.filter(jr_director.release_date >= 1990)
    jr_director = jr_director.drop('release_date')
    jr_director = df_columns_to_str(jr_director, 2, 2)

    jr_director.write.option("header", True).option("multiline", True) \
        .option("escape", "\"").csv(f"{dir}/Jerusalem_fragment_3")
    print('successfully created Jerusalem fragment 3')

    print()


def eilat(df, dir):
    os.mkdir(dir)  # create new directory for fragments
    print(f"created {dir} directory successfully")

    # part 2 from design PDF - vertical
    eilat_df = df.drop('overview', 'revenue', 'tagline', 'title') \
        .withColumnRenamed('cast', 'actors').withColumnRenamed('crew', 'director')

    eilat_df = eilat_df.drop('director', 'actors')

    # part 3 from design PDF- horizontal
    # filter by release date
    eilat_df = eilat_df.filter(eilat_df.release_date >= 1990)

    # filter by city in movies cities
    eilat_df = eilat_df.select(eilat_df['*'], f.array_contains(eilat_df.cities, 'Eilat')
                               .alias("available_eilat"))

    eilat_df = eilat_df.filter(eilat_df.available_eilat == True)
    eilat_df = eilat_df.drop('available_eilat')

    # add True False columns for the genre and the language constraint
    eilat_df = eilat_df.select(eilat_df['*'], f.array_contains(eilat_df.spoken_languages, 'English')
                               .alias("contains_eng"))

    eilat_frag_1 = eilat_df.filter(eilat_df.contains_eng == False)
    eilat_frag_1 = eilat_frag_1.drop('contains_eng')
    eilat_frag_1 = df_columns_to_str(eilat_frag_1, 0, 0)

    eilat_frag_1.write.option("header", True).option("multiline", True) \
        .option("escape", "\"").csv(f"{dir}/Eilat_fragment_1")
    print('successfully created Eilat fragment 1')

    eilat_frag_2 = eilat_df.filter(eilat_df.contains_eng == True)
    eilat_frag_2 = eilat_frag_2.drop('contains_eng')
    eilat_frag_2 = df_columns_to_str(eilat_frag_2, 0, 0)

    eilat_frag_2.write.option("multiline", True) \
        .option("escape", "\"").option("header", True).csv(f"{dir}/Eilat_fragment_2")
    print('successfully created Eilat fragment 2')
    print()


def telaviv(df, dir):
    os.mkdir(dir)  # create new directory for fragments
    print(f"created {dir} directory successfully")

    # part 2 from design PDF- vertical
    tlv_df = df.drop('overview', 'revenue', 'tagline', 'title') \
        .withColumnRenamed('cast', 'actors').withColumnRenamed('crew', 'director')

    tlv_df = tlv_df.drop('director')

    # part 3 from design PDF- horizontal
    # filter by release date
    tlv_df = tlv_df.filter(tlv_df.release_date >= 2010)

    # filter by city in movies cities
    tlv_df = tlv_df.select(tlv_df['*'], f.array_contains(tlv_df.cities, 'Tel Aviv')
                           .alias("available_tlv"), f.array_contains(tlv_df.cities, 'Jerusalem')
                           .alias("available_jerusalem"))

    tlv_df = tlv_df.filter((tlv_df.available_tlv == True) | (tlv_df.available_jerusalem == True))
    tlv_df = tlv_df.drop('available_jerusalem', 'available_tlv')

    # add True False columns for the genre and the language constraint
    tlv_df = tlv_df.select(tlv_df['*'], f.array_contains(tlv_df.spoken_languages, 'English')
                           .alias("contains_eng"))

    tlv_frag_1 = tlv_df.filter(tlv_df.contains_eng == False)
    tlv_frag_1 = tlv_frag_1.drop('contains_eng')
    tlv_frag_1 = df_columns_to_str(tlv_frag_1, 0, 1)

    tlv_frag_1.write.option("header", True).option("multiline", True) \
        .option("escape", "\"").csv(f"{dir}/Tel_Aviv_fragment_1")
    print('successfully created Tel Aviv fragment 1')

    tlv_frag_2 = tlv_df.filter(tlv_df.contains_eng == True)
    tlv_frag_2 = tlv_frag_2.drop('contains_eng')
    tlv_frag_2 = df_columns_to_str(tlv_frag_2, 0, 1)

    tlv_frag_2.write.option("header", True).option("multiline", True) \
        .option("escape", "\"").csv(f"{dir}/Tel_Aviv_fragment_2")
    print('successfully created Tel Aviv fragment 2')
    print()


def df_columns_to_str(df, withDirector, withActor):
    to_string = f.udf(lambda x: str(x), StringType())

    if (withDirector == 1) and (withActor == 1):
        df = df.withColumn("genres", to_string('genres')) \
            .withColumn("production_companies", to_string('production_companies')) \
            .withColumn("production_countries", to_string('production_countries')) \
            .withColumn("spoken_languages", to_string('spoken_languages')) \
            .withColumn("cities", to_string('cities')) \
            .withColumn("actors", to_string('actors')) \
            .withColumn("director", to_string('director'))

    elif (withDirector == 0) and (withActor == 1):
        df = df.withColumn("genres", to_string('genres')) \
            .withColumn("production_companies", to_string('production_companies')) \
            .withColumn("production_countries", to_string('production_countries')) \
            .withColumn("spoken_languages", to_string('spoken_languages')) \
            .withColumn("cities", to_string('cities')) \
            .withColumn("actors", to_string('actors'))

    elif (withDirector == 0) and (withActor == 0):
        df = df.withColumn("genres", to_string('genres')) \
            .withColumn("production_companies", to_string('production_companies')) \
            .withColumn("production_countries", to_string('production_countries')) \
            .withColumn("spoken_languages", to_string('spoken_languages')) \
            .withColumn("cities", to_string('cities'))

    elif (withDirector == 2) and (withActor == 2):  # director table stored in Jerusalem site
        df = df.withColumn('director', to_string('director'))

    return df


if __name__ == '__main__':
    """
    Deploy design pdf from question 3
    checked multiple times on colab, 
    must use:
    
    !pip install findspark
    !pip install pyspark
    
    and store the data movies.csv and credits.csv in the same directory
    
    """
    # Create spark project
    spark, sc = init_spark('project_A')
    # Load Data
    movies = load_movies("movies.csv")
    credits = load_credits("credits.csv")

    # part 1 from design PDF
    df = movies.join(credits, movies.movie_id == credits.id, "inner").drop(credits.id)

    # each function creates fragments and exports to csv with new directory
    haifa(df, 'Haifa___')
    tiberias(df, 'Tiberias_')
    telaviv(df, 'Tel_Aviv_')
    jerusalem(df, 'Jerusalem_')
    eilat(df, 'Eilat_')

    print()
    print("Successfully Exported all fregmentations")
