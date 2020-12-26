# Because we have the Bible as a Json File, we first have to convert it to a CSV File, so it is easier to later read DataFrames from the csv file.
import json
import numpy as np
import pandas as pd

# load the bible. We have to read each line by itself because the way the source data is structured, each line is a new json object.
# the bible is from following https://github.com/bibleapi/bibleapi-bibles-json git Repo.
bible = [json.loads(line) for line in open("bibleapi-bibles-json/asv.json", "r")]

# initialize the DataFrame with the appropiate columns.
df_bible = pd.DataFrame(
    columns=["chapter", "verse", "text", "translation_id", "book_id", "book_name"]
)

# iterate over each verse in the bible and create a row for it in the dataframe
for index in range(len(bible)):
    df_bible.loc[index] = bible[index]

# So we dont have to go through this process each time, we save the DataFrame as a CSV, so we can later load it from there
df_bible.to_csv(path_or_buf="src/bibleTA.csv")
