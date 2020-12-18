import json
import numpy as np
import pandas as pd


# a list of all books.
books = [
    "Genesis",
    "Exodus",
    "Leviticus",
    "Numbers",
    "Deuteronomy",
    "Joshua",
    "Judges",
    "Ruth",
    "1 Samuel",
    "2 Samuel",
    "1 Kings",
    "2 Kings",
    "1 Chronicles",
    "2 Chronicles",
    "Ezra",
    "Nehemiah",
    "Esther",
    "Job",
    "Psalm",
    "Proverbs",
    "Ecclesiastes",
    "Song of Solomon",
    "Isaiah",
    "Jeremiah",
    "Lamentations",
    "Ezekiel",
    "Daniel",
    "Hosea",
    "Joel",
    "Amos",
    "Obadiah",
    "Jonah",
    "Micah",
    "Nahum",
    "Habakkuk",
    "Zephaniah",
    "Haggai",
    "Zechariah",
    "Malachi",
    "Matthew",
    "Mark",
    "Luke",
    "John",
    "Acts",
    "Romans",
    "1 Corinthians",
    "2 Corinthians",
    "Galatians",
    "Ephesians",
    "Philippians",
    "Colossians",
    "1 Thessalonians",
    "2 Thessalonians",
    "1 Timothy",
    "2 Timothy",
    "Titus",
    "Philemon",
    "Hebrews",
    "James",
    "1 Peter",
    "2 Peter",
    "1 John",
    "2 John",
    "3 John",
    "Jude",
    "Revelation",
]

# this function returns two Dataframes, the first being the old Testament and the second being the new Testament
# Matthew is the first book of the new testatement
def get_old_new_testament():
    return True


# Function to get the Panda Dataframe
def get_df_bible():
    # This file loads the bible from the bibleTA.csv which was created in bibleToCSV.py
    df_bible = pd.read_csv("bibleTA.csv")
    df_bible.drop(["Unnamed: 0"], axis=1, inplace=True)
    return df_bible


df_bible = get_df_bible()
# TODO think of new functions which will ensures less problems in the actual workflow
print(df_bible.tail(1))
print(len(books))
