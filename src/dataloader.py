import json
import numpy as np
import pandas as pd


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


def get_old_new_testament(whole_bible):
    """Splits bible dataframe into old and new testament dataframes

    Matthew is the first book of the new testatement, therefore the split occurs before this book's first verse.

    :type whole_bible: pandas.DataFrame
    :param whole_bible: entirety of the bible as dataframe, as created by get_df_bible()
    :return: separated old and new testament in tuple (old, new)
    :rtype: tuple
    
    """
    first_matthew_verse = whole_bible.index[
        (whole_bible["book_id"] == "Matt") & (whole_bible["verse"] == 1) & (whole_bible["chapter"] == 1)].tolist()[0]

    old_testament_df = whole_bible[:(first_matthew_verse-1)]
    new_testament_df = whole_bible[first_matthew_verse:]

    return old_testament_df, new_testament_df


def get_df_bible():
    """Reads bible CSV file into pandas dataframe"""
    df_bible = pd.read_csv("src/bibleTA.csv")  # created in bibleToCSV.py
    df_bible.drop(["Unnamed: 0"], axis=1, inplace=True)
    return df_bible


if __name__ == "__main__":
    df_bible = get_df_bible()
    # TODO think of new functions which will ensures less problems in the actual workflow
    print(df_bible.tail(1))
    #print(len(books))
