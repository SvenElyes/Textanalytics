import pandas as pd
import sys
import json
from pickle_handler import PickleHandler

sys.path.append("./data/")

from character import Character
from relation import Relation

""" 
This module aims to create the relationship objects between 2 Character objects and for later to assign each realtionship its properties.
It loads the data from the bibleTA_characters.csv file,  which has a collumn called characters, which specify all the character which appeareed in each verse
"""


def load_df_2_char():
    df_bible_characters = pd.read_csv("bibleTA_characters.csv")
    df_bible_characters.dropna(subset=["characters"], inplace=True)
    """create a new coloumns containing the number of characters in each row
    
    """
    """
    df_bible_characters["number_of_characters"] = (
        df_bible_characters["characters"].str.count("| ") + 1
    )
    """
    # TODO PLS FIX THIS THE ABOVE ONE DOESNT WORK !!!!
    num_char = []
    char = df_bible_characters["characters"]
    for x in char:
        num_char.append(x.count("|") + 1)

    df_bible_characters["number_of_characters"] = num_char

    # print(df_bible_characters.head(20))
    """drop the rows in which there are not exactly 2 members as we will look at those relationships first"""

    df_bible_characters = df_bible_characters.loc[
        df_bible_characters["number_of_characters"] == 2
    ]

    return df_bible_characters


def write_df_to_obj(df_bible, pickleHandler):
    """this function creates for each name in the character column a character object and creates a corresponding relationship object and saves the characters in the pickle file"""
    for _, row in df_bible.iterrows():
        string = row["characters"]
        string = string.split("| ")
        """ Create 2 Characters"""
        char1 = Character(string[0])
        char2 = Character(string[1])

        """ Create 2 Relations for each Charachter """

        char1.add_relation(char2)
        char2.add_relation(char1)

        pickleHandler.save_character_list([char1, char2])

    """we then clean up the objects by summarizing it and deleting duplicates"""
    pickleHandler.clear_duplicates()


if __name__ == "__main__":

    df_bible = load_df_2_char()
    pickleHandler = PickleHandler()
    write_df_to_obj(df_bible, pickleHandler)

    list_of_characters = pickleHandler.load_characters()
    """
    for i in list_of_characters:
        print(i.name, i.get_relations()[0].get_target_character().name)
    """