import pandas as pd
import sys
import json
from pickle_handler import PickleHandler

sys.path.append("./data/")

from character import Character
from relation import Relation


def create_char_relation(df_bibleTA_distilled):
    """This function will create all character and relationobjects from the distilled csv
    we will use a csv which was generated in the distillDataFrame function, which is located in the eval_graph.py function.The CSV has following form:
    ,character_A,character_B,emotion
    0, God,Abraham,0.0
    1, God,ye,0.0
    One thing to remeber, is that the relations are distinct, so we will not have the same relationship in two seperate rows."""

    """each time we save a character or a relation we do have to do expensive operations in I/O a .pkl file."""
    character_list = []
    character_name_list = []
    relation_list = []
    for _, row in df_bibleTA_distilled.iterrows():
        character_A_name, character_B_name, emotion = (
            row["character_A"].lstrip(),
            row["character_B"].lstrip(),
            row["emotion"],
        )

        """check if we already encountered the character in a previous loop process"""
        if character_A_name in character_name_list:
            """get the character from the character_list"""
            for character in character_list:
                if character.get_name() == character_A_name:
                    character_A = character
            character_A_exists = False
        else:
            character_A = Character(character_A_name)
            character_name_list.append(character_A_name)
            character_list.append(character_A)
            character_A_exists = True

        if character_B_name in character_name_list:
            """get the character from the character_list"""
            for character in character_list:
                if character.get_name() == character_B_name:
                    character_B = character
            character_B_exists = False
        else:
            character_B = Character(character_B_name)
            character_name_list.append(character_B_name)
            character_list.append(character_B)
            character_B_exists = True

        relation = Relation(character_A, character_B, emotion)
        character_A.add_relation(relation)
        character_B.add_relation(relation)
        relation_list.append(relation)
    picklehandler = PickleHandler()
    picklehandler.save_character_list(character_list)
    picklehandler.save_relation_list(relation_list)


if __name__ == "__main__":
    df_bibleTA_distilled = pd.read_csv("bibleTA_distilled_new_8.csv")
    create_char_relation(df_bibleTA_distilled)
    picklehandler = PickleHandler()
    relations = picklehandler.load_relations()
    characters = picklehandler.load_characters()
