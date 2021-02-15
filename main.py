"""
Felix Munzlinger, Johannes Klueh, Sven Leschber
Text Analytics, 2020-2021

The Purpose of this file is to pipeline and document all our progress and work

bibleTA.csv # original bible file
"""

import dataloader
import pickle_handler
import relation_creator

if __name__ == "__main__":
    df_bible = dataloader.get_df_bible()

    """mine (sven) part, schaut ob ihr was dazwischen einfuegen wollt, bzw was ihr braucht"""
    pickleObject = PickleHandler()
    # clear the .pkl files before starting the process?

    # this object should be the schnittstelle for saving and accessing our list of relations and characters

    # creating characters and relationships
    relation_creator.create_char_relation(DISTLLED CSV)
    relation_creator.create_character_keywords()

    #WE NEED THE DISTILLED CSV FROM EVAL GRAPH for the first step

    if you need to get a list of charactes/reltaions just use pickleObject s load functions 
