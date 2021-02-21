"""
Felix Munzlinger, Johannes Klueh, Sven Leschber
Text Analytics, 2020-2021

The Purpose of this file is to pipeline and document all our progress and work

bibleTA.csv # original bible file
"""
from os.path import exists

import src.dataloader as dataloader
import src.pickle_handler as pickle_handler
import src.relation_creator as relation_creator
import preprocess_emotion
import join_df
import eval_graph

if __name__ == "__main__":
    if exists("bibleTA_prepro.csv") == False:
        df_bible = preprocess_emotion.main("both")

    join_df.main(character_csv="bibleTA_characters.csv", relation_csv="bibleTA_prepro.csv", out_csv="bibleTA_emotion.csv")
    eval_graph.main(threshold_getgraph=5, num_cluster=4, threshold_getcluster=(1 / 6),
                    file="bibleTA_emotion.csv", load=True)

    '''
    """mine (sven) part, schaut ob ihr was dazwischen einfuegen wollt, bzw was ihr braucht"""
    pickleObject = PickleHandler()
    # clear the .pkl files before starting the process?

    # this object should be the API for saving and accessing our list of relations and characters

    # creating characters and relationships
    relation_creator.create_char_relation(DISTLLED CSV)
    relation_creator.create_character_keywords()

    #WE NEED THE DISTILLED CSV FROM EVAL GRAPH for the first step

    #if you need to get a list of charactes/reltaions just use pickleObject s load functions 
    '''
    
