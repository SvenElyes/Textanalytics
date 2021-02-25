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
import src.preprocess_emotion as preprocess_emotion
import src.join_df as join_df
import src.eval_graph as eval_graph
import src.character_extractor as character_extractor
import pandas as pd
import os
import numpy as np

def main(testament="both"):
    os.makedirs("src/csv", exist_ok=True)
    df_bible = None
    df_bible = preprocess_emotion.main("both", None)
    # check if a dataframe is given or start setting up on by preprocess_emotion.main()
    if exists("src/csv/bibleTA_prepro.csv") == False:
        df_bible = preprocess_emotion.main("both", None, "csv/bibleTA_prepro.csv")
    else:
        df_bible = pd.read_csv("src/csv/bibleTA_prepro.csv")

    if exists("src/csv/bibleTA_characters.csv") == False:
        df_raw_bible = dataloader.get_df_bible()
        df_resolved = character_extractor.coreference_resolution(df_raw_bible)
        df_characters = character_extractor.extract_characters(df_resolved)
    else:
        df_characters = pd.read_csv("src/csv/bibleTA_characters.csv")

    # unite two dataframes since the calculation of "bibleTA_prepro.csv" takes a substancial amount of time
    df_bible = join_df.main(character_csv="src/csv/bibleTA_characters.csv", relation_csv="src/csv/bibleTA_prepro.csv",
                            out_csv="src/csv/bibleTA_emotion.csv", df_bible=df_bible, df_characters=df_characters)

    # if a specific testament is given, reduce dataframe
    if testament == "new":
        _, df_bible = dataloader.get_old_new_testament(df_bible)
    if testament == "old":
        df_bible, _ = dataloader.get_old_new_testament(df_bible)
    top = pd.DataFrame()
    columns = df_bible.columns.tolist()[1:]

    # determine the graph, create pickle objects and run a clustering of keywords
    eval_graph.main(threshold_getgraph=5, num_cluster=4, threshold_getcluster=(1 / 6),
                    file="src/csv/bibleTA_emotion.csv", load=True, df_bible=df_bible)

if __name__ == "__main__":
    main()

