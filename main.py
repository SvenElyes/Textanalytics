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

 
