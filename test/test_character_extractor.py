from unittest import TestCase
import pandas as pd
import spacy
import neuralcoref
from spacy.pipeline import EntityRuler
import en_core_web_sm

import src.character_extractor
import src.dataloader

df_bible = dataloader.get_df_bible()
df_test = df_bible.head(42)

