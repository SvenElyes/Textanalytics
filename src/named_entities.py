import dataloader
import pandas as pd
import spacy
import neuralcoref
from spacy import displacy
from spacy.pipeline import EntityRuler
import en_core_web_sm

nlp = en_core_web_sm.load()
ruler = EntityRuler(nlp, validate=True).from_disk("patterns.jsonl")
nlp.add_pipe(ruler)
neuralcoref.add_to_pipe(nlp)
df_bible = dataloader.get_df_bible()

def get_characters_from_text(text):
    doc = nlp(text)
    characters = set()
    for entity in doc.ents:
        if entity.label_ == "PERSON":
            characters.add(entity.text)
    return characters

def add_character_column(df):
    df.insert(len(df.columns), "characters", None, True)
    for i in range(len(df)):
        characters = get_characters_from_text(df.loc[i, ("text")])
        df.loc[i, ("characters")] = "|".join(characters)





test_df = df_bible.head(10)

if __name__ == "__main__":
    doc = nlp("Angela lives in Boston. She is quite happy in that city.")
    for ent in doc.ents:
        print(ent._.coref_cluster)
    #add_character_column(df_bible)
    #df_bible.to_csv("bibleTA_characters.csv", encoding="utf-8", index=False)