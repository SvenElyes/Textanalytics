import dataloader
import pandas as pd
import spacy
from spacy import displacy
from spacy.pipeline import EntityRuler
import en_core_web_sm

nlp = en_core_web_sm.load()
ruler = EntityRuler(nlp, validate=True).from_disk("patterns.jsonl")
nlp.add_pipe(ruler)
df_bible = dataloader.get_df_bible()

def get_characters_from_text(text):
    doc = nlp(text)
    characters = set()
    for entity in doc.ents:
        #print(entity.text)
        if entity.label_ == "PERSON":
            #print(entity.label_, ' | ', entity.text)
            characters.add(entity.text)
    return characters

def show_word_mapping(text):
    doc = nlp(text)
    displacy.serve(doc, style='dep')

def add_character_column(df):
    df.insert(len(df.columns), "characters", None, True)
    for i in range(len(df)):
        characters = get_characters_from_text(df.loc[i, ("text")])
        df.loc[i, ("characters")] = "| ".join(characters)

test_df = df_bible.head(10)

if __name__ == "__main__":
    add_character_column(df_bible)
    df_bible.to_csv("bibleTA_characters.csv", encoding="utf-8", index=False)