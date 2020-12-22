import dataloader
import pandas as pd
import spacy
from spacy import displacy
import en_core_web_sm

nlp = en_core_web_sm.load()
df_bible = dataloader.get_df_bible()

def get_characters_from_text(text):
    doc = nlp(text)
    for entity in doc.ents:
        if entity.label_ == "PERSON":
            print(entity.label_, ' | ', entity.text)

def show_word_mapping(text):
    doc = nlp(text)
    displacy.serve(doc, style='dep')

if __name__ == "__main__":
    for verse in df_bible["text"]:
        get_characters_from_text(verse)