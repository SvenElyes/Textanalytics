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
    """Get a set of characters, recognized by spacy, from a string
    :type text: string
    :param text: entirety of the bible, or a part of it, as a string, so we can use the nlp call on it
    :return: character names
    :rtype: set

    """
    doc = nlp(text)
    characters = set()
    for entity in doc.ents:
        if entity.label_ == "PERSON":
            characters.add(entity.text)
    return characters


def show_word_mapping(text):
    """Visualizing a dependency parse or named entities in a text

    :type text: string
    :param text: entirety of the bible, or a part of it, as a string, so we can use the nlp call on it
    :return: nothing

    """
    doc = nlp(text)
    displacy.serve(doc, style="dep")
    

def add_character_column(df):
    """Adds a new collumn in the DataFrame called characters and fills it with the occuring characters

    :type df: pandas.Dataframe
    :param text: entirety of the bible as a DataFrame
    :return: nothing

    """
    df.insert(len(df.columns), "characters", None, True)
    for i in range(len(df)):
        characters = get_characters_from_text(df.loc[i, ("resolved_text")])
        df.loc[i, ("characters")] = "|".join(characters)


def concat_verses(text_column, start=None, end=None):
    if start==None: start=0
    if (end==None or end>=len(text_column)): end=len(text_column)-1
    text = ""
    for i in range(end-start+1):
        text += str(text_column[start+i])
        text += "|"
    
    return text

def resolve_coreferences(text):
    doc = nlp(text)
    resolved = doc._.coref_resolved
    resolved = resolved.split("|")
    return resolved


if __name__ == "__main__":
    start=0
    df_bible.insert(len(df_bible.columns), "resolved_text", None, True)
    while(start <= len(df_bible)-1):
        text = concat_verses(df_bible["text"], start, start+999)
        print("start: ", start)
        #print("concatenated\n")
        resolved = resolve_coreferences(text)
        #print("resolved\n")
        for i in range(len(resolved)-1):
            df_bible.loc[start+i, ("resolved_text")] = resolved[i]
        #print("replaced\n")
        start+=1000
    add_character_column(df_bible)
    print(df_bible.head(42))
    df_bible.to_csv("bibleTA_characters.csv", encoding="utf-8", index=False) 