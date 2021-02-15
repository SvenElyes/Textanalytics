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
    """Get a set of characters, recognized by spacy, from a string
    :type text: string
    :param text: entirety of the bible, or a part of it, as a string, so we can use the nlp call on it
    :return: character names
    :rtype: set

    """
    doc = nlp(text)
    characters = set()
    for entity in doc.ents:
        # print(entity.text)
        if entity.label_ == "PERSON":
            # print(entity.label_, ' | ', entity.text)
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
        characters = get_characters_from_text(df.loc[i, ("text")])
        df.loc[i, ("characters")] = "| ".join(characters)


if __name__ == "__main__":
    add_character_column(df_bible)
    df_bible.to_csv("bibleTA_characters.csv", encoding="utf-8", index=False)