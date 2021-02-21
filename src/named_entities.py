import dataloader
import pandas as pd
import spacy
import neuralcoref
from spacy import displacy
from spacy.pipeline import EntityRuler
import en_core_web_sm

nlp = en_core_web_sm.load()
ruler = EntityRuler(nlp, validate=True, overwrite_ents=True).from_disk("patterns.jsonl")
nlp.add_pipe(ruler)

df_bible = dataloader.get_df_bible()
test_df = df_bible.head(42)



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
            if(entity.ent_id_==""):
                characters.add(entity.text)
            else:
                characters.add(entity.ent_id_)
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
    if (end>=len(text_column)): end=len(text_column)-1
    text = ""
    for i in range(end-start+1):
        text += str(text_column[start+i])
        text += "|"
    
    return text

def resolve_coreferences(text, greedy, dist_max, match_dist_max, old_english_conv_dict):
    neuralcoref.add_to_pipe(nlp, greedyness=greedy, max_dist=dist_max, dist_max_match=match_dist_max)
    if(old_english_conv_dict):
        nlp.get_pipe('neuralcoref').set_conv_dict({'Thou': "You", 'Thee': 'You', 'Thy': 'Your', 'Thine': 'your', 'Ye': 'You'})
    doc = nlp(text)
    resolved = doc._.coref_resolved
    resolved = resolved.split("|")
    nlp.remove_pipe("neuralcoref")
    return resolved

    
def coreference_resolution(df, text_column_name="text", start_line=0, end_line=None, greedyness=0.5, max_dist=50, max_dist_match=500, old_english_conv_dict=True):
    df.insert(len(df.columns), "resolved_text", None, True)
    if(end_line==None):
        end_line = len(df[text_column_name])-1
    while(start_line <= len(df)-1 and start_line<=end_line):        
        text = concat_verses(df[text_column_name], start_line, start_line+999)
        resolved = resolve_coreferences(text, greedyness, max_dist, max_dist_match, old_english_conv_dict)
        for i in range(len(resolved)-1):
            df.loc[start_line+i, ("resolved_text")] = resolved[i]
        start_line += 1000
    print("resolved")
    print(df.head())
    return df


test_df = df_bible.head(42)

if __name__ == "__main__":
    df_bible = coreference_resolution(df_bible)
    add_character_column(df_bible)
    print(df_bible.head(42))
    df_bible.to_csv("bibleTA_characters.csv", encoding="utf-8", index=False) 