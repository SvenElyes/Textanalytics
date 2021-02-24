import dataloader
import pandas as pd
import spacy
import neuralcoref
from spacy import displacy
from spacy.pipeline import EntityRuler
import en_core_web_sm



def get_characters_from_text(text, nlp):
    """Get a set of characters from text using "ner" component from spacy pipeline.
    :type text: string
    :param text: entirety of the bible, or a part of it, as a string, so we can use the nlp call on it. Works for other texts too.
    :type nlp: spacy.Language
    :param nlp: pretrained language model. Use as default spacy.lang.en.English. Works for other trained language models too.
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


def show_word_mapping(text, nlp):
    """Visualizing a dependency parse or named entities in a text

    :type text: string
    :param text: entirety of the bible, or a part of it, as a string, so we can use the nlp call on it
    :type nlp: spacy.Language
    :param nlp: pretrained language model. Use as default spacy.lang.en.English. Works for other trained language models too.
    :return: nothing

    """
    doc = nlp(text)
    displacy.serve(doc, style="dep")
    

def add_character_column(df, nlp):
    """Adds a new column in the DataFrame called "characters" and fills it with the occuring characters

    :type df: pandas.Dataframe
    :param text: entirety of the bible as a DataFrame. Works for other DataFrames too.
    :type nlp: spacy.Language
    :param nlp: pretrained language model. Use as default spacy.lang.en.English. Works for other trained language models too.
    :return: nothing

    """
    df.insert(len(df.columns), "characters", None, True)
    for i in range(len(df)):
        characters = get_characters_from_text(df.loc[i, ("resolved_text")], nlp)
        df.loc[i, ("characters")] = "|".join(characters)


def concat_verses(text_column, start=None, end=None):
    """Concatenate lines from pandas DataFrame in order to create larger text slices for coreference resolution.


    :type text_column: pandas.core.series.Series
    :param text_column: text column from pandas Dataframe or simply pandas Series containing the text to be analyzed.
    :type start: int
    :param start: First line in text_column to be concatenated.
    :type end: int
    :param end: Last line in text_columns to be concatenated.
    :return: concatenated text
    :rtype: string

    """
    if start==None: start=0
    if (end>=len(text_column)): end=len(text_column)-1
    text = ""
    for i in range(end-start+1):
        text += str(text_column[start+i])
        text += "|"
    
    return text

def resolve_coreferences(text, greedy, dist_max, match_dist_max, old_english_conv_dict, separator="|"):
    '''Resolve coreferences inside a given text.

    :type text: string
    :param text: text to be processed
    :type greedy: float
    :param greedy: value between 0 and 1 for greedyness in neuralcoref. Higher value means more links.
    :type dist_max: int
    :param dist_max: neuralcoref parmeter max_dist
    :type match_dist_max: int
    :param match_dist_max: neuralcoref parameter max_dist_match
    :type old_english_conv_dict: bool
    :param old_english_conv_dict: conversion dictionary for neuralcoref with common pronouns in old english
    :type separator: char
    :param separator: separator for string split after neuralcoref.
    :return: list of resolved lines
    :rtype: list

    Warning: greedyness parameter seems to be causing errors for some values. This is an issue of neuralcoref and not part of this library

    '''
    nlp = en_core_web_sm.load()
    neuralcoref.add_to_pipe(nlp, greedyness=greedy, max_dist=dist_max, dist_max_match=match_dist_max)
    if(old_english_conv_dict):
        nlp.get_pipe('neuralcoref').set_conv_dict({'Thou': 'You', 'Thee': 'You', 'Thy': 'Your', 'Thine': 'Your', 'Ye': 'You'})
    doc = nlp(text)
    resolved = doc._.coref_resolved
    resolved = resolved.split(separator)
    return resolved

    
def coreference_resolution(df, text_column_name="text", start_line=0, end_line=None, greedyness=0.5, max_dist=50, max_dist_match=500, old_english_conv_dict=True):
    '''Coreference resolution for given pandas DataFrame. Adds a column containing the resolved text.

    :type df: pandas.core.frame.DataFrame
    :param df: DataFrame with a text column which has to be resolved
    :type text_column_name: string
    :param text_column_name: name of the column containing the text
    :type start_line: int
    :param start_line: fist row in df to be resolved
    :type end_line: int
    :param end_line: last row in df to be resolved
    :type greedyness: float
    :param greedyness: value between 0 and 1 for greedyness in neuralcoref. Higher value means more links.
    :type dist_max: int
    :param dist_max: neuralcoref parmeter max_dist
    :type match_dist_max: int
    :param match_dist_max: neuralcoref parameter max_dist_match
    :type old_english_conv_dict: bool
    :param old_english_conv_dict: conversion dictionary for neuralcoref with common pronouns in old english
    :return: DataFrame containing "resolved_text" column
    :rtype: pandas.core.frame.DataFrame

    Warning: greedyness parameter seems to be causing errors for some values. This is an issue of neuralcoref and not part of this library

    '''
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

def extract_characters(df=None, rule_based_matching=True, use_bible_patterns=True, patterns=None, export_csv=True, csv_name="src/csv/bibleTA_characters.csv"):
    '''Using Named Entitiy Recognition from spacy to extract character Names from given text.

    :type df: pandas.core.frame:DataFrame
    :param df: DataFrame containing a text column
    :type rule_based_matching: bool
    :param rule_based_matching: use spacy's EntityRuler for additional matching rules. Can only be used when coming either with "patterns" or "use_bible_patterns"
    :type use_bible_patterns: bool
    :param use_bible_patterns: patterns containing the names of most bible characters including synonyms. "rule_based_matching" must be True
    :type patterns: list
    :param patterns: patterns for EntiyRuler. "rule_based_matching" must be True
    :type export_csv: bool
    :param export_csv: save df as csv on disk
    :type csv_name: string
    :param csv_name: file name for csv output
    :return: character column
    :rtype: pandas.core.series.Series

    '''
    if(df is None):
        df = dataloader.get_df_bible()
    nlp = en_core_web_sm.load()
    
    if(rule_based_matching and use_bible_patterns):
        ruler = EntityRuler(nlp, validate=True, overwrite_ents=True).from_disk("src/patterns.jsonl")
        nlp.add_pipe(ruler)
    elif(rule_based_matching and patterns is not None):
        ruler = EntityRuler(nlp)
        ruler.add_patterns(patterns)
        nlp.add_pipe(ruler)

    add_character_column(df, nlp)

    if(export_csv):
        df.to_csv(csv_name, encoding="utf-8", index=False)

    return df["characters"]