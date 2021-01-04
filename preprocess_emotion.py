import re
# import pandas as pd
from pattern.text.en import singularize
import spacy, en_core_web_sm
import textacy
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from tqdm import tqdm
import argparse
import numpy as np

import src.dataloader as dataloader

parser = argparse.ArgumentParser(description='train models for matching tasks with GT correspondences')
parser.add_argument('--testament', type=str, default="new", metavar='X', help='{both, old, mew}')

# currently not needed. may be removed later, if not needed then
# split sentences
def splitSentences(text):
    # split sentences and questions
    text = re.split('[.?]', text)
    clean_sent = []
    for sent in text:
        if sent != "":
            clean_sent.append(sent)
    return clean_sent


# try to singularize verbs to make them comparable to pos/neg bag of words.
# is done to detect "goes" as "go" and "likes" as "like".
def preText(text):
    # lowers the text
    text = text.lower()
    # recognize verb pattern
    pattern = [{"POS": "VERB", "OP": "*"}, {"POS": "ADV", "OP": "*"}, {"POS": "VERB", "OP": "+"},
               {"POS": "PART", "OP": "*"}]

    # nlp = en_core_web_sm.load()

    # extract verb pattern
    doc = textacy.make_spacy_doc(text, lang='en_core_web_sm')
    verbs = textacy.extract.matches(doc, pattern)
    for verb in verbs:
        # singularize verb, e.g. "likes" to "like"
        text = text.replace(verb.text, singularize(verb.text))

    return text

# needed to clear the text and proces it by bag of words
def clearText(text):
    # removing paragraph numbers
    text = re.sub('[0-9]+.\t', '', str(text))

    # remove punctuations
    text = re.sub('[.!?]', '', str(text))

    # removing new line characters
    text = re.sub('\n', '', str(text))

    # removing apostrophes
    text = re.sub("'s", '', str(text))

    # removing hyphens
    text = re.sub("-", ' ', str(text))
    text = re.sub("— ", '', str(text))

    # removing quotation marks
    text = re.sub('\"', '', str(text))

    # removing space after sentence
    text = re.sub('\. ', '.', str(text))
    return text

# parameters must be enhanced if argument parser changes
def main(testament= None):
    # load arguments given by default or within the console like
    args = parser.parse_args()
    #if testament was given from outside it overrides the default parameter of args.testament by the given one
    if testament != None:
        args.testament = testament

    # get arguments given in the console line, indicated by --<pattern> <input>
    # 600 positive bag of words - parsed from:
    # https://www.positivewordslist.com/positive-words-to-describe-personality/
    # https://www.positivewordslist.com/positive-words/
    file = open('pos_bag_of_word.txt', 'r')
    pos_bag_of_words = file.readlines()
    file.close()
    pos_bag_of_words = pos_bag_of_words[0].split(",")

    # 448 negative bag of words parsed from:
    # https://eqi.org/fw_neg.html
    file = open('neg_bag_of_word.txt', 'r')
    neg_bag_of_words = file.readlines()
    file.close()
    neg_bag_of_words = neg_bag_of_words[0].split(",")

    # return the part of the bible desired
    df_bible = dataloader.get_df_bible()
    if args.testament == "new":
        _, df_bible = dataloader.get_old_new_testament(df_bible)
    elif args.testament == "old":
        df_bible, _ = dataloader.get_old_new_testament(df_bible)
    elif args.testament == "both":
        pass
    else:
        print("testament not recognized, continued with the hole bible")

    # ToDo: may be deleted if names are given

    file = open('names.txt', 'r')
    names = file.readlines()
    mock_names = []
    for line in names:
        if len(line.split()) == 1:
            line = line.replace(" ", "")
            line = line.replace("\n", "")
            if line not in mock_names:
                mock_names.append(line)
    file.close()
    df_bible['Characters'] = None
    df_bible['Characters'] = df_bible['Characters'].astype(object)
    df_bible['keywords'] = None
    df_bible['keywords'] = df_bible['keywords'].astype(object)

    file = open('keywords.txt', 'r')
    keywords = file.readlines()
    mock_keywords = []
    for line in keywords:
        if len(line.split()) == 1:
            line = line.replace("\n", "")
            if line not in mock_keywords:
                mock_keywords.append(line)

    # iterate the desired part of the bible
    for i, df_verse in tqdm(df_bible.iterrows()):
        text = df_verse["text"]

        # ToDo: reactivate after character data is given
        # do baysian classification in the text vor being positive / negative
        # Textblob provides in-build classifiers module to create a custom classifier. it classifies the sentence propability to be positive and negative.
        # since we are using [-1, 1] we are negating the negative propability
        neg_score, pos_score = [0, 0]
        #_, neg_score, pos_score = TextBlob(str(text), analyzer=NaiveBayesAnalyzer()).sentiment
        if neg_score < pos_score:
            score_textblob = pos_score
        else:
            score_textblob = -neg_score

        # we check if the verse includes positive or negative words. positive sentences should probably include positive words.
        # check intersection between bag of words and text
        processedSentences = clearText(text)
        set_pos = list(set(processedSentences.split()) & set(pos_bag_of_words))
        set_neg = list(set(processedSentences.split()) & set(neg_bag_of_words))

        if len(set_pos) == len(set_neg):
            score_bag_of_words = 0.0
        elif len(set_pos) > len(set_neg):
            score_bag_of_words = 1.0
        else:
            score_bag_of_words = -1.0

        # find absolute score for the row pos/neg/neutral and update cell emotion in dataframe
        score = score_textblob + score_bag_of_words
        if score > 0.5:
            df_bible.loc[i, "emotion"] = 1.0
        elif score < -0.5:
            df_bible.loc[i, "emotion"] = -1.0
        else:
            df_bible.loc[i, "emotion"] = 0.0


        rnd_numbers = np.random.randint(0, len(mock_keywords), 3)
        subset_keywords = [mock_keywords[idx] for idx in rnd_numbers]
        df_bible.at[i, "keywords"] = subset_keywords
        # ToDO: delete after character data is given and produce new csv

        df_bible.loc[i, "emotion"] = np.random.randint(-4, 4, 1)
        rnd_numbers = np.random.randint(0, 5, 1)
        rnd_idx = np.random.randint(0, len(mock_names), rnd_numbers)
        subset_names = [mock_names[idx] for idx in rnd_idx]
        df_bible.at[i, "Characters"] = subset_names
        #delete until here
    df_bible.to_csv(r'bibleTA_Emotion.csv')


if __name__ == "__main__":
    main()
