import re
#import pandas as pd
from pattern.text.en import singularize
import spacy, en_core_web_sm
import textacy
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from tqdm import tqdm
import argparse

import src.dataloader as dataloader

parser = argparse.ArgumentParser(description='train models for matching tasks with GT correspondences')
parser.add_argument('--testament', type=str, default="new", metavar='X', help='{both, old, mew}')
args = parser.parse_args()


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
def preText(text):
    # lowers the text
    text = text.lower()
    # recognize verb pattern
    pattern = [{"POS": "VERB", "OP": "*"}, {"POS": "ADV", "OP": "*"}, {"POS": "VERB", "OP": "+"},
               {"POS": "PART", "OP": "*"}]

    #nlp = en_core_web_sm.load()

    #extract verb pattern
    doc = textacy.make_spacy_doc(text, lang='en_core_web_sm')
    verbs = textacy.extract.matches(doc, pattern)
    for verb in verbs:
        #singularize verb, e.g. "likes" to "like"
        text = text.replace(verb.text, singularize(verb.text))

    return text

def clearText(text):
    # removing paragraph numbers
    text = re.sub('[0-9]+.\t', '', str(text))

    #remove punctuations
    text = re.sub('[.!?]', '', str(text))

    # removing new line characters
    text = re.sub('\n', '', str(text))

    # removing apostrophes
    text = re.sub("'s", '', str(text))

    # removing hyphens
    text = re.sub("-", ' ',str(text))
    text = re.sub("— ", '',str(text))

    # removing quotation marks
    text = re.sub('\"','',str(text))

    # removing space after sentence
    text = re.sub('\. ', '.', str(text))
    return text

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

# iterate the desired part of the bible
for i, df_verse in tqdm(df_bible.iterrows()):
    text = df_verse["text"]

    # do baysian classification in the text vor being positive / negative
    _, neg_score, pos_score = TextBlob(str(text), analyzer=NaiveBayesAnalyzer()).sentiment
    if neg_score < pos_score:
        score_textblob = pos_score
    else:
        score_textblob = neg_score

    # check intersection between bag of words and text
    processedSentences = clearText(text)
    if list(set(processedSentences.split()) & set(pos_bag_of_words)) != []:
        score_bag_of_words = 1.0
    elif list(set(processedSentences.split()) & set(neg_bag_of_words)) != []:
        score_bag_of_words = -1.0
    else:
        score_bag_of_words = 0.0

    score = score_textblob + score_bag_of_words
    if score > 0.5:
        df_bible.loc[i, "Emotion"] = 1.0
    elif score < -0.5:
        df_bible.loc[i, "Emotion"] = -1.0
    else:
        df_bible.loc[i, "Emotion"] = 0.0

df_verse.to_csv(r'bibleTA_Emotion.csv')





