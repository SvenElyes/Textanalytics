import re
from pattern.text.en import singularize
import textacy
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from tqdm import tqdm
import src.dataloader as dataloader
import spacy
import numpy as np
import pandas as pd
import os

# try to singularize verbs to make them comparable to pos/neg bag of words.
# is done to detect "goes" as "go" and "likes" as "like".
# ToDo: reduce past tense!
# https://medium.com/python-in-plain-english/text-classification-using-python-spacy-7a414abcc83a
# bag should be done better!

def preText(text, pos_bow, neg_bow):
    # parameter:
    # text : takes a sentence, string
    # pos_bow: positive bag of words, list
    # neg_bow: negative bag of words, list

    # return:
    # score_word_sim : similarity score for all verbs, float
    # score_bow: score for bag of words implementation, float

    # recognize verb pattern
    pattern = [{"POS": "VERB", "OP": "*"}, {"POS": "ADV", "OP": "*"}, {"POS": "VERB", "OP": "+"},
               {"POS": "PART", "OP": "*"}]

    # extract verb pattern
    doc = textacy.make_spacy_doc(text, lang='en_core_web_lg')
    verbs = textacy.extract.matches(doc, pattern)
    score_word_sim = 0.0
    score_bow = 0.0
    for verb in verbs:
        # singularize verb, e.g. "likes" to "like"
        singularized_verb = singularize(verb.text)
        score_word_sim += wordSimilarity(pos_bow, neg_bow, singularized_verb)
        # apply bag of words to the singularized verb
        score_bow += pos_bow.count(str(singularized_verb))
        score_bow -= neg_bow.count(str(singularized_verb))


    # aggregate all verb similarity
    if score_word_sim > 0.5:
        score_word_sim = 1.0
    elif score_word_sim < -0.5:
        score_word_sim = -1.0
    else:
        score_word_sim = 0.0

    # aggregate the count with bag of words
    if score_bow > 0.5:
        score_bow = 1.0
    elif score_bow < -0.5:
        score_bow = -1.0
    else:
        score_bow = 0.0

    return score_word_sim, score_bow
# measures a similarity between a verb and 10 random from the pos/neg bag of words
def wordSimilarity(pos_bow, neg_bow, verb):
    # parameter:
    # pos_bow: bag of positive words, list
    # neg_bow: bag of negative words, list
    # verb: given verb, string

    # return:
    # score_word_sim: similarity score for spacy word similarity to 10 random words from bag of words pos/neg, float

    # get english language library of spacy
    nlp = spacy.load('en_core_web_lg')
    # get token of verb
    token_word = nlp(verb)
    # take 10 random words from list
    sub_neg_bow = [neg_bow[i] for i in np.random.randint(0, len(neg_bow), 10)]
    sub_pos_bow = [pos_bow[i] for i in np.random.randint(0, len(pos_bow), 10)]

    # tokenize these words
    nlp_sub_neg_res = []
    nlp_sub_neg_res = [nlp(neg) for neg in sub_neg_bow if nlp.vocab.has_vector(neg) == True]

    nlp_sub_pos_res = []
    nlp_sub_pos_res = [nlp(pos) for pos in sub_pos_bow if nlp.vocab.has_vector(pos) == True]

    # measure similarity and mean their result
    similarity_emotion = 0.0
    for token_bow in nlp_sub_pos_res:
        # get similarity
        similarity_emotion += token_word.similarity(token_bow)
    for token_bow in nlp_sub_neg_res:
        # get similarity
        similarity_emotion -= token_word.similarity(token_bow)

    #agregate single word similarity
    if similarity_emotion > 0.5:
        score_word_sim = 1.0
    elif similarity_emotion < -0.5:
        score_word_sim = -1.0
    else:
        score_word_sim = 0.0

    return score_word_sim

# needed to clear the text and proces it by bag of words
def clearText(text):
    # parameter:
    # text : takes a sentence, string

    # return:
    # text: returns the pre processed text, string
    # removing paragraph numbers
    text = re.sub('[0-9]+.\t', '', str(text))

    # remove punctuations
    text = re.sub('[.!?]', '', str(text))

    # removing new line characters
    text = re.sub('\n', '', str(text))

    # removing apostrophes
    text = re.sub("'s", '', str(text))

    # removing hyphens
    text = re.sub("-", '', str(text))
    text = re.sub("— ", '', str(text))

    # removing quotation marks
    text = re.sub('\"', '', str(text))

    # removing space after sentence
    text = re.sub('\. ', '.', str(text))

    # lowers the text
    text = text.lower()

    # reduce multiple blank spaces
    text = " ".join(text.split())

    return text

def loadBible(testament):
    # return the part of the bible desired
    df_bible = dataloader.get_df_bible()
    if testament == "new":
        _, df_bible = dataloader.get_old_new_testament(df_bible)
    elif testament == "old":
        df_bible, _ = dataloader.get_old_new_testament(df_bible)
    else:
        print("testament not recognized, continued with the hole bible")
    return df_bible
# parameters must be enhanced if argument parser changes
# saves a dataframe of its results
# returns the converted dataframe
def main(testament, df_bible, out):
    # get arguments given in the console line, indicated by --<pattern> <input>
    # 600 positive bag of words - parsed from:
    # https://www.positivewordslist.com/positive-words-to-describe-personality/
    # https://www.positivewordslist.com/positive-words/
    # parameter:
    # testament: determines which testament should be evaluated new/old/else
    # df_bible: if dataframe is given from outside, pandas dataframe
    # out: dir, to save the file to, string

    # return
    # df_bible: pandas dataframe, added with emotions

    print("python3 preprocess_emotion.py")
    print("Started pre-processing the dataframe to evaluate the relation information")

    file = open('src/pos_bag_of_word.txt', 'r')
    pos_bow = file.readlines()
    file.close()
    pos_bow = pos_bow[0].split(",")

    # 448 negative bag of words parsed from:
    # https://eqi.org/fw_neg.html
    file = open('src/neg_bag_of_word.txt', 'r')
    neg_bow = file.readlines()
    file.close()
    neg_bow = neg_bow[0].split(",")

    if not isinstance(df_bible, pd.DataFrame):
        df_bible = loadBible(testament)

    # for every verse
    for i, df_verse in tqdm(df_bible.iterrows()):
        text = df_verse["text"]

        # do baysian classification in the text vor being positive / negative
        # Textblob provides in-build classifiers module to create a custom classifier. it classifies the sentence propability to be positive and negative.
        # since we are using [-1, 1] we are negating the negative propability
        _, p_pos, p_neg = TextBlob(str(text), analyzer=NaiveBayesAnalyzer()).sentiment
        if p_pos > 0.5:
            score_textblob = 1.0
        elif p_neg > 0.5:
            score_textblob = -1.0
        else:
            score_textblob = 0
        df_bible.loc[i, "tb_emotion"] = score_textblob

        # we check if the verse includes positive or negative words. positive sentences should probably include positive words.
        # check intersection between bag of words and text
        processedSentences = clearText(text)
        # sigularize verbs and check their similarity with 10 random words of the pos_bow and neg_bow
        similarity_emotion, score_bow = preText(processedSentences, pos_bow, neg_bow)
        # adds score to dataframe
        df_bible.loc[i, "similarity_emotion"] = similarity_emotion

        # get intersection of words in verse and bag of words pos/neg
        set_pos = list(set(processedSentences.split()) & set(pos_bow))
        set_neg = list(set(processedSentences.split()) & set(neg_bow))

        df_bible.loc[i, "bow_emotion"] = score_bow

        # fill aggregated row; from here later calculations will be determined
        score = score_bow + score_textblob + similarity_emotion
        if score > 0.5:
            df_bible.loc[i, "emotion"] = 1.0
        elif score < -0.5:
            df_bible.loc[i, "emotion"] = -1.0
        else:
            df_bible.loc[i, "emotion"] = 0.0
    df_bible.to_csv(out)
    print("Finished pre-processing the dataframe")
    return df_bible

if __name__ == "__main__":
    main("both", None, "csv/bibleTA_prepro.csv")

