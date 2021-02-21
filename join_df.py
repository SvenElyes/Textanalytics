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
import pandas as pd
from spacy.tokenizer import Tokenizer
from nltk.tokenize import word_tokenize
#tokenizer = Tokenizer(nlp.vocab)
from spacy.lang.en.stop_words import STOP_WORDS
import argparse
parser = argparse.ArgumentParser(description='Find style templates')
parser.add_argument('--character', type=str, default='', help='name of characters csv file')
parser.add_argument('--emotion', type=str, default='', help='name of emotion csv file')
parser.add_argument('--out', type=str, default='bibleTA_emotion.csv', help='output name of csv file')
args = parser.parse_args()

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
    text = re.sub("-", ' ', str(text))
    text = re.sub("— ", '', str(text))

    # removing quotation marks
    text = re.sub('\"', '', str(text))

    # removing space after sentence
    text = re.sub('\. ', '.', str(text))

    return text

def preText(text, pos_bow, neg_bow):
    # parameter:
    # text : takes a sentence, string

    # return:
    # text: the taxt with singularized verbs, string
    # emotion : simularity based emotion

    # lowers the text
    text = text.lower()
    pos_res = []
    neg_res = []
    #text = tokenizer(text)
    print(text)
    for word in text.split():
        print(word)
        pos, neg = wordSimularity(pos_bow, neg_bow, word)
        if pos*2 >= neg or -neg*2 > pos:
             pos_res.append(pos)
             neg_res.append(neg)

    pos_res = np.mean(pos_res)
    neg_res = np.mean(neg_res)

    if pos_res > 0.5:
        emotion = 1.0
    elif neg_res > -0.5:
        emotion = -1.0
    else:
        emotion = 0.0

    return emotion
# measures a similarity between a verb and 10 random from the pos/neg bag of words
def wordSimularity(pos_bow, neg_bow, verb):
    # parameter:
    # pos_bow: bag of positive words, list
    # neg_bow: bag of negative words, list
    # verb: given verb, string

    # return:
    # pos_res: means over positive words, float
    # neg_res: mean over negative words, float

    file = open('pos_bag_of_word.txt', 'r')
    pos_bow = file.readlines()
    file.close()
    pos_bow = pos_bow[0].split(",")

    # 448 negative bag of words parsed from:
    # https://eqi.org/fw_neg.html
    file = open('neg_bag_of_word.txt', 'r')
    neg_bow = file.readlines()
    file.close()
    neg_bow = neg_bow[0].split(",")

    neg_res = []
    pos_res = []

    # get english language library of spacy
    nlp = en_core_web_sm.load()
    # get token of verb
    token_word = nlp(verb)
    # take 10 random words from list
    rnd_pos = np.random.randint(0, len(neg_bow), 55)
    for rnd in rnd_pos:
        pos = pos_bow[rnd]
        if nlp.vocab.has_vector(pos) == True:
            pos_nlp = nlp(pos)
            # get similarity
            similarity_emotion = token_word.similarity(pos_nlp)
            pos_res.append(similarity_emotion)
    pos_res = np.mean(pos_res)
    
    rnd_neg = np.random.randint(0, len(neg_bow), 50)
    for rnd in rnd_neg:
        neg = neg_bow[rnd]
        if nlp.vocab.has_vector(neg) == True:
            neg_nlp = nlp(neg)
            # get similarity
            similarity_emotion = token_word.similarity(neg_nlp)
            neg_res.append(similarity_emotion)
    neg_res = np.mean(neg_res)          
    return pos_res, neg_res

def main(character, emotion, out):
    if args.emotion == "":
        args.emotion = emotion
    if args.character == "":
        args.character = character
    if args.out == "":
        args.out = out

    file = open('pos_bag_of_word.txt', 'r')
    pos_bow = file.readlines()
    file.close()
    pos_bow = pos_bow[0].split(",")

    # 448 negative bag of words parsed from:
    # https://eqi.org/fw_neg.html
    file = open('neg_bag_of_word.txt', 'r')
    neg_bow = file.readlines()
    file.close()
    neg_bow = neg_bow[0].split(",")

    df_bible = pd.read_csv("bibleTA_Emotion_fromServer.csv")
    df_characters = pd.read_csv("bibleTA_characters_2102.csv")
    #df_bible['characters'] = df_bible['characters'].astype(object)
    names_list = []
    emotion = []
    for i, (df_b_verse, df_c_verse) in tqdm(enumerate(zip(df_bible.iterrows(), df_characters.iterrows()))):
        text = df_b_verse[1]["text"]
        text = clearText(text)
        #print(df_c_verse["characters"])
        score_bow = df_bible.loc[i, "bow_emotion"]
        #print(score_bag_of_words)
        score_textblob = df_bible.loc[i, "tb_emotion"]
        #print(score_textblob)
        #simularity_emotion = preText(text, pos_bow, neg_bow)
        # adds score to dataframe
        #df_bible.loc[i, "simularity_emotion"] = simularity_emotion
        score = score_bow + score_textblob \
        #+ simularity_emotion
        if score > 0.75:
            emotion.append(1.0)
        elif score < -0.75:
            emotion.append(-1.0)
        else:
            emotion.append(0.0)

        if not pd.isna(df_characters.loc[i, "characters"]):
            #input(df_characters.loc[i, "characters"])
            names = df_characters.loc[i, "characters"]
            #names = names.replace("|", ";")
            names = "[" + str(names) + "]"
        else:
            names = "[]"
        names_list.append(names)

    df_bible["characters"] = names_list
    df_bible["emotion"] = emotion
    df_bible.to_csv(args.out)

if __name__ == "__main__":
    main(character="bibleTA_characters.csv", emotion="bibleTA_Emotion_fromServer.csv", out="bibleTA_emotion.csv")
