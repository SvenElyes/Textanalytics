import re
from pattern.text.en import singularize
import textacy
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from tqdm import tqdm
import src.dataloader as dataloader
import spacy
import numpy as np

# try to singularize verbs to make them comparable to pos/neg bag of words.
# is done to detect "goes" as "go" and "likes" as "like".
# ToDo: reduce past tense!
# https://medium.com/python-in-plain-english/text-classification-using-python-spacy-7a414abcc83a
# bag should be done better!

def preText(text, pos_bow, neg_bow):
    # parameter:
    # text : takes a sentence, string

    # return:
    # text: the taxt with singularized verbs, string
    # emotion : simrlarity based emotion

    # lowers the text
    text = text.lower()
    # recognize verb pattern
    pattern = [{"POS": "VERB", "OP": "*"}, {"POS": "ADV", "OP": "*"}, {"POS": "VERB", "OP": "+"},
               {"POS": "PART", "OP": "*"}]

    # extract verb pattern
    doc = textacy.make_spacy_doc(text, lang='en_core_web_lg')
    verbs = textacy.extract.matches(doc, pattern)
    pos_res = []
    neg_res = []
    for verb in verbs:
        # singularize verb, e.g. "likes" to "like"
        singularized_verb = singularize(verb.text)
        pos, neg = wordSimilarity(pos_bow, neg_bow, singularized_verb)
        pos_res.append(pos)
        neg_res.append(neg)
        text = text.replace(verb.text, singularized_verb)

    pos_res = np.mean(pos_res)
    neg_res = np.mean(neg_res)

    if pos_res > 0.5:
        emotion = 1.0
    elif neg_res > -0.5:
        emotion = -1.0
    else:
        emotion = 0.0

    return text, emotion
# measures a similarity between a verb and 10 random from the pos/neg bag of words
def wordSimilarity(pos_bow, neg_bow, verb):
    # parameter:
    # pos_bow: bag of positive words, list
    # neg_bow: bag of negative words, list
    # verb: given verb, string

    # return:
    # pos_res: means over positive words, float
    # neg_res: mean over negative words, float

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
    pos_res = []
    for token_bow in nlp_sub_pos_res:
        # get similarity
        similarity_emotion = token_word.similarity(token_bow)
        pos_res.append(similarity_emotion)
    pos_res = np.mean(pos_res)

    # measure similarity and mean their result
    neg_res = []
    for token_bow in nlp_sub_neg_res:
        # get similarity
        similarity_emotion = token_word.similarity(token_bow)
        neg_res.append(similarity_emotion)
    neg_res = np.mean(neg_res)

    return pos_res, neg_res

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
    text = re.sub("-", ' ', str(text))
    text = re.sub("— ", '', str(text))

    # removing quotation marks
    text = re.sub('\"', '', str(text))

    # removing space after sentence
    text = re.sub('\. ', '.', str(text))

    return text

# parameters must be enhanced if argument parser changes
# saves a dataframe of its results
# returns the converted dataframe
def main(testament):
    # get arguments given in the console line, indicated by --<pattern> <input>
    # 600 positive bag of words - parsed from:
    # https://www.positivewordslist.com/positive-words-to-describe-personality/
    # https://www.positivewordslist.com/positive-words/
    # parameter:
    # testament: determines which testament should be evaluated new/old/else

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

    # return the part of the bible desired
    df_bible = dataloader.get_df_bible()
    if testament == "new":
        _, df_bible = dataloader.get_old_new_testament(df_bible)
    elif testament == "old":
        df_bible, _ = dataloader.get_old_new_testament(df_bible)
    else:
        print("testament not recognized, continued with the hole bible")
    # for every verse
    for i, df_verse in tqdm(df_bible.iterrows()):
        text = df_verse["text"]

        # do baysian classification in the text vor being positive / negative
        # Textblob provides in-build classifiers module to create a custom classifier. it classifies the sentence propability to be positive and negative.
        # since we are using [-1, 1] we are negating the negative propability
        _, neg_score, pos_score = TextBlob(str(text), analyzer=NaiveBayesAnalyzer()).sentiment
        if pos_score > 0.5:
            score_textblob = 1.0
        elif neg_score > 0.5:
            score_textblob = -1.0
        else:
            score_textblob = 0
        df_bible.loc[i, "tb_emotion"] = score_textblob

        # we check if the verse includes positive or negative words. positive sentences should probably include positive words.
        # check intersection between bag of words and text
        processedSentences = clearText(text)
        # sigularize verbs and check their similarity with 10 random words of the pos_bow and neg_bow
        text, simrlarity_emotion = preText(processedSentences, pos_bow, neg_bow)
        # adds score to dataframe
        df_bible.loc[i, "similarity_emotion"] = simrlarity_emotion

        # get intersection of words in verse and bag of words pos/neg
        set_pos = list(set(processedSentences.split()) & set(pos_bow))
        set_neg = list(set(processedSentences.split()) & set(neg_bow))

        # find absolute score for the row pos/neg/neutral and update cell emotion in dataframe
        if len(set_pos) == len(set_neg):
            score_bow = 0.0
        elif len(set_pos) > len(set_neg):
            score_bow = 1.0
        else:
            score_bow = -1.0
        df_bible.loc[i, "bow_emotion"] = score_bow

        # fill aggregated row; from here later calculations will be determined
        score = score_bow + score_textblob + simrlarity_emotion
        if score > 0.5:
            df_bible.loc[i, "emotion"] = 1.0
        elif score < -0.5:
            df_bible.loc[i, "emotion"] = -1.0
        else:
            df_bible.loc[i, "emotion"] = 0.0

    df_bible.to_csv('csv/bibleTA_prepro.csv')
    print("Finished pre-processing the dataframe")
    return df_bible

if __name__ == "__main__":
    main("both")

