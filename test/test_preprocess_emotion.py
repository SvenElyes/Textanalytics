import unittest
import pandas as pd
import os
import pytest
import numpy as np


from pandas.testing import assert_frame_equal

import src.preprocess_emotion as preprocess_emotion


class Test_preprocess_emotion(unittest.TestCase):
    def test_clearText(self):
        test = "this IS just love \n going to be a testcase - AMEN."
        out = preprocess_emotion.clearText(test)
        check = "this is just love going to be a testcase amen"
        self.assertEqual(out, check)

    def test_main(self):
        prepro_in = pd.read_csv("test/csv/prepro_in.csv")
        df_bible = preprocess_emotion.main("", prepro_in, "test/csv/out.csv")
        # dropped from evaluation because it uses randomness
        df_bible.drop(["similarity_emotion"], axis=1, inplace=True)
        prepro_out = pd.read_csv("test/csv/prepro_out.csv")
        prepro_out = prepro_out[
            ["Unnamed: 0", "characters", "text", "tb_emotion", "bow_emotion", "emotion"]
        ]
        assert_frame_equal(prepro_out, df_bible)

    def test_preText(self):
        test = "Jesus likes all the people"
        bow_pos = [
            "love",
            "love",
            "love",
            "love",
            "love",
            "like",
            "love",
            "love",
            "love",
            "love",
            "love",
        ]
        bow_neg = ["bad", "bad", "bad", "bad", "bad", "bad", "bad", "bad", "bad", "bad"]
        # score_word_sim not evaluated because of random choises
        _, score_bow = preprocess_emotion.preText(test, bow_pos, bow_neg)
        emotion_out = 1.0
        self.assertEqual(emotion_out, score_bow)

    def test_wordSimilarity(self):
        bow_pos = [
            "love",
            "love",
            "love",
            "love",
            "love",
            "love",
            "love",
            "love",
            "love",
            "love",
            "love",
        ]
        bow_neg = ["bad", "bad", "bad", "bad", "bad", "bad", "bad", "bad", "bad", "bad"]

        verb = "romance"
        out = preprocess_emotion.wordSimilarity(bow_pos, bow_neg, verb)
        out_prejected = 1.0
        self.assertEqual(out, out_prejected)
