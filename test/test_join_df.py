import unittest
import pandas as pd
import os
import pytest
import numpy as np

os.chdir('../')

from pandas.testing import assert_frame_equal

import src.join_df as join_df

class Test_join_df(unittest.TestCase):
    def test_main(self):
        emotion = pd.read_csv("test/csv/join_emotion_test.csv")
        characters = pd.read_csv("test/csv/join_characters_test.csv")
        res = join_df.main("", "", "csv/out.csv", emotion, characters)
        out = pd.read_csv("test/csv/join_result_test.csv")
        out.sort_index(axis=1) == res.sort_index(axis=1)
        out = out[['Unnamed: 0', 'bow_emotion', 'emotion', 'similarity_emotion', 'tb_emotion', 'text', 'characters']]
        assert_frame_equal(res, out)
