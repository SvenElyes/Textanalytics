import unittest
import pandas as pd
import os
import pytest
import numpy as np
from pandas.testing import assert_frame_equal

import src.join_df as join_df


class Test_join_df(unittest.TestCase):
    def test_main(self):

        emotion = pd.read_csv("test/csv_test/join_emotion_test.csv")
        characters = pd.read_csv("test/csv_test/join_characters_test.csv")
        res = join_df.main("", "", "test/csv_test/out.csv", emotion, characters)
        out = pd.read_csv("test/csv_test/join_result_test.csv")
        out.sort_index(axis=1) == res.sort_index(axis=1)
        out = out[
            [
                "Unnamed: 0",
                "bow_emotion",
                "emotion",
                "similarity_emotion",
                "tb_emotion",
                "text",
                "characters",
            ]
        ]

        # assert_frame_equal(res, out)
