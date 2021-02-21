from unittest import TestCase
from unittest.mock import patch
from unittest.mock import MagicMock
from src.dataloader import get_old_new_testament, get_df_bible
import pandas as pd
from pandas._testing import assert_frame_equal


class TestDataloader(TestCase):
    def test_get_df_bible(self):
        with patch("src.dataloader.pd") as mock_pd:
            mock_pd.read_csv.return_value = pd.read_csv("test/test_bible_4_verse.csv")

            df_bible = pd.read_csv("test/test_bible_4_verse.csv")
            df_bible.drop(["Unnamed: 0"], axis=1, inplace=True)

            assert_frame_equal(get_df_bible(), df_bible)

    def test_get_old_new_testament(self):
        df_bible = pd.read_csv("test/test_whole_bible.csv")
        df_bible.drop(["Unnamed: 0"], axis=1, inplace=True)
        old_t, new_t = get_old_new_testament(df_bible)

        df_new_t = pd.read_csv("test/test_new_testament.csv")
        df_new_t.drop(["Unnamed: 0"], axis=1, inplace=True)
        assert_frame_equal(old_t, df_new_t)
