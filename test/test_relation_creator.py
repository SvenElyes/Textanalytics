from unittest.mock import patch
from unittest import TestCase
from unittest.mock import MagicMock
from src.relation_creator import create_char_relation, create_character_keywords
import pandas as pd
from src.data import character


class TestRelationcreator(TestCase):
    def test_create_char_relation(self):

        with patch("pickle.dump") as mock_dump:
            df_dis = pd.read_csv("test/csv_test/test_distilled.csv")
            create_char_relation(df_dis)
            self.assertEqual(
                mock_dump.call_count, 2
            )  # GETS CALLED TWICE. ONCE IN each override at the end of the function

    def test_create_character_keywords(self):
        # mock picklehanlder here again

        c1, c2 = (character.Character("Sven"), character.Character("Felix"))

        listofc = (c1, c2)

        with patch("src.dataloader.pd") as mock_pd:
            with patch("src.pickle_handler.pickle") as mock_pickle:
                with patch("src.keyword_extractor") as mock_kw:

                    mock_pd.read_csv.return_value = pd.read_csv(
                        "test/csv_test/test_keyword_extractor.csv"
                    )

                    mock_pickle.load.return_value = listofc
                    create_character_keywords()

                    mock_pickle.dump.assert_called_once()
                    """ gets called in the end of the function, in picklehandler.save_override_character_list(character_list)

                    # further checking, to see if the characters have the keywords seem to be too dificutl, because we are not having a
                    # return value for the functions. Checking with the assert_called_once_with() seems to complex, bc we have to mimic the other parameters
                    # Creating a new pickle file and comparing to the one, that would be created also seems illogical"""
