from unittest import mock
from unittest.mock import patch
from unittest.mock import MagicMock
from unittest import TestCase

from src.pickle_handler import PickleHandler
from src.data import character
import pandas as pd


"""Because the relationa and character load functions are the same we only test the character one."""


class testPickleHandler(TestCase):
    def test_load_characters(self):
        ph = PickleHandler()
        with patch("src.pickle_handler.pickle") as mock_pickle:
            ph.load_characters()
            mock_pickle.load.assert_called_once()

    def test_load_character_by_name(self):
        ph = PickleHandler()
        c1, c2, c3 = (
            character.Character("Sven"),
            character.Character("Felix"),
            character.Character("Johannes"),
        )
        ph.characters = (c1, c2, c3)

        self.assertEqual(c1, ph.load_character_by_name("Sven"))

    def test_save_character(self):
        ph = PickleHandler()
        with patch("src.pickle_handler.pickle") as mock_pickle:
            c1 = character.Character("Sven")
            ph.save_character(c1)
            self.assertEqual(ph.characters, [c1])
            mock_pickle.dump.assert_called_once()

    def test_save_character_list(self):
        ph = PickleHandler()

        with patch("src.pickle_handler.pickle") as mock_pickle:
            c1, c2 = character.Character("Sven"), character.Character("Felix")
            ph.save_character_list((c1, c2))
            self.assertEqual(ph.characters, [c1, c2])
            mock_pickle.dump.assert_called_once()

    def test_save_override_character_list(self):
        ph = PickleHandler()

        with patch("src.pickle_handler.pickle") as mock_pickle:
            c1, c2 = character.Character("Sven"), character.Character("Felix")
            ph.save_override_character_list((c1, c2))
            self.assertEqual(ph.characters, (c1, c2))
            mock_pickle.dump.assert_called_once()

            """issue: In this case we self assert with () and not with []. I really want to fix this, but there is so much code already depending on this,
            that I cant change it, so early before the deadline"""

    def test_get_character_by_name_from_list(self):
        ph = PickleHandler()
        with patch("src.pickle_handler.pickle") as mock_pickle:
            c1, c2 = character.Character("Sven"), character.Character("Felix")

            self.assertEqual(ph.get_character_by_name_from_list((c1, c2), "Sven"), c1)
