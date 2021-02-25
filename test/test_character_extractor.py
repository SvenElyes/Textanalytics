from unittest import TestCase
import pandas as pd
from pandas._testing import assert_frame_equal
import spacy
import os
import neuralcoref
import en_core_web_sm

os.chdir('../')

import src.character_extractor as character_extractor
import src.dataloader as dataloader

#Functions that are not tested here, are being called by at least one of the tested functions

df_bible = dataloader.get_df_bible()
df_test = df_bible.head(42)


class TestUserFunctions(TestCase):

    def test_coreference_resolution_on_different_parameters(self):
        df_resolved = character_extractor.coreference_resolution(df_test)
        df_resolved_start_20 = character_extractor.coreference_resolution(df_test, start=20)
        df_resolved_invalid_1 = character_extractor.coreference_resolution(df_test, start=20, end=10)
        df_resolved_invalid_2 = character_extractor.coreference_resolution(df_test, start=50)
        df_resolved_invalid_3 = character_extractor.coreference_resolution(df_test, end=50)

        self.assertEqual(df_resolved.columns, df_resolved_start_20.columns)
        self.assertEqual(df_resolved.columns, df_resolved_invalid_1.columns)

        self.assertIsNotNone(df_resolved["resolved_text"][0])
        self.assertIsNone(df_resolved_start_20["resolved_text"][0])
        self.assertIsNone(df_resolved_invalid_1["resolved_text"][0])
        self.assertIsNone(df_resolved_invalid_1["resolved_text"][0])
        self.assertIsNone(df_resolved_invalid_1["resolved_text"][0])

        assert_frame_equal(df_resolved_invalid_1, df_resolved_invalid_2)
        assert_frame_equal(df_resolved, df_resolved_invalid_3)

    def test_extract_characters_output_frame(self):
        df_characters = character_extractor.extract_characters(df_test)

        self.assertIn("characters", df_characters.columns)
        self.assertEqual("God", df_characters["characters"][0])
    
    def test_extract_characters_without_ruler(self):
        df_characters = character_extractor.extract_characters(df_test, rule_based_matching=False)
        df_wrong_param_1 = character_extractor.extract_characters(df_test, use_bible_patterns=False)
        df_wrong_param_2 = character_extractor.extract_characters(df_test, rule_based_matching=False)
        df_wrong_param_3 = character_extractor.extract_characters(df_test, rule_based_matching=False, patterns = [{"label": "PERSON", "pattern": "God"}])

        self.assertIn("characters", df_characters.columns)
        self.assertIsNone(df_characters["characters"][0])
        assert_frame_equal(df_characters, df_wrong_param_1)
        assert_frame_equal(df_characters, df_wrong_param_2)
        assert_frame_equal(df_characters, df_wrong_param_3)
    
    def test_extract_characters_own_patterns(self):
        df_characters = character_extractor.extract_characters(df_test, use_bible_patterns=False, patterns=[{"label": "PERSON", "pattern": "earth"}])

        self.assertIn("characters", df_characters.columns)
        self.assertEqual("earth", df_characters["characters"][0])


if __name__=="__main__":
    unittest.main()