import unittest
import pandas as pd
import os
import pytest
import numpy as np

os.chdir('../')

from pandas.testing import assert_frame_equal

import src.eval_graph as eval_graph


class Test_graph_eval(unittest.TestCase):
    def test_loadCSV(self):
        function_return = eval_graph.loadCSV("both", "src/csv/bibleTA_emotion.csv")
        base = pd.read_csv("src/csv/bibleTA_emotion.csv")

        assert_frame_equal(function_return, base)

    def test_formatBible(self):
        df_bible = pd.read_csv("test/csv/test_eval_graph.csv")
        function_return = eval_graph.formate_bible(df_bible)
        base = pd.read_csv("test/csv/test_return_formatBible.csv")
        base.drop(["Unnamed: 0"], axis=1, inplace=True)
        assert_frame_equal(base, function_return)

    def test_distillDataframe(self):
        df_bible = pd.read_csv("test/csv/test_eval_graph.csv")
        formated_bible = eval_graph.formate_bible(df_bible)
        function_return, label, load = eval_graph.distillDataframe(formated_bible, False, 1, False)
        #function_return.sort_values(by=['character_A'])

        base = pd.read_csv("test/csv/test_return_distillBible.csv")
        base.drop(["Unnamed: 0"], axis=1, inplace=True)
        #base.sort_values(by=['character_A'])
        assert_frame_equal(base, function_return)

    def test_dataframe2graph(self):
        df_bible = pd.read_csv("test/csv/test_eval_graph.csv")
        formated_bible = eval_graph.formate_bible(df_bible)

        function_return, label, load = eval_graph.distillDataframe(formated_bible, False, 1, False)
        edges, color_emotion, label = eval_graph.dataframe2graph(function_return, True, "character_A", "character_B", "emotion")
        mock_colors = ['red', 'black']
        mock_edges = [[0, 2], [1, 2]]

        self.assertEqual(mock_colors, color_emotion)
        self.assertEqual(edges, mock_edges)

    def test_check_empty(self):
        empty = pd.DataFrame()
        res = eval_graph.check_empty(empty)
        self.assertEqual(res, True)

        new_row = {"person": "John", "cluster": "cluster1"}
        empty = empty.append(new_row, ignore_index=True)
        res = eval_graph.check_empty(empty)
        self.assertEqual(res, False)

    def test_checkFunction(self):
        empty = pd.DataFrame()
        with pytest.raises(Exception) as execinfo:
            eval_graph.checkFunction(empty)

        assert str(execinfo.value) == "dataframe is empty"

        new_row = {"person": "John", "cluster": "cluster1"}
        empty = empty.append(new_row, ignore_index=True)
        eval_graph.checkFunction(empty)
        # if nothing happens no exception has been raised, which is right

    def test_adopt_clusters(self):
        df_incluster = pd.read_csv("test/csv/test_return_incluster.csv")
        df_incluster.drop(["Unnamed: 0"], axis=1, inplace=True)
        df_bible = pd.read_csv("test/csv/test_eval_graph.csv")
        formated_bible = eval_graph.formate_bible(df_bible)
        function_return, label, load = eval_graph.distillDataframe(formated_bible, False, 0, False)
        df_emotion = eval_graph.adopt_clusters(df_incluster, function_return, 0)
        res_cluster = pd.read_csv("test/csv/test_cluster_return.csv")

        res_cluster.drop(["Unnamed: 0"], axis=1, inplace=True)
        res_cluster.reset_index(inplace=True)
        res_cluster.drop(['index'], axis=1, inplace=True)

        df_emotion.reset_index(inplace=True)
        df_emotion.drop(['index'], axis=1, inplace=True)

        assert_frame_equal(df_emotion, res_cluster)


    def test_concat_cluster(self):
        df_cluster = pd.read_csv("test/csv/test_return_concat_clusters.csv")

        df_emotion = eval_graph.concat_cluster(df_cluster)
        df_emotion.drop(["Unnamed: 0"], axis=1, inplace=True)
        df_emotion.reset_index(inplace=True)
        df_emotion.drop(['index'], axis=1, inplace=True)

        df_res = pd.read_csv("test/csv/test_return_concat_res.csv")
        df_res.drop(["Unnamed: 0"], axis=1, inplace=True)
        df_res.reset_index(inplace=True)
        df_res.drop(['index'], axis=1, inplace=True)

        assert_frame_equal(df_emotion, df_res)

if __name__ == "__main__":
    unittest.main()
