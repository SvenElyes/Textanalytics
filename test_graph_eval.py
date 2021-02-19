import unittest
import eval_graph
import pandas as pd
import src.dataloader as dataloader


class Test_graph_eval(unittest.TestCase):
    def test_loadCSV(self):
        print("start testing loadCSV()")
        df_bible = eval_graph.loadCSV()
        df_bible = pd.read_csv("bibleTA_Emotion.csv")
        self.assertEqual(df_bible, df_bible)

        df_bible = eval_graph.loadCSV("both")
        self.assertEqual(df_bible, df_bible)

        df_bible = eval_graph.loadCSV("new")
        self.assertNotEqual(df_bible, df_bible)

        df_bible = eval_graph.loadCSV("old")
        self.assertNotEqual(df_bible, df_bible)

    def test_formatBible(self):
        print("start testing formate_bible()")

        base = pd.DataFrame()
        entry = [{"characters": "[Maria]", "emotion": 1},
                         {"characters": "[Lukas|Jesus|Maria]", "emotion": 1},
                         {"characters": "", "emotion": 1}]
        base.append(entry, ignore_index=True)

        aim = pd.DataFrame()
        entry = [{"character_A": "Lukas", "character_B": "Jesus", "emotion": 1},
                     {"character_A": "Lukas", "character_B": "Maria", "emotion": 1},
                     {"character_A": "Jesus", "character_B": "Maria", "emotion": 1}]
        aim.append(base, ignore_index=True)

        base = eval_graph.formate_bible(df_bible=base)
        self.assertEqual(base, aim)

    def test_distillDataframe(self):
        base = pd.DataFrame()
        entry = {"character_A": "Lukas", "character_B": "Jesus", "emotion": 1}
        base = base.append(entry, ignore_index=True)
        entry = {"character_A": "Jesus", "character_B": "Maria", "emotion": 1}
        base = base.append(entry, ignore_index=True)
        entry = {"character_A": "Lukas", "character_B": "Maria", "emotion": 1}
        base = base.append(entry, ignore_index=True)
        entry = {"character_A": "Jesus", "character_B": "Maria", "emotion": 1}
        base = base.append(entry, ignore_index=True)
        entry = {"character_A": "Lukas", "character_B": "Maria", "emotion": 1}
        base = base.append(entry, ignore_index=True)
        entry = {"character_A": "Jesus", "character_B": "Maria", "emotion": 1}
        base = base.append(entry, ignore_index=True)

        base, _ = eval_graph.distillDataframe(df_bible=base, load=False, threshold=1, save=False)
        aim = pd.DataFrame()
        entry = {"character_A": "Maria", "character_B": "Jesus", "emotion": 1}
        aim = aim.append(entry, ignore_index=True)
        entry = {"character_A": "Maria", "character_B": "Lukas", "emotion": 1}
        aim = aim.append(entry, ignore_index=True)

        self.assertEqual(base, aim)

    def test_plotGraph(edges, color_emotion, label, location, exp):
        pass
    def test_convertToIGraph(dataframe):
        pass
    def test_getGraph(df_bible, load, threshold, testament, location):
        pass
    def test_Cluster2Graph(df_cluster):
        pass
    def test_create_pickle_objects(df_emotion):
        pass
    def test_load_pickle_objects():
        pass
    def test_cluster_data(num_cluster, threshold):
        pass
    def test_getCluster(load, num_cluster, threshold, location):
        pass
    def test_replaceCluster(cluster_log, df_emotion):
        pass
    def test_investigateNeighbor(cluster, cluster_id, neighbor, df_cluster, df_emotion, cluster_log,
                                 found_neighbors):
        pass
    def test_adopt_clusters(df_cluster, df_emotion, min_neighbor_cluster):
        pass
    def test_concat_cluster(df_emotion):
        pass
    def test_check_empty(dataframe):
        pass
    def test_checkFunction(dataframe):
        pass
    def test_adjust_graph(df_cluster, df_emotion, load, location, min_neighbor_cluster):
        pass

class pandas_testing():
    def test_distillDataframe(self):
        base = pd.DataFrame()
        entry = {"character_A": "Lukas", "character_B": "Jesus", "emotion": 1}
        base = base.append(entry, ignore_index=True)
        entry = {"character_A": "Jesus", "character_B": "Maria", "emotion": 1}
        base = base.append(entry, ignore_index=True)
        entry = {"character_A": "Lukas", "character_B": "Maria", "emotion": 1}
        base = base.append(entry, ignore_index=True)
        entry = {"character_A": "Jesus", "character_B": "Maria", "emotion": 1}
        base = base.append(entry, ignore_index=True)
        entry = {"character_A": "Lukas", "character_B": "Maria", "emotion": 1}
        base = base.append(entry, ignore_index=True)
        entry = {"character_A": "Jesus", "character_B": "Maria", "emotion": 1}
        base = base.append(entry, ignore_index=True)

        base, _ = eval_graph.distillDataframe(df_bible=base, load=False, threshold=1, save=False)
        aim = pd.DataFrame()
        entry = {"character_A": "Maria", "character_B": "Jesus", "emotion": 1}
        aim = aim.append(entry, ignore_index=True)
        entry = {"character_A": "Maria", "character_B": "Lukas", "emotion": 1}
        aim = aim.append(entry, ignore_index=True)

        print(pd.testing.assert_series_equal(aim,base,check_dtype=True,

if __name__ == "__main__":
    unittest.main()
    pandas_testing()