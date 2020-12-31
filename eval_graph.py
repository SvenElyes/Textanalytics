import tqdm
import igraph as Graph
import pandas as pd
import os
import numpy as np

def distillDataframe(df_bible):
    # hier muss das dataset von Character list, emotion in
    # A: character, B: character, C: emotion
    # danach muss emotion aggregiert werden um die emotion zwischen den Charakteren zu mitteln,
    # sodass A: character, B: character, C: emotion eine unique combination in einem undirected graph ist.
    # wenn keine doppelkanten oder kanten von und nach einem selbst, okay

    #print(df_bible.head(5))
    df_bible = df_bible[['book_name', 'book_id', 'emotion']]
    A = df_bible['book_name'].unique().tolist()
    B = df_bible['book_id'].unique().tolist()
    label = A + B
    label = list(set(label))
    label_remove = label

    df_destilled = pd.DataFrame(
        columns=["character_a", "character_b", "emotion"]
    )
    #print(list(df_bible.columns.values))
    for character_A in label:
        for character_B in label_remove:
            subset_A = df_bible.loc[(df_bible['book_name'] == character_A) & (df_bible['book_id'] == character_B) & (df_bible['emotion'].notna() == True)]
            subset_B = df_bible.loc[(df_bible['book_name'] == character_B) & (df_bible['book_id'] == character_A) & (df_bible['emotion'].notna() == True)]
            '''
            subset_A = df_bible.loc[
                (df_bible['book_name'] == "Matt") & (df_bible['book_id'] == "Matthew") & (
                            df_bible['emotion'].notna() == True)]
            subset_B = df_bible.loc[
                (df_bible['book_name'] == "Matthew") & (df_bible['book_id'] == "Matt") & (
                            df_bible['emotion'].notna() == True)]
            '''

            frames = [subset_A, subset_B]
            subset = pd.concat(frames, sort=False)
            empty_list = subset.empty

            if empty_list == False:
                emotion_mean = np.mean(subset['emotion'])
                if emotion_mean > 0.5:
                    emotion_mean = 1.0
                elif emotion_mean < -0.5:
                    emotion_mean = -1.0
                else:
                    emotion_mean = 0.0

                new_row = {'character_a': character_A, 'character_b': character_B, 'emotion': emotion_mean}
                df_destilled = df_destilled.append(new_row, ignore_index=True)

        label_remove.remove(character_A)
    print(df_destilled.head(5))
    return df_destilled, label

def plotGraph(g, color_emotion, label, location, exp):
    os.makedirs(location, exist_ok = True)
    out = Graph.plot(g, vertex_size=20,
                     vertex_color=['white'],
                     # vertex_color=['blue', 'red', 'green', 'yellow'],
                     vertex_label=label,
                     # edge_width=[1, 4])
                     edge_color=color_emotion)
    out.save(os.path.join(location, exp + '.png'))

def convertToIGraph(df, label):
    print(label)
    label2id = {label[i] : i for i in range(0, len(label))}
    id2label = {i : label[i] for i in range(0, len(label))}
    color_dict = {1: "green", -1: "red", 0: "black"}

    edges = []
    color_emotion = []
    for i, df_verse in df.iterrows():
        A = label2id[df_verse["character_a"]]
        B = label2id[df_verse["character_b"]]
        relation = df_verse["emotion"]
        edges.append([A, B])
        color_emotion.append(color_dict[relation])

    return edges, color_emotion

def loadCSV(path = ""):
    df = pd.read_csv(os.path.join(path, "bibleTA_Emotion.csv"))
    return df

df_bible = loadCSV()
df_bible = pd.DataFrame(data=df_bible)
#print(df_bible.head(3))

df_relation, label = distillDataframe(df_bible)

# df_relation = {'A': ['Josef', 'Maria', 'Maria Magdalena', 'Jesus'], 'B': ['Jesus', 'Jesus', 'Jesus', 'Jesus'],  'emotion': [1, 0, -1, 0]}
# df_relation = pd.DataFrame(data=df_relation)

edges, color_emotion = convertToIGraph(df_relation, label)

# label = ['Josef', 'Maria', 'Jesus', 'Heiliger Geist', 'Gott', 'Maria Magdalena', 'Lukas', 'Matthäus', 'Petrus', 'Judas']
# edges = [[0, 1], [2, 3], [8, 9], [7, 8], [6, 8], [7, 6], [6, 9], [7, 9], [6, 2], [7, 2], [8, 2], [9, 2], [0, 2], [1, 2], [4, 2], [4, 3], [5, 2],[5, 0],[5, 1]]

g = Graph.Graph(n=len(label), edges=edges)
plotGraph(g, color_emotion, label, "images", "graph")



