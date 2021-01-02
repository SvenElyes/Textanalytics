import tqdm
import igraph as Graph
import pandas as pd
import os
import numpy as np

# the dataframe has been preprocessed by many other functions. However we only need a subset of this information to
# create a graph representation of the relations.
# distillDataframe(df_bible) takes the dataframe given to it. It selects
# a) character_A
# b) character_B
# ToDo: take one cell containing a list

def formate_bible(df_bible):
    df_bible_formate = pd.DataFrame()
    for i, row in df_bible.iterrows():
        names = row["Characters"]
        emotion = row["emotion"]
        names = names.replace("[", "")
        names = names.replace("]", "")
        names = names.replace(", ", ",")
        names = names.replace("'", "")
        names = names.split(',')
        names_remove = names.copy()
        for name in names:
            for r_name in names_remove:
                if name != r_name:
                    new_row = {'character_A': name, 'character_B': r_name, 'emotion': emotion}
                    df_bible_formate = df_bible_formate.append(new_row, ignore_index=True)
            names_remove.remove(name)
    df_bible_formate.to_csv(r'bibleTA_Graph.csv')
    return df_bible_formate

def distillDataframe(df_bible, load = False):
    # create a list of labels (names) which have been detected in both rows character_A and #character_B
    df_bible = df_bible[['Characters', 'emotion']]
    if load == True:
        df_bible = pd.read_csv("bibleTA_Graph.csv")
    else:
        df_bible = formate_bible(df_bible)
    A = df_bible['character_A'].unique().tolist()
    B = df_bible['character_B'].unique().tolist()
    label = A + B
    label = list(set(label))
    # create output dataframe to be further processed
    df_destilled = pd.DataFrame()
    # iterate over all labels
    # only count in one direction e.g. "character_A" = "Lukas", "character_B"="Jesus" ;
    # do not do a subsequent "character_b" = "Lukas", "character_a"="Jesus" search ; implemented by removal of labels in label_remove list
    label_remove = label.copy()
    for character_A in label:
        for character_B in label_remove:
            if character_A != character_B:
                # count emotions in both directions

                subset_A = df_bible.loc[(df_bible['character_A'] == character_A) & (df_bible['character_B'] == character_B) & (df_bible['emotion'].notna() == True)]
                subset_B = df_bible.loc[(df_bible['character_A'] == character_B) & (df_bible['character_B'] == character_A) & (df_bible['emotion'].notna() == True)]

                # join both dataframes
                frames = [subset_A, subset_B]
                subset = pd.concat(frames, sort=False)
                # subset = df_bible.loc[(character_A in df_bible['Characters']) & (character_B in df_bible['Characters']) & (df_bible['emotion'].notna() == True)]
                # check if empty
                empty_list = subset.empty
                if empty_list == False:
                    # calculate mean over emotions
                    emotion_mean = np.mean(subset['emotion'])

                    # round it to an absolute emotion (needed for coloring in graph)
                    if emotion_mean > 0.5:
                        emotion_mean = 1.0
                    elif emotion_mean < -0.5:
                        emotion_mean = -1.0
                    else:
                        emotion_mean = 0.0

                    # add new row to the distilled dataframe
                    # df_distilled will have distinct and aggregated emotion rows. From this we can create edge colors and the concrete edges of the graph
                    new_row = {'character_A': character_A, 'character_B': character_B, 'emotion': emotion_mean}
                    df_destilled = df_destilled.append(new_row, ignore_index=True)

        label_remove.remove(character_A)

    return df_destilled, label

# plot the graph based on its edges and edge colors and save it to the given path
def plotGraph(g, color_emotion, label, location, exp):
    os.makedirs(location, exist_ok = True)
    out = Graph.plot(g, vertex_size=20,
                     vertex_color=['white'],
                     # vertex_color=['blue', 'red', 'green', 'yellow'],
                     vertex_label=label,
                     # edge_width=[1, 4])
                     edge_color=color_emotion)
    out.save(os.path.join(location, exp + '.png'))

# converts the distinct list to nodes and edges. Notes will be our names, which then are converted to a number using a
# dict. Those numbers are translated into edges between character_A and character_B.
def convertToIGraph(df, label):
    label2id = {l : i for i, l in enumerate(label)}
    id2label = {i : l for i, l in enumerate(label)}
    #color dict for transfering the emotion score to a colored edge
    color_dict = {1: "green", -1: "red", 0: "black"}

    edges = []
    color_emotion = []

    for i, df_verse in df.iterrows():
        A = label2id[df_verse["character_A"]]
        B = label2id[df_verse["character_B"]]
        relation = df_verse["emotion"]
        edges.append([A, B])
        color_emotion.append(color_dict[relation])

    return edges, color_emotion

# load csv in case new textAnalytics outout has been generated. can be set in main.
def loadCSV(path = ""):
    df = pd.read_csv(os.path.join(path, "bibleTA_Emotion.csv"))
    return df

# main function, which calls all the other functions and can be called from outside.
# can be given a dataframe (or it loads one from the folder)
# can also be given load, which loads the last distilled dataframe with distinct character_A to character_B mappings with an aggregated emotion value
def getGraph(df_bible = None, load = True):
    # loads bible dataframe if not given one
    if df_bible == None:
        df_bible = loadCSV()
        df_bible = pd.DataFrame(data=df_bible)

    # if a dataframe is given, the dataframe has to be re-distilled
    if df_bible != None:
        print("Dataframe will be distilled, because outside df_bible was given. load set to TRUE")
        load = True

    # dataframe destillation to get data suitable for graph generation
    df_relation, label = distillDataframe(df_bible, load)

    # convert distilled data to nodes and edges. Also generate colored edges
    edges, color_emotion = convertToIGraph(df_relation, label)

    # make and plot graph + save to path
    g = Graph.Graph(n=len(label), edges=edges)
    plotGraph(g, color_emotion, label, path="images", exp="graph")

if __name__ == "__main__":
    getGraph(load= True)

