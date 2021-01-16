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
        names = row["characters"]
        emotion = row["emotion"]
        names = names.replace("[", "")
        names = names.replace("]", "")
        names = names.replace(",", "|")
        names = names.replace("'", "")
        names = names.rstrip()
        names = names.split('|')
        names_remove = names.copy()

        if len(names) >= 2:
            for name in names:
                for r_name in names_remove:
                    if name != r_name:
                        new_row = {'character_A': name, 'character_B': r_name, 'emotion': emotion}
                        df_bible_formate = df_bible_formate.append(new_row, ignore_index=True)
                names_remove.remove(name)
    print("formated names to list bible")
    return df_bible_formate

def distillDataframe(df_bible, load = False, threshhold=5, testament="new"):
    # create a list of labels (names) which have been detected in both rows character_A and #character_B
    file = "bibleTA_distilled_" + str(testament) + "_" + str(threshhold) + ".csv"
    df_bible = df_bible[['characters', 'emotion']]
    if load == True:
        try:
            df_distilled = pd.read_csv(file)
        except:
            raise ValueError('Could not load file, make sure to following file exists: ' + str(file))

        A = df_distilled['character_A'].unique().tolist()
        B = df_distilled['character_B'].unique().tolist()
        label = A + B
        label = list(set(label))
        try:
            label.remove("")
        except:
            pass
    else:
        df_bible = formate_bible(df_bible)
        A = df_bible['character_A'].unique().tolist()
        B = df_bible['character_B'].unique().tolist()
        label = A + B
        label = list(set(label))
        try:
            label.remove("")
        except:
            pass
        # create output dataframe to be further processed
        df_distilled = pd.DataFrame()
        # iterate over all labels
        # only count in one direction e.g. "character_A" = "Lukas", "character_B"="Jesus" ;
        # do not do a subsequent "character_b" = "Lukas", "character_a"="Jesus" search ; implemented by removal of labels in label_remove list
        label_remove = label.copy()
        for i, character_A in enumerate(label):
            if i % 10 == 0:
                print(str(i) + "/" + str(len(label)))
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
                    if empty_list == False and subset.shape[0] > threshhold:

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

                        # create object from relation class like new_row

                        df_distilled = df_distilled.append(new_row, ignore_index=True)
            label_remove.remove(character_A)
        df_distilled.to_csv(file)
    return df_distilled

# plot the graph based on its edges and edge colors and save it to the given path
def plotGraph(edges, color_emotion, label, location, exp):
    res = []
    label2 = label.copy()
    for a, b in edges:
        if label[a] in label2:
            label2.remove(label[a])
        if label[b] in label2:
            label2.remove(label[b])


    graph = Graph.Graph(n=len(label), edges=edges)
    os.makedirs(location, exist_ok = True)
    out = Graph.plot(graph, vertex_size=10,
                     vertex_color=['white'],
                     vertex_label=label,
                     edge_color=color_emotion)
    out.save(os.path.join(location, exp + '.png'))

# converts the distinct list to nodes and edges. Notes will be our names, which then are converted to a number using a
# dict. Those numbers are translated into edges between character_A and character_B.
def convertToIGraph(df):
    A = df['character_A'].unique().tolist()
    B = df['character_B'].unique().tolist()
    label = A + B
    label = list(set(label))

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

    return edges, color_emotion, label

# load csv in case new textAnalytics outout has been generated. can be set in main.
def loadCSV(path = "", Testament = "new"):
    df_bible = pd.read_csv(os.path.join(path, "bibleTA_Emotion.csv"))

    if Testament == "new":
        first_matthew_verse = df_bible.index[
            (df_bible["book_id"] == "Matt") & (df_bible["verse"] == 1) & (df_bible["chapter"] == 1)].tolist()[0]
        df_bible = df_bible[first_matthew_verse:]

    if Testament == "old":
        first_matthew_verse = df_bible.index[
            (df_bible["book_id"] == "Matt") & (df_bible["verse"] == 1) & (df_bible["chapter"] == 1)].tolist()[0]
        df_bible = df_bible[:(first_matthew_verse - 1)]
    return df_bible

def distill_cluster(cluster):
    # Lukas Jesus
    # Markus Jesus
    # list has to be distinct
    return cluster

def findcluster(keyword_vectors):
    # get vector representation
    # kmeans to get a cluster of k nearest clusters
    cluster = ""
    return cluster

def get_book_relation(df_bible):
    # split df into books
    # aggregate keywords each person in book
    # take top x
    keyword_vectors = ""
    cluster = findcluster(keyword_vectors)
    return cluster

def get_chapter_relation(df_bible):
    # split df into chapters
    # aggregate keywords each person in book
    # take top x
    keyword_vectors = ""
    cluster = findcluster(keyword_vectors)
    distilled_cluster = distill_cluster
    return distilled_cluster

def get_verse_relation(df_bible, load, threshhold, testament):
    # dataframe destillation to get data suitable for graph generation
    df_relation = distillDataframe(df_bible, load, threshhold=threshhold, testament=testament)
    df_relation.to_csv(r'bibleTA_distinct_relation.csv')

    # convert distilled data to nodes and edges. Also generate colored edges
    edges, color_emotion, label = convertToIGraph(df_relation)
    return label, edges, color_emotion

# main function, which calls all the other functions and can be called from outside.
# can be given a dataframe (or it loads one from the folder)
# can also be given load, which loads the last distilled dataframe with distinct character_A to character_B mappings with an aggregated emotion value
def getGraph(df_bible = None, load = False, threshhold = 8, testament = "both"):
    # loads bible dataframe if not given one
    if df_bible == None:
        df_bible = loadCSV(Testament=testament)
        df_bible = pd.DataFrame(data=df_bible)
    else:
        # if a dataframe is given, the dataframe has to be re-distilled
        print("Dataframe will be distilled, because outside df_bible was given. load set to TRUE")
        load = True

    # get relation on verse stage
    label, edges, color_emotion = get_verse_relation(df_bible, load, threshhold, testament)
    # get relation on chapter stage
    get_chapter_relation(df_bible)

    # get relation on book stage
    get_book_relation(df_bible)

    # make and plot graph + save to path
    plotGraph(edges, color_emotion, label, location="images", exp="graph")



if __name__ == "__main__":
    getGraph()

