import tqdm
import igraph as Graph
import pandas as pd
import os
import numpy as np
import spacy
from sklearn.cluster import KMeans
from pylab import *
from scipy.ndimage import measurements
import re
import time

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
        names = names.strip()
        names = names.replace("\n", "")
        names = names.split('|')
        names_remove = names.copy()

        if len(names) >= 2:
            for name in names:
                for r_name in names_remove:
                    if name != r_name:
                        new_row = {'character_A': name.strip(), 'character_B': r_name.strip(), 'emotion': emotion}
                        df_bible_formate = df_bible_formate.append(new_row, ignore_index=True)
                names_remove.remove(name)
    print("formated names to list bible")
    return df_bible_formate

def distillDataframe(df_bible, load = False, threshhold=2, testament="new"):
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
                        # sort names alphabetically, will later be "exploited" while shrinking the graph
                        new_row = {'character_A': min(character_A, character_B), 'character_B': max(character_A, character_B), 'emotion': emotion_mean}

                        # create object from relation class like new_row

                        df_distilled = df_distilled.append(new_row, ignore_index=True)
            label_remove.remove(character_A)
        df_distilled.to_csv(file)
    return df_distilled, label


# plot the graph based on its edges and edge colors and save it to the given path
def plotGraph(edges, color_emotion, label, location, exp):
    graph = Graph.Graph(n=len(label), edges=edges)
    os.makedirs(location, exist_ok=True)

    if color_emotion == []:
        out = Graph.plot(graph, vertex_size=10,
                         vertex_color=['white'],
                         vertex_label=label)
        out.save(os.path.join(location, exp + '.png'))
    else:
        out = Graph.plot(graph, vertex_size=10,
                         vertex_color=['white'],
                         vertex_label=label,
                         edge_color=color_emotion)
        out.save(os.path.join(location, exp + '.png'))

# converts the distinct list to nodes and edges. Notes will be our names, which then are converted to a number using a
# dict. Those numbers are translated into edges between character_A and character_B.
def convertToIGraph(dataframe):
    A = dataframe['character_A'].unique().tolist()
    B = dataframe['character_B'].unique().tolist()
    label = A + B
    label = list(set(label))

    label2id = {l : i for i, l in enumerate(label)}
    id2label = {i : l for i, l in enumerate(label)}

    #color dict for transfering the emotion score to a colored edge
    color_dict = {1: "green", -1: "red", 0: "black"}

    edges = []
    color_emotion = []

    for i, df_verse in dataframe.iterrows():
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
    df_relation, label_list = distillDataframe(df_bible, load, threshhold=threshhold, testament=testament)
    df_relation.to_csv(r'bibleTA_distinct_relation.csv')

    # convert distilled data to nodes and edges. Also generate colored edges
    edges, color_emotion, label = convertToIGraph(df_relation)
    return label, edges, color_emotion, df_relation, label_list

# main function, which calls all the other functions and can be called from outside.
# can be given a dataframe (or it loads one from the folder)
# can also be given load, which loads the last distilled dataframe with distinct character_A to character_B mappings with an aggregated emotion value
def getGraph(df_bible, load, threshhold, testament, exp_name):
    # loads bible dataframe if not given one
    if df_bible == None:
        df_bible = loadCSV(Testament=testament)
        df_bible = pd.DataFrame(data=df_bible)
    else:
        # if a dataframe is given, the dataframe has to be re-distilled
        print("Dataframe will be distilled, because outside df_bible was given. load set to TRUE")
        load = False

    # get relation on verse stage
    label, edges, color_emotion, df_relation, label_list = get_verse_relation(df_bible, load, threshhold, testament)
    # get relation on chapter stage
    get_chapter_relation(df_bible)

    # get relation on book stage
    get_book_relation(df_bible)

    # make and plot graph + save to path
    plotGraph(edges, color_emotion, label, location=exp_name, exp="emotion_graph")
    return df_relation, label_list

def Cluster2Graph(df_cluster):
    A = df_cluster['person'].unique().tolist()
    B = df_cluster['cluster'].unique().tolist()
    label = A + B
    label = list(set(label))

    label2id = {l : i for i, l in enumerate(label)}
    id2label = {i : l for i, l in enumerate(label)}

    edges = []

    for i, df_verse in df_cluster.iterrows():
        A = label2id[df_verse["person"]]
        B = label2id[df_verse["cluster"]]
        edges.append([A, B])
    return edges, label

def mock_names_keywords(label_list, mock_words):
    file = open('keywords.txt', 'r')
    keywords = file.readlines()
    file.close()
    keywords = [keyword.replace("\n", "") for keyword in keywords]

    res = []
    temp_res = []
    for name in label_list:
        for idx in range(mock_words):
            check = True
            while check:
                rnd_num = np.random.randint(0, len(keywords),1)[0]
                name = keywords[rnd_num]
                if name not in temp_res:
                    temp_res.append(keywords[rnd_num])
                    check = False
        res.append(temp_res)
        temp_res = []

    return label_list, res

def cluster_data(num_cluster, people_cluster, threshold, label_list, mock_words):

    characters, res = mock_names_keywords(label_list, mock_words)
    # extract distinct keywords to convert them to word-vectors and afterwards determine clusters
    distinct_res = []
    for keywords_res in res:
        for keyword in keywords_res:
            if keyword not in distinct_res:
                distinct_res.append(keyword)


    # load spaCy's word2vec
    nlp = spacy.load("en_core_web_lg")

    # vectorize the list of disinct keywords
    vectorized_distinct_res = np.empty((len(distinct_res), 300))
    for i, keyword in enumerate(distinct_res):
        word_vector = nlp(keyword).vector
        vectorized_distinct_res[i, :] = word_vector

    # cluster word vectors in "num_cluster" cluster
    kmeans = KMeans(n_clusters=num_cluster)
    clustered_words = kmeans.fit_predict(vectorized_distinct_res)

    # dict that gives cluster to word, to then convert the keywords of character to keywords in cluster
    # e.g. Jesus, keywords = ["Lord", "raised from dead", "son", "god"]
    # into: Jesus, clustered_res = [1, 0, 4, 2]
    keyword2cluster = {keyword : cluster for keyword, cluster in zip(distinct_res, clustered_words)}
    clustered_res = np.empty_like(res)

    for i, keywords_res in enumerate(res):
        for ii, keyword in enumerate(keywords_res):
            #print(keyword)
            label = keyword2cluster[keyword]
            clustered_res[i, ii] = label

    clustered_res = clustered_res.astype(np.float)

    # threshold from which there are enough keywords from the same cluster to form an edge within a graph
    count_threshold = clustered_res.shape[1] * threshold
    df_cluster = pd.DataFrame()

    # for all cluster (e.g. 4 --> 0, 1, 2, 3)
    for cluster in range(num_cluster):
        # count how often this cluster occurred in the the clusterized keywords (clustered_res == cluster) each person (axis = 1)
        result = np.count_nonzero(clustered_res == cluster, axis=1)
        # make node for graph (may later be turned into a cloud if possible)
        cluster_name = "cluster" + str(cluster)
        for i, count in enumerate(result):
            # if keywords from same cluster have occurred more often that threshold says, they are added as edge to the graph
            if count > count_threshold:
                # append to dataframe
                new_row = {'person': characters[i], 'cluster': cluster_name}
                df_cluster = df_cluster.append(new_row, ignore_index=True)

    df_cluster.to_csv("clustered_keywords.csv")
    return df_cluster

def getCluster(load, mock_words, num_cluster, people_cluster, threshold, label_list, exp_name):
    # if load = True, load pre-evaluated csv file
    if load == False:
        # from character with keywords to a dataframe that shows edges, where more keywords from one cluster have occurred than threshold says
        df_cluster = cluster_data(num_cluster=num_cluster, people_cluster=people_cluster, threshold=threshold, label_list=label_list, mock_words=mock_words)
    else:
        df_cluster = pd.read_csv("clustered_keywords.csv")
    # convert edges to nummeric edges and prepare node labels
    edges, label = Cluster2Graph(df_cluster)
    # plot the graph
    plotGraph(edges, [], label, location=exp_name, exp="cluster_graph")
    return df_cluster

# obsolet
def cluster2graph(df_cluster):
    clusters = df_cluster['cluster'].unique().tolist()

    df_graph = pd.DataFrame()

    for cluster in clusters:
        characters_cluster = [name for name in df_cluster.loc[df_cluster['cluster'] == cluster]['person']]
        characters_cluster_remove = characters_cluster.copy()
        for character in characters_cluster:
            for character_remove in characters_cluster_remove:
                new_row = {'character_A': character, 'character_B': character_remove}
                df_graph.append(new_row, ignore_index=True)
            character_remove.remove(character)

    return df_graph
# wenn relation gefunden wird, df_cluster.loc[i, "character_A"] = cluster
# zum vorgänger, dann recursiv aufrufen.

def investigateNeighbor(person, cluster, cluster_id, neighbor, df_cluster, df_emotion):
    # probe if the node has neigbhor nodes
    subset_A = df_emotion.loc[df_emotion['character_A'] == neighbor]['character_B']
    subset_B = df_emotion.loc[df_emotion['character_B'] == neighbor]['character_A']
    frames = [subset_A, subset_B]
    new_neighbors = pd.concat(frames, sort=False).unique().tolist()

    person = person.strip()
    cluster = cluster.strip()
    neighbor = neighbor.strip()

    try:
        new_neighbors.remove(person)
    except:
        pass
    # if yes, probe if those nodes are also in the cluster
    if len(new_neighbors) > 0:
        for ii, new_neighbor in enumerate(new_neighbors):
            new_neighbor = new_neighbor.strip()
            check_cluster = df_cluster.loc[(df_cluster['cluster'] == cluster) & (df_cluster['person'] == new_neighbor)]
            empty_list = check_cluster.empty

            #if yes, apply cluster to graph
            if empty_list == False:
                # first delete the row from cluster_frame
                check_cluster = df_cluster.loc[(df_cluster['cluster'] == cluster) & (df_cluster['person'] == person)]
                df_cluster = df_cluster.drop(check_cluster.index)

                # second, change the person name to cluster_id
                if min(person, neighbor) == person:
                    row = df_emotion.loc[(df_emotion['character_A'] == person) & (df_emotion['character_B'] == neighbor)]
                    df_emotion.loc[row.index, 'character_A'] = cluster_id
                else:
                    row = df_emotion.loc[(df_emotion['character_A'] == neighbor) & (df_emotion['character_B'] == person)]
                    df_emotion.loc[row.index, 'character_B'] = cluster_id

                investigateNeighbor(neighbor, cluster, cluster_id, new_neighbor, df_cluster, df_emotion)
    return df_emotion

def adopt_clusters(df_cluster, df_emotion):
    clusters = df_cluster['cluster'].unique().tolist()

    for cluster in clusters:
        characters_in_cluster = df_cluster.loc[df_cluster['cluster'] == cluster]['person']
        characters_in_cluster = [person for person in characters_in_cluster]
        cluster = cluster.strip()

        for idx_p, cluster_person in enumerate(characters_in_cluster):
            person = cluster_person.strip()

            subset_A = df_emotion.loc[df_emotion['character_A'] == cluster_person]
            subset_B = df_emotion.loc[(df_emotion['character_B'] == cluster_person)]
            frames = [subset_A, subset_B]
            neighbors = pd.concat(frames, sort=False)

            empty_list = neighbors.empty
            if empty_list == False:
                new_neighbors = [new_n for new_n in neighbors['character_B']]
                for ii, new_neighbor in enumerate(new_neighbors):
                    new_neighbor = new_neighbor.strip()
                    cluster_id = str(cluster) + "_" + str(idx_p)
                    investigateNeighbor(person, cluster, cluster_id, new_neighbor, df_cluster, df_emotion)

    for i, row in df_emotion.iterrows():
        if row['character_A'] == row['character_B']:
            df_emotion.drop(df_cluster.index[i])

    return df_emotion

def distill_shrunken_df(df_emotion):
    # create a list of labels (names) which have been detected in both rows character_A and #character_B
    A = df_emotion['character_A'].unique().tolist()
    B = df_emotion['character_B'].unique().tolist()
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

                subset_A = df_emotion.loc[(df_emotion['character_A'] == character_A) & (df_emotion['character_B'] == character_B) & (df_emotion['emotion'].notna() == True)]
                subset_B = df_emotion.loc[(df_emotion['character_A'] == character_B) & (df_emotion['character_B'] == character_A) & (df_emotion['emotion'].notna() == True)]

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
                    # sort names alphabetically, will later be "exploited" while shrinking the graph
                    new_row = {'character_A': min(character_A, character_B), 'character_B': max(character_A, character_B), 'emotion': emotion_mean}

                    # create object from relation class like new_row

                    df_distilled = df_distilled.append(new_row, ignore_index=True)
        label_remove.remove(character_A)
    return df_distilled

def concat_cluster(distilled_df, df_emotion):
    A = df_emotion['character_A'].unique().tolist()
    B = df_emotion['character_B'].unique().tolist()
    label = A + B
    label = list(set(label))
    try:
        label.remove("")
    except:
        pass

    label_wo_cluster = [l for l in label if "cluster" not in l]
    label_wo_cluster_remove = label_wo_cluster.copy()

    df_concated_cluster = pd.DataFrame()

    for i, lwc in enumerate(label_wo_cluster):
        if i % 10 == 0:
            print(str(i) + "/" + str(len(label)))

        subset_A = df_emotion.loc[(df_emotion['character_B'] == lwc) & (df_emotion['character_A'].str.contains(r"cluster[0-9]+", regex=True))]
        subset_B = df_emotion.loc[(df_emotion['character_A'] == lwc) & (df_emotion['character_B'].str.contains(r"cluster[0-9]+", regex=True))]
        frames = [subset_A, subset_B]
        subset = pd.concat(frames, sort=False)


        # subset = df_bible.loc[(character_A in df_bible['Characters']) & (character_B in df_bible['Characters']) & (df_bible['emotion'].notna() == True)]
        # check if empty
        empty_list = subset.empty
        if empty_list == False and subset.shape[0] > 1:
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
            # sort names alphabetically, will later be "exploited" while adaopting clusters to the graph
            cluster_name = "concat_cl_" + str(i)
            new_row = {'character_A': min(lwc, cluster_name), 'character_B': max(lwc, cluster_name), 'emotion': emotion_mean}

            # create object from relation class like new_row

            df_concated_cluster = df_concated_cluster.append(new_row, ignore_index=True)

        # we dont just want the clusters
        # but also those, which are between 2 person, non-clustered

    for i, lwc in enumerate(label_wo_cluster):
        if i % 10 == 0:
            print(str(i) + "/" + str(len(label)))
        for ii, lwc_r in enumerate(label_wo_cluster_remove):
            subset_A = df_emotion.loc[(df_emotion['character_A'] == lwc) & (df_emotion['character_B'] == lwc_r)]
            subset_B = df_emotion.loc[(df_emotion['character_A'] == lwc_r) & (df_emotion['character_B'] == lwc)]
            frames = [subset_A, subset_B]
            subset = pd.concat(frames, sort=False)

            # check if empty
            empty_list = subset.empty
            if empty_list == False:
                df_concated_cluster = df_concated_cluster.append(subset, ignore_index=True)

        label_wo_cluster_remove.remove(lwc)
    return df_concated_cluster

def adjust_graph(df_cluster, df_emotion, load):
    if load == True:
        try:
            dataframe = pd.read_csv("bibleTA_clustered_concat.csv")
        except:
            raise ValueError(
                'Could not load file, make sure to following file exists: ' + str("bibleTA_clustered_concat.csv"))

    else:
        dataframe = adopt_clusters(df_cluster=df_cluster, df_emotion=df_emotion)
        dataframe = distill_shrunken_df(df_emotion=dataframe)
        dataframe = concat_cluster(distilled_df=dataframe, df_emotion=df_emotion)
        dataframe.to_csv("bibleTA_clustered_concat.csv")

    return dataframe

def main():
    exp_name = time.strftime("%Y%m%d_%H%M%S", time.gmtime(time.time())) + "_images"
    df_emotion, label_list = getGraph(df_bible=None, load=True, threshhold=6, testament="new", exp_name=exp_name)
    df_cluster = getCluster(load=False, mock_words=5, num_cluster=8, people_cluster=3, threshold=(1/3), label_list=label_list, exp_name=exp_name)

    dataframe = adjust_graph(df_cluster=df_cluster, df_emotion=df_emotion, load=False)

    edges, color_emotion, label = convertToIGraph(dataframe=dataframe)
    plotGraph(edges, color_emotion, label, location="exp_name", exp="clustered_emotion_graph")

if __name__ == "__main__":
    main()

