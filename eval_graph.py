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
# distillDataframe() takes the dataframe given to it. It selects
# a) character_A
# b) character_B

# formate_bible should transform all rows, that contain >= 2 characters into multiple rows between all characters
# from [lukas, mark, maria] to [[lukas, mark],[lukas, maria], [maria, mark]]

# Parameter
# df_bible : expects a pandas dataframe that consists of "characters" and "emotion" column

# Return
# df_bible_formate : pandas dataframe, that consists of 3 columns "character_A", "character_B", "emotion"


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

# distillDataframe should turn the dataframe to distinct rows, which have been aggregated in terms of
# their emotion. The dataframe needs to be aggregated because characters may occur at multiple verses
# to not list those multiple times within an graph and to given an more "global" represenation of their
# emotional state the emotion is aggregated. If the emotion_mean > 0.75, the relation is considered to be positive
# if the emotion_mean < -0.75 the relation is considered to be negative.
# elsewise it is neutral. The relation will later be used to project an color to the graph.


# Parameter
# df_bible : pandas dataframeof the bible
# load : determines if a csv should be loaded or if one has to be produced by the function, bool
# threshhold : counts how often relations should occur before being considered reasonable i.e. one time mentions may not be displayed, integer

# Return
# df_distilled :  pandas dataframe consistent of distinct relations
# label : unique list of all characters

def distillDataframe(df_bible, load, threshhold):
    # create a list of labels (names) which have been detected in both rows character_A and #character_B
    file = "bibleTA_distilled_" + "_" + str(threshhold) + ".csv"
    if load == True:
        try:
            df_distilled = pd.read_csv(file)
        except:
            raise ValueError('Could not load file, make sure to following file exists: ' + str(file))
        # get list of unique characters
        A = df_distilled['character_A'].unique().tolist()
        B = df_distilled['character_B'].unique().tolist()
        label = A + B
        label = list(set(label))
        try:
            label.remove("")
        except:
            pass
    else:
        # get list of unique characters
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
            if (i+1) % 10 == 0:
                print(str(i+1) + "/" + str(len(label)))
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
                        if emotion_mean > 0.75:
                            emotion_mean = 1.0
                        elif emotion_mean < -0.75:
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

        A = df_distilled['character_A'].unique().tolist()
        B = df_distilled['character_B'].unique().tolist()
        label = A + B
        label = list(set(label))

        df_distilled.to_csv(file)
    return df_distilled, label


# plotGraph plots the graph based on its edges and edge colors and save it to the given path
# first it builds the graph based on its nodes and edges.
# than it tries to create the experiment folder, where it than saves the graph plot
# if no relation information is given in color_emotion all edges will be black.

# Parameter:
# edges : numerical edges which are mapped by the dict to its label, numpy array
# color_emotion : based on "emotion" column a color is chosen to represent the kind of relation, list
#                   black: neutral, red: bad, green: good
# label : unique list of all characters
# location : place to save the experiment to, string
# exp : name of the experiment, string

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
# Parameters:
# dataframe : pandas dataframe consistent of labels and relation between characters

# Return
# edges : numpy array which transfers labels to numerical values
# color_emotion : list of colors in the same length as edges has rows
# label : unique list of labels which index matches the edges

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
# load the bible as csv and can differentiate between the old an the new testament

# Parameter:
# testament : "new", "old", string

# Return
# df_bible : pandas dataframe which contains the bible loaded from csv file

def loadCSV(testament):
    df_bible = pd.read_csv("bibleTA_Emotion.csv")

    if testament == "new":
        first_matthew_verse = df_bible.index[
            (df_bible["book_id"] == "Matt") & (df_bible["verse"] == 1) & (df_bible["chapter"] == 1)].tolist()[0]
        df_bible = df_bible[first_matthew_verse:]

    if testament == "old":
        first_matthew_verse = df_bible.index[
            (df_bible["book_id"] == "Matt") & (df_bible["verse"] == 1) & (df_bible["chapter"] == 1)].tolist()[0]
        df_bible = df_bible[:(first_matthew_verse - 1)]
    return df_bible

# sub function which calls the individual functions

# Parameter:
# df_bible : pandas dataframe loaded from csv or given from outside
# load : wether or not to load the processed dataframe, bool
# threshhold : how often do relations have to appear, int

# Return
# label : list of distinct characters by convertToIGraph()
# edges : relations displayed as numpy array
# color_emotion : list of emotions with the same length as edges is depth
# df_relation : hand through the relation pandas dataframe, which will be needed to later apply the clusters
# label_list : label list concerning df_relations, list of strings
def get_verse_relation(df_bible, load, threshhold):
    # get the distinct relations as a dataframe
    df_relation, label_list = distillDataframe(df_bible, load, threshhold=threshhold)

    # convert distilled data to nodes and edges. Also generate colored edges
    edges, color_emotion, label = convertToIGraph(df_relation)
    return label, edges, color_emotion, df_relation, label_list

# main function, which calls all the other functions and can be called from outside.
# can be given a dataframe (or it loads one from the folder)
# can also be given load, which loads the last distilled dataframe with distinct character_A to character_B mappings with an aggregated emotion value

# Parameter:
# df_bible : pandas dataframe, may be given from outside
# load : load calculations from previous run
# threshhold : counts the encounterments of two characters in one verse, int
# testament : "old", "new", else both testaments, string
# exp_name : name of the experiment, string

# Return:
# df_relation : pandas dataframe of the relations, dataframe consistes of ["character_A", "character_B", "emotion"]
# label_list : unique list of characters in dataframe


def getGraph(df_bible, load, threshhold, testament, exp_name):
    # loads bible dataframe if not given one
    if df_bible == None:
        df_bible = loadCSV(testament=testament)
        df_bible = pd.DataFrame(data=df_bible)
    else:
        # if a dataframe is given, the dataframe has to be re-distilled
        print("Dataframe will be distilled, because outside df_bible was given. load set to TRUE")
        load = False

    # get relation on verse stage
    label, edges, color_emotion, df_relation, label_list = get_verse_relation(df_bible, load, threshhold)

    # make and plot graph + save to path
    plotGraph(edges, color_emotion, label, location=exp_name, exp="1_emotion_graph")
    return df_relation, label_list

# Cluster2Graph is used to display the clusters in the graph to the dataframe, such that the clusters may be visual
# Parameter:
# df_cluster : pandas dataframe which contains the cluster of all persons
# Return:
# edges: numpy array of numerical edges
# label : index matches numerical values of edges, list of node names (Names and Cluster)

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


# cluster keywords and create list of people in this cluster that have threshold enough keywords coming from the same cluster
# Parameter:
# num_cluster: number of cluster centroid - results in labels for keywords, int
# threshold: min count of cluster label to app person to cluster, int
# label_list: may be removed
# mock_words: may be removed

# return:
# df_cluster: pandas dataframe, consistent of cluster name and character

def cluster_data(num_cluster, threshold, label_list, mock_words):

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

# getCluster loads the clusters to a dataframe. Either from csv file or by calling cluster_data()
# dataframe is then prepared to be displayed as a graph and subsequent plotted
# parameter:
# load: if true, load data from csv file, bool
# num_cluster: number of cluster centroid - results in labels for keywords, int
# threshold: min count of cluster label to app person to cluster, int
# label_list: may be removed
# mock_words: may be removed
# exp_name: name of the experiment to save the plot, string

def getCluster(load, mock_words, num_cluster, threshold, label_list, exp_name):
    # if load = True, load pre-evaluated csv file
    if load == False:
        # from character with keywords to a dataframe that shows edges, where more keywords from one cluster have occurred than threshold says
        df_cluster = cluster_data(num_cluster=num_cluster, threshold=threshold, label_list=label_list, mock_words=mock_words)
        empty_list = df_cluster.empty
        # if yes, apply cluster to graph
        if empty_list == True:
            print('Was not able to find any cluster using the given setting.')
    else:
        df_cluster = pd.read_csv("clustered_keywords.csv")
    # convert edges to nummeric edges and prepare node labels
    edges, label = Cluster2Graph(df_cluster)
    # plot the graph
    plotGraph(edges, [], label, location=exp_name, exp="2_cluster_graph")
    return df_cluster

# apply the found clusters in the graph to the actual dataframe
# practically difficult because one character can be in multiple clusters
# returns a new dataframe that considers the previous dataframe

###########################################################################################################

# Idea:

# from character: Jesus
# has neighbors: cluster1, cluster2, Lukas
# has clusters: cluster6, cluster4
# result: cluster6-cluster1, cluster6-cluster2, cluster6-Lukas, cluster4-cluster1, cluster4-cluster2, cluster4-lukas

# or, if character has no cluster:
# from character: Jesus
# has neighbors: cluster1, cluster2, Lukas
# has clusters: Jesus <-- has the character itself as cluster
# result: Jesus-cluster1, Jesus-cluster2, Jesus-Lukas, Jesus-cluster1, Jesus-cluster2, Jesus-lukas

###########################################################################################################

# parameter:
# cluster_log: pandas dataframe that hold all character, that are in cluster and can be found in graph
# df_emotion: bible dataframe, has 3 rows: character_A, character_B and emotion

# return:
# df_emotion_cluster: new bible dataframe, including the clusters, also in format: character_A, character_B and emotion


def replaceCluster(cluster_log, df_emotion):
    # remove dublicates in log
    cluster_log = cluster_log.drop_duplicates()

    # get all characters in the bible dataframe
    A = df_emotion['character_A'].unique().tolist()
    B = df_emotion['character_B'].unique().tolist()
    label = A + B
    label = list(set(label))
    df_label = pd.DataFrame(label, columns=['node'])

    # return dataframe initial
    df_emotion_cluster = pd.DataFrame()

    # for all characters in the bible
    for i, row in df_label.iterrows():
        character = row['node']
        clusters = cluster_log.loc[cluster_log['from'] == character]['to'].values

        # get all neighbors to the current character
        subset_A = df_emotion.loc[df_emotion['character_A'] == character]['character_B']
        subset_B = df_emotion.loc[df_emotion['character_B'] == character]['character_A']
        frames = [subset_A, subset_B]
        neighbors = pd.concat(frames, sort=False).tolist()

        # dataframe, which will either hold the character or the cluster(s) that the character will be replaced with; in idea called "has neighbors:"
        for n in neighbors:
            # get the emotion of the current row
            emotion = df_emotion.loc[(df_emotion['character_A'] == character) & (df_emotion['character_B'] == n)]
            empty_list = emotion.empty
            if empty_list == True:
                emotion = df_emotion.loc[(df_emotion['character_B'] == character) & (df_emotion['character_A'] == n)]
            emotion = emotion.values[0][3]

            # find clusters the character is in, if empty, add character itself to neighbor_cluster
            check_cluster = cluster_log.loc[cluster_log['from'] == n]
            empty_list = check_cluster.empty
            if empty_list == False:
                # for all clusters (may be multiple), in idea called "has clusters:"
                # add rows to the dataframe
                for i, cl in check_cluster.iterrows():
                    clu = cl['to']
                    cluster_entry = {'character_A': clu.strip(),
                                     'character_B': character.strip(),
                                     'emotion': emotion}
                    df_emotion_cluster = df_emotion_cluster.append(cluster_entry, ignore_index=True)
            else:
                # if character has no cluster
                cluster_entry = {'character_A': n.strip(),
                                 'character_B': character.strip(),
                                 'emotion': emotion}
                df_emotion_cluster = df_emotion_cluster.append(cluster_entry, ignore_index=True)


    df_emotion_cluster = df_emotion_cluster.drop_duplicates()
    return df_emotion_cluster

# recursive call to do a depth first search
# is given a person which is in cluster x and checks every relation/neighbor node if this node is also in the cluster
# if so, the previous node is added to the cluster / marked "in cluster"
# parameter:
# cluster: current cluster the function should find neighbors in, string
# cluster_id: cluster may be found at multiple places in graph which
#               do not need to overlap; therefore each root-cluster has an id
# neighbor: neighbor that should be investigated by the function, string
# df_cluster: person in cluster dataframe, that should be searched for in the graph, pandas dataframe
# df_emotion: bible dataframe, has 3 rows: character_A, character_B and emotion
# cluster_log: pandas dataframe that should be enhanced by clusters located in the graph
# found_neighbors: already investigated neighbors; should not investigate them multiple times, string array

# return:
# cluster_log: pandas dataframe that hold all character, that are in cluster and can be found in graph

def investigateNeighbor(cluster, cluster_id, neighbor, df_cluster, df_emotion, cluster_log, found_neighbors):
    # probe if the node has neigbhor nodes

    subset_A = df_emotion.loc[(df_emotion['character_A'] == neighbor) & (~df_emotion['character_B'].isin(found_neighbors))]['character_B']
    subset_B = df_emotion.loc[(df_emotion['character_B'] == neighbor) & (~df_emotion['character_A'].isin(found_neighbors))]['character_A']
    frames = [subset_A, subset_B]
    new_neighbors = pd.concat(frames, sort=False).unique().tolist()
    cluster = cluster.strip()
    neighbor = neighbor.strip()

    # if yes, probe if those nodes are also in the cluster
    for ii, new_neighbor in enumerate(new_neighbors):
        found_neighbors.append(new_neighbor)
        if new_neighbor != neighbor:
            new_neighbor = new_neighbor.strip()
            check_cluster = df_cluster.loc[(df_cluster['cluster'] == cluster) & (df_cluster['person'] == new_neighbor)]
            empty_list = check_cluster.empty
            #if yes, apply cluster to graph
            if empty_list == False:
                # first delete the row from cluster_frame
                df_cluster = df_cluster.drop(check_cluster.index)
                log_entry = {'from': new_neighbor,
                             'to': cluster_id}
                cluster_log = cluster_log.append(log_entry, ignore_index=True)

                cluster_log = investigateNeighbor(cluster, cluster_id, new_neighbor, df_cluster, df_emotion, cluster_log, found_neighbors)

    return cluster_log


# main function hat is looking for clusters in the dataframe. Finds initial pair and starts the recursive call to investigate that cluster
# parameters:
# df_cluster: gets the dataframe, which includes the characters and their clusters; does not say anything about the
# question of if the cluster, the character is in can be found in the dataframe
# df_emotion: pandas dataframe, which includes all relations of the bible by using 3 columns: character_A, character_B, emotion
# max_neighbor_cluster: threshold for cluster to have at least n-characters with the cluster (exclude mini-clusters), int

# return:
# df_emotion: is the ralation pandas dataframe, that has been adjusted, such that it includes the clusters

def adopt_clusters(df_cluster, df_emotion, max_neighbor_cluster):
    # get all clusters available in the data
    clusters = df_cluster['cluster'].unique().tolist()
    # add characters that have been found in the dataframe and run in the same cluster; needs at least 2
    # neighboring characters which are in the same cluster to add them
    cluster_log = pd.DataFrame()

    for cluster in clusters:
        # find the characters at the current cluster
        characters_in_cluster = df_cluster.loc[df_cluster['cluster'] == cluster]['person'].values
        cluster = cluster.strip()

        for idx_p, cluster_person in enumerate(characters_in_cluster):
            person = cluster_person.strip()

            # get all dataframe entries of the bible for the people in the current cluster
            # df_emotion has 3 columns: character_A, character_B, emotion
            subset_A = df_emotion.loc[df_emotion['character_A'] == cluster_person]['character_B']
            subset_B = df_emotion.loc[df_emotion['character_B'] == cluster_person]['character_A']
            frames = [subset_A, subset_B]
            neighbors = pd.concat(frames, sort=False).unique().tolist()

            # check if neighbors have been found and that at least the max_neighbor_cluster are in that cluster (throw away mini clusters)
            if len(neighbors) > max_neighbor_cluster:
                for ii, new_neighbor in enumerate(neighbors):
                    new_neighbor = new_neighbor.strip()
                    # Since the same cluster may be found at multiple locations in the graph it has to get an individual name : cluster_id
                    cluster_id = str(cluster) + "_" + str(idx_p)
                    # initialize the set of neighbors, which have already been found in the data
                    found_neighbors = [cluster_person, new_neighbor]

                    # since already on couple has been found, add them to the dataframe
                    log_entry = {'from': cluster_person,
                                 'to': cluster_id}
                    cluster_log = cluster_log.append(log_entry, ignore_index=True)
                    log_entry = {'from': person,
                                 'to': cluster_id}
                    cluster_log = cluster_log.append(log_entry, ignore_index=True)

                    # check if further neighbors exists
                    check_cluster = df_cluster.loc[
                        (df_cluster['cluster'] == cluster) & (df_cluster['person'] == new_neighbor)
                        & (~df_cluster['person'].isin(found_neighbors))]

                    # investigate those neighbors
                    empty_list = check_cluster.empty
                    if empty_list == False:
                        cluster_log = investigateNeighbor(cluster, cluster_id, new_neighbor, df_cluster, df_emotion, cluster_log, found_neighbors)

    # check if clusters could be found in the data
    empty_list = cluster_log.empty
    if empty_list == True:
        print('No cluster was assigned')
    else:
        # apply the cluster_log to the df_emotion dataframe, such that any cluster found in the data, overrides the existing data in the frame
        df_emotion = replaceCluster(cluster_log, df_emotion)
    return df_emotion

# distill the dataframe to only allow on edge between two characters.
# takes mean of their emotion
# parameter:
# df_emotion: pandas dataframe, that has 3 columns: character_A, character_B, emotion; has been preprocessed and now includes clusters

# return: distilled dataframe, that only has one edge between two nodes

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
        if (i+1) % 10 == 0:
            print(str(i+1) + "/" + str(len(label)))
        for character_B in label_remove:
            if character_A != character_B:
                # find emotions in both directions
                subset_A = df_emotion.loc[(df_emotion['character_A'] == character_A) & (df_emotion['character_B'] == character_B) & (df_emotion['emotion'].notna() == True)]
                subset_B = df_emotion.loc[(df_emotion['character_A'] == character_B) & (df_emotion['character_B'] == character_A) & (df_emotion['emotion'].notna() == True)]

                # join both dataframes
                frames = [subset_A, subset_B]
                subset = pd.concat(frames, sort=False)
                empty_list = subset.empty
                if empty_list == False:

                    # calculate mean over relations
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
                    df_distilled = df_distilled.append(new_row, ignore_index=True)
        label_remove.remove(character_A)

    return df_distilled

# aim is to pull clusters together which lay at the same node
# if e.g. jesus has neighbors clusters cluster1 and cluster2, those clusters may be concatinated, such that only one cluster exist at the end.
# parameter:
# df_emotion: pandas dataframe containing all relations

# return:
# df_concated_cluster: df_emotion, but with concatinated clusters; >1 cluster at one node

def concat_cluster(df_emotion):
    A = df_emotion['character_A'].unique().tolist()
    B = df_emotion['character_B'].unique().tolist()
    labels = A + B
    labels = list(set(labels))
    try:
        labels.remove("")
    except:
        pass
    # find all labels which are
    labels_remove = labels.copy()

    df_concated_cluster = pd.DataFrame()

    for i, label in enumerate(labels):
        if (i+1) % 10 == 0:
            print(str(i+1) + "/" + str(len(labels)))

        # Find all rows which have "cluster*" and the target character in their configuration - visa versa
        # looks for any existing edge between both
        # regex: wants to find cluster indications
        # cluster always added to character_B in replaceCluster()

        subset = df_emotion.loc[(df_emotion['character_A'] == label) & (df_emotion['character_B'].str.contains(r"cluster[0-9]+", regex=True))]
        empty_list = subset.empty
        if empty_list == False and subset.shape[0] > 1:
            # new mean is calculated since i'm grabing multiple rows.
            # Those rows have individual emotions; I then value them like before.
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

            cluster_name = "cluster" + str(i)
            new_row = {'character_A': min(label, cluster_name), 'character_B': max(label, cluster_name), 'emotion': emotion_mean}
            df_concated_cluster = df_concated_cluster.append(new_row, ignore_index=True)

            for ii, label_r in enumerate(labels_remove):
                # we dont just want the clusters
                # but also those, which are between 2 person, non-clustered
                # regex: wants to find cluster indications
                subset_A = df_emotion.loc[(df_emotion['character_A'] == label) & (df_emotion['character_B'] == label_r) &
                                          (df_emotion['character_B'].str.contains(r"[\w,\s]+", regex=True)) &
                                          (df_emotion['character_A'].str.contains(r"[\w,\s]+", regex=True))]

                subset_B = df_emotion.loc[(df_emotion['character_A'] == label_r) & (df_emotion['character_B'] == label_r) &
                                          (df_emotion['character_B'].str.contains(r"[\w,\s]+", regex=True)) &
                                          (df_emotion['character_A'].str.contains(r"[\w,\s]+", regex=True))]

                frames = [subset_A, subset_B]
                subset = pd.concat(frames, sort=False)

                # check if no entry like subset_A and subset_B have been found in the dataframe
                # what follows can only happen if there is data given
                empty_list = subset.empty
                if empty_list == False:
                    df_concated_cluster = df_concated_cluster.append(subset, ignore_index=True)

            labels_remove.remove(label)
    return df_concated_cluster

# main functionality call to apply the cluster changes to the graph
# question of if the cluster, the character is in can be found in the dataframe

# parameter:
# df_cluster: df_cluster: gets the dataframe, which includes the characters and their clusters; does not say anything about the
# df_emotion: pandas dataframe, which includes all relations of the bible by using 3 columns: character_A, character_B, emotion
# load: load data from csv file or compute them, bool
# exp_name: name of the experiment to save the plot, string
# max_neighbor_cluster: threshold for cluster to have at least n-characters with the cluster (exclude mini-clusters), int

# return:
# dataframe: relations dataframe, adjusted by the clustering, included concatinating multiple cluster nodes at one character

def adjust_graph(df_cluster, df_emotion, load, exp_name, max_neighbor_cluster):
    if load == True:
        try:
            dataframe = pd.read_csv("bibleTA_clustered_concat.csv")
        except:
            raise ValueError(
                'Could not load file, make sure to following file exists: ' + str("bibleTA_clustered_concat.csv"))

    else:
        # find and include clusters in the graph
        dataframe = adopt_clusters(df_cluster=df_cluster, df_emotion=df_emotion, max_neighbor_cluster=max_neighbor_cluster)
        # only allow one edge between two nodes
        dataframe = distill_shrunken_df(df_emotion=dataframe)
        # convert character names and cluster names to numerical nodes
        edges, color_emotion, label = convertToIGraph(dataframe=dataframe)
        # plot adjusted graph
        plotGraph(edges, color_emotion, label, location=exp_name, exp="3_adjusted_graph")
        # pull clusters together which lay at the same node
        dataframe = concat_cluster(dataframe)
        # save dataframe to file, to reload later
        dataframe.to_csv("bibleTA_clustered_concat.csv")

    return dataframe

# function to call from outside

def main():
    # experiment name, to later save files in this folder
    exp_name = "images/" + time.strftime("%Y%m%d_%H%M%S", time.gmtime(time.time())) + "_images"
    # load the bible and take up the relations of characters
    df_emotion, label_list = getGraph(df_bible=None, load=True, threshhold=6, testament="new", exp_name=exp_name)
    # loads the clusters based on keywords to a dataframe
    df_cluster = getCluster(load=True, mock_words=6, num_cluster=6, threshold=(1/2), label_list=label_list, exp_name=exp_name)
    # apply the clusters to the dataframe and distill it
    dataframe = adjust_graph(df_cluster=df_cluster, df_emotion=df_emotion, load=False, exp_name=exp_name, max_neighbor_cluster=3)
    # get numerical nodes and edges from dataframe
    edges, color_emotion, label = convertToIGraph(dataframe=dataframe)
    # plot datafrane
    plotGraph(edges, color_emotion, label, location=exp_name, exp="4_clustered_emotion_graph")

if __name__ == "__main__":
    main()

