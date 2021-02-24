import igraph as Graph
import pandas as pd
import os
import numpy as np
import spacy
from sklearn.cluster import KMeans
from pylab import *
import re
import time

import src.pickle_handler as ph
import src.relation_creator as rc

# the dataframe has been preprocessed by many other functions. However we only need a subset of this information to
# create a graph representation of the relations.
# distillDataframe() takes the dataframe given to it. It selects
# a) character_A
# b) character_B

# formate_bible should transform all rows, that contain >= 2 characters into multiple rows between all characters
# from [lukas, mark, maria] to [[lukas, mark],[lukas, maria], [maria, mark]]


def formate_bible(df_bible):
    # Parameter
    # df_bible : expects a pandas dataframe that consists of "characters" and "emotion" column

    # Return
    # df_bible_formate : pandas dataframe, that consists of 3 columns "character_A", "character_B", "emotion"
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
        names = names.split("|")
        names_remove = names.copy()

        if len(names) >= 2:
            for name in names:
                for r_name in names_remove:
                    if name != r_name:
                        new_row = {
                            "character_A": name.strip(),
                            "character_B": r_name.strip(),
                            "emotion": emotion,
                        }
                        df_bible_formate = df_bible_formate.append(
                            new_row, ignore_index=True
                        )
                names_remove.remove(name)
    print(" - Pre-processed the dataframe to run the graph generation")
    return df_bible_formate


# distillDataframe should turn the dataframe to distinct rows, which have been aggregated in terms of
# their emotion. The dataframe needs to be aggregated because characters may occur at multiple verses
# to not list those multiple times within an graph and to given an more "global" represenation of their
# emotional state the emotion is aggregated. If the emotion_mean > 0.75, the relation is considered to be positive
# if the emotion_mean < -0.75 the relation is considered to be negative.
# else wise it is neutral. The relation will later be used to project an color to the graph.


def distillDataframe(df_bible, load, threshold, save):
    # Parameter
    # df_bible : pandas dataframe of the bible
    # load : determines if a csv should be loaded or if one has to be produced by the function, bool
    # threshold : counts how often relations should occur before being considered reasonable
    # save : if file should be saved at the end, bool
    # i.e. one time mentions may not be displayed, integer

    # Return
    # df_distilled :  pandas dataframe consistent of distinct relations
    # label : unique list of all characters

    # create a list of labels (names) which have been detected in both rows character_A and #character_B
    file = os.path.join("src", "csv", "bibleTA_distilled" + "_" + str(threshold) + ".csv")
    if load == True:
        try:
            df_distilled = pd.read_csv(file)
            # get list of unique characters
            A = df_distilled["character_A"].unique().tolist()
            B = df_distilled["character_B"].unique().tolist()
            label = A + B
            label = list(set(label))
            label.sort(reverse=False)
            print(label)
            try:
                label.remove("")
            except:
                pass
            print(" - Dataframe has been loaded successfully from : " + str(file))
        except:
            print(
                " - WARNING: Could not load file, make sure to following file exists: " + str(file)
            )
            load = False

    if load == False:
        # get list of unique characters
        A = df_bible["character_A"].unique().tolist()
        B = df_bible["character_B"].unique().tolist()
        label = A + B
        label = list(set(label))
        label.sort(reverse=False)
        try:
            label.remove("")
        except:
            pass
        # create output dataframe to be further processed
        df_distilled = pd.DataFrame()
        # iterate over all labels
        # only count in one direction e.g. "character_A" = "Lukas", "character_B"="Jesus" ;
        # do not do a subsequent "character_b" = "Lukas", "character_a"="Jesus" search ;
        # implemented by removal of labels in label_remove list
        label_remove = label.copy()
        for i, character_A in enumerate(label):
            if (i + 1) % 10 == 0:
                print(str(i + 1) + "/" + str(len(label)))
            for character_B in label_remove:
                if character_A != character_B:
                    # count emotions in both directions
                    subset_A = df_bible.loc[
                        (df_bible["character_A"] == character_A)
                        & (df_bible["character_B"] == character_B)
                        & (df_bible["emotion"].notna() == True)
                    ]
                    subset_B = df_bible.loc[
                        (df_bible["character_A"] == character_B)
                        & (df_bible["character_B"] == character_A)
                        & (df_bible["emotion"].notna() == True)
                    ]

                    # join both dataframes
                    frames = [subset_A, subset_B]
                    subset = pd.concat(frames, sort=False)


                    if check_empty(subset) == False and subset.shape[0] > threshold:
                        # calculate mean over emotions
                        emotion_mean = np.mean(subset["emotion"])
                        # round it to an absolute emotion (needed for coloring in graph)
                        if emotion_mean > 0.75:
                            emotion_mean = 1.0
                        elif emotion_mean < -0.75:
                            emotion_mean = -1.0
                        else:
                            emotion_mean = 0.0
                        # add new row to the distilled dataframe
                        # df_distilled will have distinct and aggregated emotion rows. From this we
                        # can create edge colors and the concrete edges of the graph
                        # sort names alphabetically, will later be "exploited" while shrinking the graph
                        new_row = {
                            "character_A": character_A,
                            "character_B": character_B,
                            "emotion": emotion_mean,
                        }

                        # create object from relation class like new_row

                        df_distilled = df_distilled.append(new_row, ignore_index=True)
            label_remove.remove(character_A)

        A = df_distilled["character_A"].unique().tolist()
        B = df_distilled["character_B"].unique().tolist()
        label = A + B
        label = list(set(label))
        print(" - Dataframe has been distilled successfully")
        if save == True:
            df_distilled.to_csv(file)
            print(" - Distilled dataframe is saved to : " + str(file))
    return df_distilled, label, load


# plotGraph plots the graph based on its edges and edge colors and save it to the given path
# first it builds the graph based on its nodes and edges.
# than it tries to create the experiment folder, where it than saves the graph plot
# if no relation information is given in color_emotion all edges will be black.


def plotGraph(edges, color_emotion, label, location, exp):
    # Parameter:
    # edges : numerical edges which are mapped by the dict to its label, numpy array
    # color_emotion : based on "emotion" column a color is chosen to represent the kind of relation, list
    #                   black: neutral, red: bad, green: good
    # label : unique list of all characters
    # location : place to save the experiment to, string
    # exp : name of the experiment, string

    graph = Graph.Graph(n=len(label), edges=edges)
    file = os.path.join(location, exp + ".png")
    if color_emotion == []:
        out = Graph.plot(
            graph, vertex_size=10, vertex_color=["white"], vertex_label=label
        )
        out.save(file)
    else:
        out = Graph.plot(
            graph,
            vertex_size=10,
            vertex_color=["white"],
            vertex_label=label,
            edge_color=color_emotion,
        )

        out.save(file)
    print(" - Graph has been saved to : " + str(file))

# converts a distinct dataframe to numerical nodes and edges.
# has the option to include a value {-1,0,1} which will create a color to the edge
# between the neighboring nodes {red, black, green}
# in context of the bible evaluation this stands for {negative, positive, neutral}
def dataframe2graph(dataframe, relations, side_A, side_B, relation_value):
    # Parameters:
    # dataframe : pandas dataframe consistent of labels and relation between characters
    # relations : should the evaluation consider a relation to be positive/neutral/negative, bool
    # side_A : in which column of the dataframe will the function find neighboring node A, string
    # side_B : in which column of the dataframe will the function find neighboring node B, string
    # relation_value : in which column of the dataframe will the function find the color value, string

    # Return
    # edges : numpy array which transfers labels to numerical values
    # color_emotion : returned if relations == True, to list of colors of the edges based on their relation, list
    # label : unique list of labels which index matches the edges

    A = dataframe[str(side_A)].unique().tolist()
    B = dataframe[str(side_B)].unique().tolist()
    label = A + B
    label = list(set(label))
    label.sort(reverse=False)

    label2id = {l: i for i, l in enumerate(label)}
    id2label = {i: l for i, l in enumerate(label)}

    if relations == True:
        # color dict for transfering the emotion score to a colored edge
        color_dict = {1: "green", -1: "red", 0: "black"}
        color_emotion = []

    edges = []

    for i, df_verse in dataframe.iterrows():
        A = label2id[df_verse[str(side_A)]]
        B = label2id[df_verse[str(side_B)]]
        edges.append([A, B])

        if relations == True:
            relation = df_verse[str(relation_value)]
            color_emotion.append(color_dict[relation])

    print(" - Dataframe has been converted to numerical nodes and edges")
    if relations == True:
        return edges, color_emotion, label
    else:
        return edges, label


# load csv in case new textAnalytics outout has been generated. can be set in main.
# load the bible as csv and can differentiate between the old an the new testament


def loadCSV(testament, file):
    # Parameter:
    # testament : "new", "old", string
    # give csv file name, string

    # Return
    # df_bible : pandas dataframe which contains the bible loaded from csv file
    df_bible = pd.read_csv(file)

    if testament == "new":
        first_matthew_verse = df_bible.index[
            (df_bible["book_id"] == "Matt")
            & (df_bible["verse"] == 1)
            & (df_bible["chapter"] == 1)
        ].tolist()[0]
        df_bible = df_bible[first_matthew_verse:]

    if testament == "old":
        first_matthew_verse = df_bible.index[
            (df_bible["book_id"] == "Matt")
            & (df_bible["verse"] == 1)
            & (df_bible["chapter"] == 1)
        ].tolist()[0]
        df_bible = df_bible[: (first_matthew_verse - 1)]
    return df_bible


# main function, which calls all the other functions and can be called from outside.
# can be given a dataframe (or it loads one from the folder)
# can also be given load, which loads the last distilled dataframe with distinct
# character_A to character_B mappings with an aggregated emotion value
def getGraph(df_bible, load, threshold, testament, location, file):
    # Parameter:
    # df_bible : pandas dataframe, may be given from outside
    # load : load calculations from previous run
    # threshold : counts the encounterments of two characters in one verse, int
    # testament : "old", "new", else both testaments, string
    # location : name of the experiment, string
    # file : csv file name used to load the data, string

    # Return:
    # df_relation : pandas dataframe of the relations, dataframe consistes of ["character_A", "character_B", "emotion"]
    # label_list : unique list of characters in dataframe
    # load : chain of trust in load; if one load fails all subsequent are set to False, bool

    # loads bible dataframe if not given one
    if not isinstance(df_bible, pd.DataFrame):
        df_bible = loadCSV(testament=testament, file=file)
        df_bible = pd.DataFrame(data=df_bible)

    df_bible = formate_bible(df_bible)
    df_relation, label_list, load = distillDataframe(
        df_bible, load, threshold=threshold, save=True
    )

    # convert distilled data to nodes and edges. Also generate colored edges
    edges, color_emotion, label = dataframe2graph(df_relation, True, "character_A", "character_B", "emotion")

    # make and plot graph + save to path
    plotGraph(edges, color_emotion, label, location=location, exp="1_emotion_graph")
    return df_relation, label_list, load

# This Function aims to create the pickle, objecs using the relation_creator. We need a distiled csv to do the work.
def create_pickle_objects(df_emotion):
    # parameter :
    # df_emotion : pandas dataframe with relations, just distilled

    rc.create_char_relation(df_emotion)
    rc.create_character_keywords()


# function is to load the pickle objects, to process their keywords each person
def load_pickle_objects():
    pickle_obj = ph.PickleHandler()
    pickle_list = pickle_obj.load_characters()
    labels = []
    res = []
    temp_res = []
    for obj in pickle_list:
        name = obj.name
        labels.append(name)
        most_frequent_words = obj.most_frequent_words

        for word in most_frequent_words:
            temp_res.append(word[0])
        res.append(temp_res)
        temp_res = []

    return labels, res


# cluster keywords and create list of people in this cluster that have threshold enough keywords coming from the same cluster
def cluster_data(num_cluster, threshold):
    # Parameter:
    # num_cluster: number of cluster centroid - results in labels for keywords, int
    # threshold: min count of cluster label to app person to cluster, int

    # return:
    # df_cluster: pandas dataframe, consistent of cluster name and character

    file = os.path.join("src", "csv", "clustered_keywords_" + str(num_cluster) + str(int(1/threshold)) + ".csv")
    # load the pickle objects to find keyword clusters
    characters, res = load_pickle_objects()

    # extract distinct keywords to convert them to word-vectors and afterwards determine clusters
    distinct_res = []
    for keywords_res in res:
        for keyword in keywords_res:
            if keyword not in distinct_res:
                distinct_res.append(keyword)

    # load spaCy's word2vec
    nlp = spacy.load("en_core_web_lg")

    # vectorize the list of disinct keywords
    # 300 is vector length of spacy word representation
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
    keyword2cluster = {
        keyword: cluster for keyword, cluster in zip(distinct_res, clustered_words)
    }

    clustered_res = []
    temp = []
    for i, keywords_res in enumerate(res):
        for ii, keyword in enumerate(keywords_res):
            label = keyword2cluster[keyword]
            temp.append(label)
        clustered_res.append(temp)

    df_cluster = pd.DataFrame()
    # for all cluster (e.g. 4 --> 0, 1, 2, 3)
    for cluster in range(num_cluster):
        # count how often this cluster occurred in the the clusterized keywords (clustered_res == cluster) each person (axis = 1)
        for i, clustered_char in enumerate(clustered_res):
            ct = clustered_char.count(cluster)
            ind_threshold = threshold * len(clustered_char)
            if ct > ind_threshold:
                # result = np.count_nonzero(clustered_res == cluster, axis=1)
                # make node for graph (may later be turned into a cloud if possible)
                cluster_name = "cluster" + str(cluster)
                # append to dataframe
                new_row = {"person": characters[i], "cluster": cluster_name}
                df_cluster = df_cluster.append(new_row, ignore_index=True)
    df_cluster.to_csv(file)
    print(" - Clusters have been created and will be saved to : " + str(file))
    return df_cluster


# getCluster loads the clusters to a dataframe. Either from csv file or by calling cluster_data()
# dataframe is then prepared to be displayed as a graph and subsequent plotted


def getCluster(load, num_cluster, threshold, location):
    # parameter:
    # load: if true, load data from csv file, bool
    # num_cluster: number of cluster centroid - results in labels for keywords, int
    # threshold: min count of cluster label to app person to cluster, int
    # location: name of the experiment to save the plot, string

    # return:
    # df_cluster = pandas dataframe, consistent of cluster name and character

    # if load = True, load pre-evaluated csv file
    file = os.path.join("src", "csv", "clustered_keywords_" + str(num_cluster) + str(int(1/threshold)) + ".csv")
    if load == True:
        try:
            df_cluster = pd.read_csv(file)
            print(" - Clusters have been loaded from file : " + str(file))
        except:
            print(" - WARNING: Could not load file: " + str(file))
            print(" - run adjust_graph() to create clusters")
            load = False

    if load == False:
        # from character with keywords to a dataframe that shows edges, where
        # more keywords from one cluster have occurred than threshold says
        df_cluster = cluster_data(num_cluster=num_cluster, threshold=threshold)

    if check_empty(df_cluster)== False:
        # convert edges to nummeric edges and prepare node labels
        edges, label = dataframe2graph(df_cluster, False, "person", "cluster", "")
        # plot the graph
        plotGraph(edges, [], label, location=location, exp="2_cluster_graph")
    else:
        print(" - WARNING: could not add any cluster, lower threshold")

    return df_cluster, load


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
def replaceCluster(cluster_log, df_emotion):
    # parameter:
    # cluster_log: pandas dataframe that hold all character, that are in cluster and can be found in graph
    # df_emotion: bible dataframe, has 3 rows: character_A, character_B and emotion

    # return:
    # df_emotion_cluster: new bible dataframe, including the clusters, also in format: character_A, character_B and emotion

    # remove duplicates in log
    cluster_log = cluster_log.drop_duplicates()
    # get all people in some cluster
    # iterate over the labels
    labels = cluster_log["from"].unique().tolist()
    for label in labels:
        # get the clusters the label is in
        charincluster = cluster_log.loc[cluster_log["from"] == label]
        for i, row in charincluster.iterrows():
            # get any neighbor of the current label
            from_char, to_char = row.values
            subset_A = df_emotion.loc[df_emotion["character_A"] == from_char]
            subset_B = df_emotion.loc[df_emotion["character_B"] == from_char]
            frames = [subset_A, subset_B]
            neighbors = pd.concat(frames, sort=False)

            # include the current cluster "to_char" to all neighbors, because the node is reduced from the graph
            for ii, row_neighbor in neighbors.iterrows():
                try:
                    _, char_A, char_B, emotion = row_neighbor.values
                except:
                    char_A, char_B, emotion = row_neighbor.values

                # character can be in "character_A" oder "character_B"...
                if char_A == from_char:
                    entry = {"character_A": to_char, "character_B": char_B, "emotion": emotion}
                else:
                    entry = {"character_A": to_char, "character_B": char_A, "emotion": emotion}
                df_emotion = df_emotion.append(entry, ignore_index=True)

        # because character is transfered to a cluster, the initial rows are dropped from the dataframe
        subset_A = df_emotion.loc[df_emotion["character_A"] == label]
        subset_B = df_emotion.loc[df_emotion["character_B"] == label]

        frames = [subset_A, subset_B]
        drop_df = pd.concat(frames, sort=False)

        for ii, row in drop_df.iterrows():
            df_emotion = df_emotion.drop(index=ii)
    # also, there may be self relations because of a neighbor chain within the same cluster
    # delete those relations, e.g. "cluster1" to "cluster1"
    subset = df_emotion.loc[df_emotion["character_B"] == df_emotion["character_A"]]
    for ii, row in subset.iterrows():
        df_emotion = df_emotion.drop(index=ii)
    return df_emotion

# recursive call to do a depth first search
# is given a person which is in cluster x and checks every relation/neighbor node if this node is also in the cluster
# if so, the previous node is added to the cluster / marked "in cluster"


def investigateNeighbor(
    cluster, cluster_id, neighbor, df_cluster, df_emotion, cluster_log, found_neighbors
):
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

    # probe if the node has neighbor nodes

    subset_A = df_emotion.loc[
        (df_emotion["character_A"] == neighbor)
        & (~df_emotion["character_B"].isin(found_neighbors))
    ]["character_B"]
    subset_B = df_emotion.loc[
        (df_emotion["character_B"] == neighbor)
        & (~df_emotion["character_A"].isin(found_neighbors))
    ]["character_A"]
    frames = [subset_A, subset_B]
    new_neighbors = pd.concat(frames, sort=False).unique().tolist()

    # if yes, probe if those nodes are also in the cluster
    for ii, new_neighbor in enumerate(new_neighbors):
        found_neighbors.append(new_neighbor)
        if new_neighbor != neighbor:
            check_cluster = df_cluster.loc[
                (df_cluster["cluster"] == cluster)
                & (df_cluster["person"] == new_neighbor)
            ]

            if check_empty(check_cluster)== False:
                # first delete the row from cluster_frame
                df_cluster = df_cluster.drop(check_cluster.index)
                log_entry = {"from": new_neighbor.strip(), "to": cluster_id}
                cluster_log = cluster_log.append(log_entry, ignore_index=True)

                # start recursive call to find other members of the current cluster
                cluster_log, df_cluster = investigateNeighbor(
                    cluster,
                    cluster_id,
                    new_neighbor,
                    df_cluster,
                    df_emotion,
                    cluster_log,
                    found_neighbors,
                )

    return cluster_log, df_cluster


# main function hat is looking for clusters in the dataframe. Finds initial pair and starts the recursive call
# to investigate that cluster


def adopt_clusters(df_cluster, df_emotion, min_neighbor_cluster):
    # parameters:
    # df_cluster: gets the dataframe, which includes the characters and their clusters; does not say anything about the
    # question of if the cluster, the character is in can be found in the dataframe
    # df_emotion: pandas dataframe, which includes all relations of the bible by using 3 columns:
    #               character_A, character_B, emotion
    # min_neighbor_cluster: threshold for cluster to have at MAX n-characters with the cluster
    # to not replace vast sums of characters
    #                       (exclude mini-clusters), int

    # return:
    # df_emotion: is the ralation pandas dataframe, that has been adjusted, such that it includes the clusters

    # get all clusters available in the data
    clusters = df_cluster["cluster"].unique().tolist()
    # add characters that have been found in the dataframe and run in the same cluster; needs at least 2
    # neighboring characters which are in the same cluster to add them
    cluster_log = pd.DataFrame()

    for cluster in clusters:
        # find the characters at the current cluster
        i = 0
        found_neighbors = []
        while True:
            characters_in_cluster = df_cluster.loc[df_cluster["cluster"] == cluster]
            # either first iteration or there are rows in the dataframe left
            if characters_in_cluster.shape[0] > 1 or (
                i > 0 and characters_in_cluster.shape[0] > 0
            ):
                cluster_person = characters_in_cluster.head(1)["person"].values[0]
                found_neighbors.append(cluster_person)
                # get all dataframe entries of the bible for the people in the current cluster
                # df_emotion has 3 columns: character_A, character_B, emotion
                subset_A = df_emotion.loc[df_emotion["character_A"] == cluster_person]["character_B"]
                subset_B = df_emotion.loc[df_emotion["character_B"] == cluster_person]["character_A"]
                frames = [subset_A, subset_B]
                neighbors = pd.concat(frames, sort=False).unique().tolist()

                # Since the same cluster may be found at multiple locations in the graph it has to
                # get an individual name : cluster_id
                cluster_id = str(cluster.strip()) + "_" + str(i)
                if len(neighbors) >= min_neighbor_cluster:
                    for ii, new_neighbor in enumerate(neighbors):
                        characters_in_cluster = df_cluster.loc[(df_cluster["cluster"] == cluster) & (df_cluster["person"] == new_neighbor)]
                        if check_empty(characters_in_cluster) == True:
                            continue
                        else:
                            # initialize the set of neighbors, which have already been found in the data
                            found_neighbors.append(new_neighbor)

                            # since already on couple has been found, add them to the dataframe
                            log_entry = {"from": cluster_person.strip(), "to": cluster_id}
                            cluster_log = cluster_log.append(log_entry, ignore_index=True)

                            # delete entry from cluster dataframe because one character can only be once in the dataframe
                            check_cluster = df_cluster.loc[
                                (df_cluster["cluster"] == cluster)
                                & (df_cluster["person"] == cluster_person)
                            ]
                            df_cluster = df_cluster.drop(check_cluster.index)

                            log_entry = {"from": new_neighbor.strip(), "to": cluster_id}
                            cluster_log = cluster_log.append(log_entry, ignore_index=True)

                            check_cluster = df_cluster.loc[
                                (df_cluster["cluster"] == cluster)
                                & (df_cluster["person"] == new_neighbor)
                            ]
                            df_cluster = df_cluster.drop(check_cluster.index)

                            # check if further neighbors exists
                            check_cluster = df_cluster.loc[
                                (df_cluster["cluster"] == cluster)
                                & (df_cluster["person"] == new_neighbor)
                                & (~df_cluster["person"].isin(found_neighbors))
                            ]

                            # investigate those neighbors
                            if check_empty(check_cluster) == False:
                                cluster_log, df_cluster = investigateNeighbor(
                                    cluster,
                                    cluster_id,
                                    new_neighbor,
                                    df_cluster,
                                    df_emotion,
                                    cluster_log,
                                    found_neighbors,
                                )
                else:
                    check_cluster = df_cluster.loc[
                        (df_cluster["cluster"] == cluster)
                        & (df_cluster["person"] == cluster_person)
                    ]
                    df_cluster = df_cluster.drop(check_cluster.index)
                i += 1
            else:
                break

    # check if clusters could be found in the data
    if check_empty(cluster_log) == True:
        print("No cluster was assigned")
    else:
        # apply the cluster_log to the df_emotion dataframe, such that any cluster found in the data, overrides the existing data in the frame
        df_emotion = replaceCluster(cluster_log, df_emotion)
    return df_emotion


# concat_cluster() should concatenate clusters which point to each other such as
# character_A = some cluster, character_B = some cluster
# so, edges from cluster to cluster should be reduced.
# tbe algorithm takes one of these rows each iteration
# and changes all the other edges to the new cluster, which will be
# formed from the concatenation


def concat_cluster(df_emotion):
    # parameter:
    # df_emotion: pandas dataframe containing all relations

    # return:
    # df_emotion: df_emotion, but with concatinated clusters - no cluster 2 cluster edges
    i = 0
    while True:
        # find cluster that either have been concatenated from other clusters
        # "cluster_[0-9]+" or "original" clusters "cluster[0-9]+"
        cluster2cluster = df_emotion.loc[
            (df_emotion["character_A"].str.contains(r"cluster_?[0-9]+", regex=True))
            & (df_emotion["character_B"].str.contains(r"cluster_?[0-9]+", regex=True))
        ]
        # check if those exist
        if check_empty(cluster2cluster) == True:
            break
        else:
            # once in a row, get the first row
            row = cluster2cluster.head(1)
            # extract the cluster names
            cluster_A = row["character_A"].values[0]
            cluster_B = row["character_B"].values[0]

            # delete the row from the dataframe
            df_emotion = df_emotion.drop(row.index)

            # take new name, concatenated clusters can be detected by "_"
            new_cluster_name = "cluster_" + str(i)
            i += 1
            # find all characters that are i in the clusters
            involved_characters_A = df_emotion.loc[
                df_emotion["character_A"] == cluster_A
            ]
            involved_characters_B = df_emotion.loc[
                df_emotion["character_A"] == cluster_B
            ]

            # join both dataframes
            frames = [involved_characters_A, involved_characters_B]
            involved_characters = pd.concat(frames, sort=False)
            if check_empty(involved_characters) == False:
                for i, character in involved_characters.iterrows():
                    # change the cluster name to the new cluster name
                    # keep rows, as they will later be aggregated by distill_shrunken_df()
                    df_emotion.loc[i, "character_A"] = new_cluster_name
            involved_characters_A = df_emotion.loc[
                df_emotion["character_B"] == cluster_A
            ]
            involved_characters_B = df_emotion.loc[
                df_emotion["character_B"] == cluster_B
            ]

            # join both dataframes
            frames = [involved_characters_A, involved_characters_B]
            involved_characters = pd.concat(frames, sort=False)
            if check_empty(involved_characters) == False:
                for i, character in involved_characters.iterrows():
                    # change the cluster name to the new cluster name
                    # keep rows, as they will later be aggregated by distill_shrunken_df()
                    df_emotion.loc[i, "character_B"] = new_cluster_name
    print(df_emotion)
    return df_emotion


# checks if dataframe is empty
def check_empty(dataframe):
    # parameter :
    # dataframe : pandas dataframe to be checked

    # return: true / false
    empty_list = dataframe.empty
    if empty_list == False:
        return False
    else:
        return True


# checks if dataframe has any data
def checkFunction(dataframe):
    # parameter :
    # dataframe : pandas dataframe to be checked
    empty = check_empty(dataframe)
    if empty:
        raise Exception("dataframe is empty")


# main functionality call to apply the cluster changes to the graph
# question of if the cluster, the character is in can be found in the dataframe


def adjust_graph(df_cluster, df_emotion, load, location, min_neighbor_cluster):
    # parameter:
    # df_cluster: df_cluster: gets the dataframe, which includes the
    #               characters and their clusters; does not say anything about the
    # df_emotion: pandas dataframe, which includes all relations of the bible
    #               by using 3 columns: character_A, character_B, emotion
    # load: load data from csv file or compute them, bool
    # exp_name: name of the experiment to save the plot, string
    # min_neighbor_cluster: threshold for cluster to have at least n-characters with
    # the cluster (exclude mini-clusters), int

    # return:
    # dataframe: relations dataframe, adjusted by the clustering, included concatenating
    # multiple cluster nodes at one character
    file = os.path.join(
        "src", "csv", "bibleTA_clustered_concat" + str(min_neighbor_cluster) + ".csv"
    )
    if load == True:
        try:
            dataframe = pd.read_csv(file)
        except:
            print("Could not load file: " + str(file))
            print("load set to False to create data")
            load = False

    if load == False:
        # find and include clusters in the graph
        dataframe = adopt_clusters(
            df_cluster=df_cluster,
            df_emotion=df_emotion,
            min_neighbor_cluster=min_neighbor_cluster,
        )
        # raises an exception if the dataframe is empty
        checkFunction(dataframe)
        # only allow one edge between two nodes
        dataframe, _, _ = distillDataframe(
            df_bible=dataframe, load=False, threshold=0, save=False
        )
    # raises an exception if the dataframe is empty
    checkFunction(dataframe)
    # convert character names and cluster names to numerical nodes
    edges, color_emotion, label = dataframe2graph(dataframe, True, "character_A", "character_B", "emotion")
    # plot adjusted graph
    plotGraph(
        edges, color_emotion, label, location=location, exp="3_adjusted_graph"
    )
    dataframe.to_csv(file)
    return dataframe


# function to call from outside


def main(threshold_getgraph, num_cluster, threshold_getcluster, file, load, df_bible):
    print("python3 eval_graph.py")
    # experiment name, to later save files in this folder
    location = "exp/" + time.strftime("%Y%m%d_%H%M%S", time.gmtime(time.time()))
    os.makedirs(location, exist_ok=True)

    if not isinstance(df_bible, pd.DataFrame):
        df_bible = None

    print("1. Pre-processing of the dataframe has started")
    # load the bible and take up the relations of characters
    df_emotion, label_list, load = getGraph(
        df_bible=df_bible, load=load, threshold=5, testament="new", location=location, file=file
    )
    # raises an exception if the dataframe is empty
    checkFunction(df_emotion)

    print("2. Pickle objects are going to be created")
    create_pickle_objects(df_emotion)
    print("3. Cluster evaluation of keywords has started")
    # loads the clusters based on keywords to a dataframe
    df_cluster, load = getCluster(load=False, num_cluster=num_cluster, threshold=threshold_getcluster, location=location)
    # apply the clusters to the dataframe and distill it
    print("4. Cluster are going to be applied to the dataframe")
    dataframe = adjust_graph(
        df_cluster=df_cluster,
        df_emotion=df_emotion,
        load=False,
        location=location,
        min_neighbor_cluster=4,
    )
    print("5. Neighboring clusters are concatenated")
    # concatenate neighbor clusters
    dataframe = concat_cluster(dataframe)
    # raises an exception if the dataframe is empty
    checkFunction(dataframe)
    # save dataframe to file, to reload later

    # get numerical nodes and edges from dataframe
    edges, color_emotion, label = dataframe2graph(dataframe, True, "character_A", "character_B", "emotion")

    # plot datafrane
    plotGraph(
        edges, color_emotion, label, location=location, exp="4_clustered_emotion_graph"
    )


if __name__ == "__main__":
    main(threshold_getgraph=5, num_cluster=4, threshold_getcluster=(1/6), file="src/csv/bibleTA_emotion.csv", load=True,
         df_bible=None)
