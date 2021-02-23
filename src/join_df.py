from tqdm import tqdm
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Find style templates')
parser.add_argument('--character', type=str, default='', help='name of characters csv file')
parser.add_argument('--emotion', type=str, default='', help='name of emotion csv file')
parser.add_argument('--out', type=str, default='bibleTA_emotion.csv', help='output name of csv file')
args = parser.parse_args()

def main(character_csv, relation_csv, out_csv, df_bible):
    # parameter:
    # character_csv : path to csv file with characters, string
    # relation_csv: path to csv file with relations, string
    # out_csv: : path to save the csv file to, string
    # df_bible : pandas dataframe might be given by calling it from main(), else dataframe is loaded

    # return:
    # df_bible : pandas dataframe with characters and relation information in one file

    print("python3 join_df.py")
    print(" - started joining the dataframes " + str(relation_csv) + " and " + str(character_csv))

    # check if parameters have been given by the console
    if args.emotion == "":
        args.emotion = relation_csv
    if args.character == "":
        args.character = character_csv
    if args.out == "":
        args.out = out_csv

    # if no dataframe was given load the dataframe given in the parameters
    if not isinstance(df_bible, pd.DataFrame):
        df_bible = pd.read_csv(relation_csv)
    # load found characters
    df_characters = pd.read_csv(character_csv)

    names_list = []
    emotion = []
    for i, (df_b_verse, df_c_verse) in tqdm(enumerate(zip(df_bible.iterrows(), df_characters.iterrows()))):
        # get relation data to change their aggregation
        # check if they exist in dataframe
        if 'bow_emotion' in df_b_verse[1]:
            score_bow = df_b_verse[1]["bow_emotion"]
        else:
            similarity_emotion = 0.0

        if 'tb_emotion' in df_b_verse[1]:
            score_textblob = df_b_verse[1]["tb_emotion"]
        else:
            similarity_emotion = 0.0

        if 'similarity_emotion' in df_b_verse[1]:
            score_similarity = df_b_verse[1]["similarity_emotion"]
        else:
            score_similarity = 0.0

        # adds all relation data
        score = score_bow + score_textblob + score_similarity

        # do aggregation
        if score > 0.75:
            emotion.append(1.0)
        elif score < -0.75:
            emotion.append(-1.0)
        else:
            emotion.append(0.0)

        # get characters from dataframe; check if the row is not NaN
        if not pd.isna(df_c_verse[1]["characters"]):
            names = df_c_verse[1]["characters"]
            names = "[" + str(names) + "]"
        else:
            names = "[]"
        names_list.append(names)

    # append dataframe with data
    df_bible["characters"] = names_list
    df_bible["emotion"] = emotion

    # save to file
    df_bible.to_csv(args.out)
    print(" - finished joining both dataframes and forwarded file")
    return df_bible

if __name__ == "__main__":
    main(character_csv="src/csv/bibleTA_characters.csv", relation_csv="src/csv/bibleTA_prepro.csv",
         out_csv="src/csv/bibleTA_emotion.csv")
