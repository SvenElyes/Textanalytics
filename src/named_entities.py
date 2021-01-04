import dataloader
import pandas as pd
import spacy
from spacy import displacy
import en_core_web_sm

major_personalities = {"God", "Jehova", "the Lord", "the Holy Spirit", "Adam", "Cain", "Abel", "Seth", "Enos", "Kenan",
"Mahalalel", "Jared", "Enoch", "Methuselah", "Lamech", "Noah", "Shem", "Irad", "Mehujael",
"Methusael", "Tubal", "Arpachshad", "Shelah", "Eber", "Peleg", "Reu", "Serug", "Nahor", "Terah",
"Abraham", "Isaac", "Jacob", "Judah", "Perez", "Hezron", "Ram", "Amminadab", "Nahshon",
"Salmon", "Boaz", "Obed", "Jesse", "David", "Levi", "Joseph", "Sarah", "Rebecca", "Rahel",
"Leah", "Moses", "Aaron", "Miriam", "Eldad", "Medad", "Phinehas", "Joshua", "Deborah", "Gideon",
"Eli", "Elkanah", "Hannah", "Abigail", "Samuel", "Gad", "Nathan", "David", "Solomon", "Jeduthun",
"Ahijah", "Shemaiah", "Elijah", "Elisha", "Iddo", "Hanani", "Jehu", "Micaiah", "Jahaziel", "Eliezer",
"Zechariah", "Huldah", "Isaiah", "Jeremiah", "Ezekiel", "Daniel", "Hosea", "Joel", "Amos",
"Obadiah", "Jonah", "Micah", "Nahum", "Habakuk", "Zaphaniah", "Haggai", "Zechariah", "Malachi",
"Beor", "Balaam", "Job", "Amoz", "Beeri", "Baruch", "Agur", "Uriah", "Buzi", "Mordecai", "Esther",
"Oded", "Azariah", "Saul", "Ish-bosheth", "Jeroboam", "Nadab", "Baasha", "Elah", "Zimri", "Tibni",
"Omni", "Ahab", "Ahaziah", "Jehoram", "Jehu", "Jehoahaz", "Jehoash", "Shallum", "Menahem", "Pekahiah", "Pekah",
"Hoshea", "Rehoboam", "Abijam", "Asa", "Jehoshaphat", "Athaliah", "Amaziah", "Uzziah", "Jotham",
"Ahaz", "Hezekiah", "Manasseh", "Amon", "Josiah", "Jehoahaz", "Jehoiakim", "Jeconiah", "Zedekiah",
"Simon Thassi", "John Hyrcanus", "Aristobulus", "Alexander Jannaeus", "Salome Alexandra", "Hyrcanus",
"Antigonus", "Eleazar", "Asher", "Benjamin", "Dan", "Gad", "Issachar", "Ephraim", "Menasheh", "Naphtali",
"Reuben", "Simeon", "Zebulun", "Eleazar Abaran", "John Gaddi", "Jonathan Apphus", "Judas Maccabeus", 
"Mattathias", "Alexander the Great", "Philipp II of Macedon", "Astyages", "Darius", "Tobit", "Judith",
"Susanna", "Jesus", "Mary", "James", "Thaddeus", "Jude", "Simon", "Mary of Clopas", "Cleopas", "Alphaeus", "Peter",
"Andrew", "James, son of Zebedee", "John, son of Zebedee", "John", "Bartholomew", "Nathanael", "Thomas",
"Matthew", "James, son of Alphaeus", "Judas, son of James", "Simon the Zealot", "Judas Iscariot", "Paul", "Matthias",
"Barnabas", "Caiaphas", "Annas", "Agabus", "Anna", "Simeon", "John the Babtist", "Apollos", "Aquila",
"Dionysius the Areopagite", "Dionysius", "Epaphras", "John Mark", "Lazarus", "Luke", "Martha", "Mary Magdalene", 
"Mary, sister of Martha", "Nicodemus", "Onesimus", "Philemon", "Priscilla", "Silas", "Sopater", "Stehen", "Timothy",
"Titus", "Agrippa", "Felix", "Herodias", "Herod the Great", "Philip the Tetrarch", "Pontius Pilate", "Quirinius", "Augustus",
"Tiberius", "Nero"}

nlp = en_core_web_sm.load()
#coref = neuralcoref.NeuralCoref(nlp.vocab)
#nlp.add_pipe(coref, name='neuralcoref')
df_bible = dataloader.get_df_bible()

def get_characters_from_text(text):
    doc = nlp(text)
    characters = set()
    for i in range(len(doc)):
        if doc[i].text in major_personalities:
            characters.add(doc[i].text)
    for entity in doc.ents:
        #print(entity.text)
        if entity.label_ == "PERSON":
            #print(entity.label_, ' | ', entity.text)
            characters.add(entity.text)
    return characters

def show_word_mapping(text):
    doc = nlp(text)
    displacy.serve(doc, style='dep')

def add_character_column(df):
    df.insert(len(df.columns), "characters", None, True)
    for i in range(len(df)):
        characters = get_characters_from_text(df.loc[i, ("text")])
        df.loc[i, ("characters")] = "| ".join(characters)

test_df = df_bible.head(100)

if __name__ == "__main__":
    add_character_column(df_bible)
    df_bible.to_csv('bibleTA_characters.csv', index=False, encoding='utf-8')