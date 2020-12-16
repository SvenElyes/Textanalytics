import json
#if you want to look at the bible, from a clear perspective, look at the 
with open("bible/bible.json") as bible:
    bible = json.load(bible)

#Matthew is the first book of the new testatement
books = [
    'Genesis',
    'Exodus',
    'Leviticus',
    'Numbers',
    'Deuteronomy',
    'Joshua',
    'Judges',
    'Ruth',
    '1 Samuel',
    '2 Samuel',
    '1 Kings',
    '2 Kings',
    '1 Chronicles',
    '2 Chronicles',
    'Ezra',
    'Nehemiah',
    'Esther',
    'Job',
    'Psalm',
    'Proverbs',
    'Ecclesiastes',
    'Song of Solomon',
    'Isaiah',
    'Jeremiah',
    'Lamentations',
    'Ezekiel',
    'Daniel',
    'Hosea',
    'Joel',
    'Amos',
    'Obadiah',
    'Jonah',
    'Micah',
    'Nahum',
    'Habakkuk',
    'Zephaniah',
    'Haggai',
    'Zechariah',
    'Malachi',
    'Matthew',
    'Mark',
    'Luke',
    'John',
    'Acts',
    'Romans',
    '1 Corinthians',
    '2 Corinthians',
    'Galatians',
    'Ephesians',
    'Philippians',
    'Colossians',
    '1 Thessalonians',
    '2 Thessalonians',
    '1 Timothy',
    '2 Timothy',
    'Titus',
    'Philemon',
    'Hebrews',
    'James',
    '1 Peter',
    '2 Peter',
    '1 John',
    '2 John',
    '3 John',
    'Jude',
    'Revelation'
]
# in the json file, the books are titled as chapters, while the word book is being used to reference the whole bible.


def get_book(bookname):
    index = books.index(bookname)
    return bible["Book"][index]

def get_verseid(verseid):
    #do we iterate over all books?
    #verseid example 00000003
    booktotal= len(books)
    for bookindex in range(0,booktotal):
        versetotal =len(bible["Book"][bookindex]['Chapter'])
        for verseindex in range (0,versetotal):
            for key in bible["Book"][bookindex]['Chapter'][verseindex]['Verse']:
                #print(key["Verseid"],key["Verse"])
                if key["Verseid"] == verseid:
                    #print(key["Verse"])
                    return key["Verse"],bookindex
def old_or_new_testament(bookname):
    matthewindex = books.index('Matthew')
    bookindex= books.index(bookname)
    if matthewindex < bookindex :
        return "new"
    else:
        return "old"
print(get_verseid("65021019"))
print(old_or_new_testament("Genesis"))

