from src.data.relation import Relation


class Character:
    """Simple class to store and retrieve information about a character. Specifially: name, aliases for the name, relations to other characters, keywords."""

    def __init__(self, name):
        self.name = name
        self.alias = list()
        self.relations = list()
        self.most_frequent_words = list()

    def add_most_frequent_words(self, wordlist):
        self.most_frequent_words.extend(wordlist)

    def add_most_frequent_word(self, word):
        self.most_frequent_words.append(word)

    def set_most_frequent_words(self, wordlist):
        """erases existing wordlist and sets wordlist"""
        self.most_frequent_words = wordlist

    def add_relation(self, relation):
        self.relations.append(relation)

    def get_relations(self):
        return self.relations

    def get_most_frequent_words(self):
        return self.most_frequent_words

    def get_name(self):
        return self.name
