import Relation

class Character:
"""Simple class to store and retrieve information about a character. Specifially: name, aliases for the name, relations to other characters, keywords."""

    def __init__(self, name):
        self.name = name
        self.alias = list()
        self.relations = list()
        self.most_frequent_words = list()
    
    def add_relation(self, target_character):
        self.relations.append(Relation(self, target_character))

    def get_relations(self):
        return self.relations
    
    def get_most_frequent_words(self):
        return self.most_frequent_words
