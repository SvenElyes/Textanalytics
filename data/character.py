import Relation

class Character:

    def __init__(self, name):
        self.name = name
        self.alias = list()
        self.relations = list()
        selt.most_frequent_words = list()
    
    def add_relation(self, target_character):
        self.relations.append(Relation(self, target_character))

    def get_relations(self):
        return relations
    
    def get_most_frequent_words(self):
        return most_frequent_words