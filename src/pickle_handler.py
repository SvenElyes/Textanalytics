"""
This module is for saving objects using the pickle module 
https://stackoverflow.com/questions/4529815/saving-an-object-data-persistence this as help.
"""
import pickle
import sys

sys.path.append("./data/")

from character import Character
from relation import Relation


class PickleHandler:
    def __init__(self):
        self.characters = list()
        self.relations = list()

    def load_characters(self):
        with open("characters.pkl", "rb") as input:
            self.characters = pickle.load(input)
            return self.characters

    def load_character_by_name(self, name):
        for character in self.characters:
            if character.name == name:
                return character
        return False

    def save_character(self, character):
        self.characters.append(character)
        with open("characters.pkl", "wb") as output:  # Overwrites any existing file.
            pickle.dump(self.characters, output, pickle.HIGHEST_PROTOCOL)

    def save_character_list(self, character_list):
        for character in character_list:
            self.characters.append(character)
        with open("characters.pkl", "wb") as output:  # Overwrites any existing file.
            pickle.dump(self.characters, output, pickle.HIGHEST_PROTOCOL)

    def save_override_character_list(self, character_list):
        self.characters = character_list
        with open("characters.pkl", "wb") as output:  # Overwrites any existing file.
            pickle.dump(self.characters, output, pickle.HIGHEST_PROTOCOL)

    def get_character_by_name_from_list(self, list_of_characters, character_name):
        """List of character and a name return a character"""
        for character in list_of_characters:
            if character.name == character_name:
                return character
        return False

    def load_relations(self):
        with open("relations.pkl", "rb") as input:
            self.relations = pickle.load(input)
            return self.relations

    def save_relation(self, relation):
        self.relations.append(relation)
        with open("relations.pkl", "wb") as output:  # Overwrites any existing file.
            pickle.dump(self.relations, output, pickle.HIGHEST_PROTOCOL)

    def save_relation_list(self, relation_list):
        """appends a list of relations to the already existing  list of relations, which are written into the relations.pkl file"""
        for relation in relation_list:
            self.relations.append(relation)
        with open("relations.pkl", "wb") as output:  # Overwrites any existing file.
            pickle.dump(self.relations, output, pickle.HIGHEST_PROTOCOL)

    def save_override_relation_list(self, relation_list):
        """overrides the list of relations in the pickle FIle and replaces it with the relation list"""
        self.relations = relation_list
        with open("relations.pkl", "wb") as output:  # Overwrites any existing file.
            pickle.dump(self.relations, output, pickle.HIGHEST_PROTOCOL)

    def clear_duplicates(self):

        """ because the way the character list is created, we need to summarize each occournce under the same object."""
        temp = []
        created_names = []
        created_chars = []
        """create a temp list, which contains a tuple like this ('ye bite', 'ye') with the first element being the name of the character and the second being the
        only relation the character has"""

        for i in self.characters:
            temp.append((i.name, i.get_relations()[0].get_target_character().name))

        """remove every second element in temp because it has following structure:
        ('ye', 'Jesus')
        ('Jesus', 'ye')
        ('Jesus', 'Caiaphas')
        ('Caiaphas', 'Jesus')
        """

        del temp[::2]

        """we iterate over the list and create the character, if it hasnt been created"""
        while temp:
            element = temp.pop()
            first_character_name = element[0]
            second_character_name = element[1]

            """check if we created the character already"""
            first_character_created = first_character_name in created_names
            second_character_created = second_character_name in created_names

            """if both are true we try to see if they have already have a relationship. 
            important to remeber is that relationships are bidirectional"""

            if first_character_created == True and second_character_created == True:
                c1 = self.get_character_by_name_from_list(
                    created_chars, first_character_name
                )
                c2 = self.get_character_by_name_from_list(
                    created_chars, second_character_name
                )
                relationlist = c1.get_relations()
                if second_character_name in relationlist:
                    """The relationship is already alive"""
                    pass
                else:
                    """create a relationship"""
                    c1.add_relation(c2)
                    c2.add_relation(c1)

            """ both character dont exist so we have to create both of them"""
            if first_character_created == False and second_character_created == False:
                c1 = Character(first_character_name)
                c2 = Character(second_character_name)
                c1.add_relation(c2)
                c2.add_relation(c1)

                """append the character and the names to the approp. lists"""
                created_names.extend((first_character_name, second_character_name))
                created_chars.extend((c1, c2))

            """first character exist and second doenst"""
            if first_character_created == True and second_character_created == False:
                c1 = self.get_character_by_name_from_list(
                    created_chars, first_character_name
                )
                c2 = Character(second_character_name)
                c1.add_relation(c2)
                c2.add_relation(c1)

                created_names.append(second_character_name)
                created_chars.append(c2)

            """second character exists and first doesnt"""
            if first_character_created == False and second_character_created == True:
                c2 = self.get_character_by_name_from_list(
                    created_chars, second_character_name
                )
                c1 = Character(first_character_name)
                c1.add_relation(c2)
                c2.add_relation(c1)

                created_names.append(first_character_name)
                created_chars.append(c1)

        """Example for what Relationships Jesus has """
        print("Jesus")
        for i in self.get_character_by_name_from_list(
            created_chars, "Jesus"
        ).get_relations():
            print(i.target_character.name)

        """at the end save the created_chars """
        self.characters = created_chars
        with open("characters.pkl", "wb") as output:  # Overwrites any existing file.
            pickle.dump(self.characters, output, pickle.HIGHEST_PROTOCOL)
