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
    """class for handling the creation and handling .pkl files for saving progress when working with charactres
    and relations
    """

    def __init__(self):
        """initiate a Picklehandler that has 2 lists
        one for characters and one for relations
        """
        self.characters = list()
        self.relations = list()

    def load_characters(self):
        """load a list of characters from the characters.pkl file

        Returns:
            list: a list of characters
        """
        with open("characters.pkl", "rb") as input:
            self.characters = pickle.load(input)
            return self.characters

    def load_character_by_name(self, name):
        """returns a single character or False, given the name of the object

        Args:
            name (string): The name of the character

        Returns:
            [Character]: character object with matching name, or False, if not found
        """
        for character in self.characters:
            if character.name == name:
                return character
        return False

    def save_character(self, character):
        """Saves a single character into the .pkl file and the character list attribute

        Args:
            character (Character): Character object
        """
        self.characters.append(character)
        with open("characters.pkl", "wb") as output:  # Overwrites any existing file.
            pickle.dump(self.characters, output, pickle.HIGHEST_PROTOCOL)

    def save_character_list(self, character_list):
        """Saves a list of characters towards the .pkl file and the character list attribute

        Args:
            character_list (list(charcter)): a list of characters to be saved
        """
        for character in character_list:
            self.characters.append(character)
        with open("characters.pkl", "wb") as output:  # Overwrites any existing file.
            pickle.dump(self.characters, output, pickle.HIGHEST_PROTOCOL)

    def save_override_character_list(self, character_list):
        """override the list of characters in the .pkl file and charcter list attribute.

        Args:
            character_list (list(character)): a list of characters
        """
        self.characters = character_list
        with open("characters.pkl", "wb") as output:  # Overwrites any existing file.
            pickle.dump(self.characters, output, pickle.HIGHEST_PROTOCOL)

    def get_character_by_name_from_list(self, list_of_characters, character_name):
        """returns a character from a list of character, chosen by name

        Args:
            list_of_characters (list(characters)): list of characters, in which the charactre is looked for
            character_name (string): character name

        Returns:
            [Character]: character object or false
        """
        for character in list_of_characters:
            if character.name == character_name:
                return character
        return False

    def load_relations(self):
        """load the list of relations from the .pkl file

        Returns:
            [list(relations)]: list of all relations
        """
        with open("relations.pkl", "rb") as input:
            self.relations = pickle.load(input)
            return self.relations

    def save_relation(self, relation):
        """save a single relation to the PickleHandler attribute list and
        the .pkl file

        Args:
            relation (Relation): single relation object
        """
        self.relations.append(relation)
        with open("relations.pkl", "wb") as output:  # Overwrites any existing file.
            pickle.dump(self.relations, output, pickle.HIGHEST_PROTOCOL)

    def save_relation_list(self, relation_list):
        """Saves a relation list to the PickleHandler attribute and the
        .pkl file

        Args:
            relation_list (list(Relations)): list of relations
        """
        for relation in relation_list:
            self.relations.append(relation)
        with open("relations.pkl", "wb") as output:  # Overwrites any existing file.
            pickle.dump(self.relations, output, pickle.HIGHEST_PROTOCOL)

    def save_override_relation_list(self, relation_list):
        """Override the attribute and the .pkl file content with a new list


        Args:
            relation_list (list(relations)): relation list
        """
        self.relations = relation_list
        with open("relations.pkl", "wb") as output:  # Overwrites any existing file.
            pickle.dump(self.relations, output, pickle.HIGHEST_PROTOCOL)

    def clear_duplicates(self):
        """Deprecetad function that hat the goal of removing duplicate characters in the .pkl file
        we now dont use the function because eval_graph.py gives a distilled csv, removing the concept of duplicates

        The function looks for multiple occurunces of the same name, and saves all the relations of that character under the same ONE
        character.

        This function however is not equipped with handling the numerical value we assigned to relations, bc it was written later on"""

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
