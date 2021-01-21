class Relation:
    def __init__(self, origin_character, target_character, emotion):
        self.origin_character = origin_character
        self.target_character = target_character
        self.emotion = emotion

    def __abs__(self):
        return self.emotion

    def get_target_character(self):
        return self.target_character

    def get_origin_character(self):
        return self.origin_character

    def get_emotion(self):
        return self.emotion

    def set_emotion(self, value):
        self.emotion = value
