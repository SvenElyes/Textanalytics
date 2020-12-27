class Relation:
    def __init__(self, origin_character, target_character):
        self.origin_character = origin_character
        self.target_character = target_character
        self.liking = 0
        self.intensity = 0

    def __abs__(self):
        return self.intensity

    def __lt__(self, other):
        if self.liking >= other.liking:
            return False
        else:
            return True

    def __gt__(self, other):
        if self.liking <= other.liking:
            return False
        else:
            return True

    def get_target_character(self):
        return self.target_character

    def influence_positive(self):
        self.liking += 1

    def influence_negative(self):
        self.liking -= 1

    def intentify(self):
        self.intensity += 1

    def get_liking(self):
        return self.liking

    def set_liking(self, value):
        self.liking = value

    def get_intensity(self):
        return self.intensity

    def set_intensity(self, value):
        self.intensity = value

    # TODO
    """
    liking = property(get_liking, set_liking)
    intensity = property(get_intensity, set_intensity)
    """
