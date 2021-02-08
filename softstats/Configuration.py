"""
Object to set configuration options
"""


class Configuration(object):

    def __init__(self, data):
        attributes = []
        for setter in data.keys():
            name = setter.__name__
            setattr(self, f"_{name}", setter)
            self.set(name, data[setter])
            attributes.append(name)
        self._attributes = attributes

    def set(self, key, value):
        setter = getattr(self, f"_{key}")
        setattr(self, key, setter(value))

    def show(self):
        print(self.__str__())

    def __str__(self):
        s = {}
        for attr in self._attributes:
            s[attr] = getattr(self, attr)
        return str(s)
