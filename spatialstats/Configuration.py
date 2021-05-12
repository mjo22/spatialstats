"""
The spatialstats configuration object
"""


class Configuration(object):
    """
    Object that dynamically sets its own properties
    and stores a collection of their user-specified setters.
    """
    def __init__(self, setters):
        properties, defaults = [], []
        for setter in setters.keys():
            prop = setter.__name__
            setattr(self, f"_set_{prop}", setter)
            properties.append(prop)
            defaults.append(setters[setter])
        self.PROPERTIES = tuple(properties)
        for i in range(len(properties)):
            setattr(self, properties[i], defaults[i])

    def __setattr__(self, attr, value):
        if hasattr(self, "PROPERTIES"):
            if attr in self.PROPERTIES:
                setter = getattr(self, f"_set_{attr}")
                value = setter(value)
        return object.__setattr__(self, attr, value)

    def show(self):
        print(self.__str__())

    def __str__(self):
        s = {}
        for prop in self.PROPERTIES:
            s[prop] = getattr(self, prop)
        return str(s)
