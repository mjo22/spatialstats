"""
Object to set configuration options
"""

import warnings
import toml
import os


class Configuration(object):

    def __init__(self, fn, setters=[], *args, **kwargs):
        settings = toml.load(fn)
        for setter in setters:
            setattr(self, setter.__name__, setter)
        for attr in settings.keys():
            self.set(attr, settings[attr])
        self._attributes = list(settings.keys())

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

    def _warn(self, action):
        warnings.simplefilter(action)
        return action


class immutable_dict(dict):

    def __hash__(self):
        return id(self)

    def _immutable(self, *args, **kws):
        raise TypeError("Operation is immutable")

    __delitem__ = _immutable
    clear = _immutable
    update = _immutable
    setdefault = _immutable
    pop = _immutable
    popitem = _immutable


def get_config(init_path):
    return os.path.join(os.path.dirname(init_path),
                        f"{os.path.basename(os.path.dirname(init_path))}.toml")
