"""Class that reads .json configuration file and sets attributes."""
import json


class _NestedObject:
    """For nesting objects as attributes of Config class."""
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):  # If nested dictionary
                setattr(self, key, _NestedObject(value))  # Nest an object
            else:
                setattr(self, key, value)  # Else set attribute to value


class Config(_NestedObject):
    def __init__(self, json_file):
        self.json_file = json_file

        with open(json_file, 'r') as config_file:
            config_dict = json.load(config_file)

        super().__init__(config_dict)
