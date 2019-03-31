import os
import sys
import json


class _NestedObject:
    """For nesting objects as attributes of Config."""
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):  # If nested dictionary
                setattr(self, key, _NestedObject(value))  # Nest an object
            else:
                setattr(self, key, value)  # Else set attribute to value


class Config(_NestedObject):
    """Read .json configuration file and set attributes from dictionary."""
    def __init__(self, json_file):
        self.json_file = json_file

        with open(json_file, 'r') as config_file:
            config_dict = json.load(config_file)

        super().__init__(config_dict)

        # Set directories according to directory of execution
        self.experiment_dir = sys.path[0]
        self.results_dir = f'{self.experiment_dir}/results'
        if not os.path.isdir(self.results_dir):
            os.makedirs(self.results_dir)
        self.history_filename = f'{self.results_dir}/history.pickle'
        self.save_dir = f'{self.experiment_dir}/saved_models'
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        self.model_path = f'{self.save_dir}/{self.model_name}'
