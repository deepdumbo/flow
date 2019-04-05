from pathlib import Path
import json
import __main__


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

        # Set directories according to directory of main file
        self.main_file = Path(__main__.__file__).resolve()
        self.flow_dir = self.main_file.parents[3]
        self.experiment_name = self.main_file.parts[-2]
        self.experiment_dir = self.flow_dir / self.experiment_name
        self.results_dir = self.experiment_dir/'results'
        print('PRINTING', self.results_dir.is_dir())
        self.history_filename = self.results_dir/'history.pickle'
        self.save_dir = self.experiment_dir/'saved_models'
        print('PRINTING', self.save_dir.is_dir())
        self.model_path = self.save_dir/self.model_name
        if not self.results_dir.is_dir():
            self.results_dir.mkdir(parents=True)
        if not self.save_dir.is_dir():
            self.save_dir.mkdir(parents=True)
        print('PRINTING', self.history_filename)
        print('PRINTING', self.model_path)
        print('PRINTING', self.main_file)
