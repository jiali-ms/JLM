"""For sharing directory paths among different files.
If there is any need to modify directory structure etc., modifying this file should handle all dependencies."""

import os
import json

'''
root_path
  data
  train 
  decoder
'''

cd = os.path.dirname(__file__)
root_path = cd
print("root path of project: {}".format(root_path))
train_path = os.path.abspath(os.path.join(root_path, "train"))
data_path = os.path.abspath(os.path.join(root_path, "data"))
experiment_path = os.path.abspath(os.path.join(train_path, "experiments"))

class ExperimentConfig:
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __repr__(self):
        return str(self.__dict__)

def get_configs(experiment):
    config_data = json.loads(open(os.path.join(experiment_path, str(experiment), "config.json"), "rt").read())
    return ExperimentConfig(**config_data)