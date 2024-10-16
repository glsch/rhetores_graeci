from datetime import datetime
import json
import os
from typing import Union
from dotenv import load_dotenv

class PathManager:

    _instance = None
    root_path = None # root of the project
    data_path = None # data_df folder
    raw_path = None # raw data_df folder (ie restricted XML files)
    preprocessed_path = None # preprocessed data_df folder (ie DF with all split into sentences)
    output_path = None # output folder (for models, logs, visualizations, etc.)
    models_path = None # models folder
    access_key = None # access key to download the data_df data_df
    hub_token = None # Huggingface access token
    templates_path = None # templates folder (ie templates for reports)
    scripts_path = None # scripts folder (ie scripts for seamless execution of the project)
    dataset_name = None # dataset name (will be looked for in the preprocessed folder)
    dataset_path = None # combination of preprocessed path and dataset name
    author_metadata_name = None # author metadata name (will be looked for in the preprocessed folder)
    author_metadata_path = None # combination of preprocessed path and author metadata name
    model_basename = None
    tokenizer_basename = None
    classifier_basename = None

    def __new__(cls):
        location = os.getenv("LOCATION", "local")

        if cls._instance is None:
            cls._instance = super(PathManager, cls).__new__(cls)

            with open("paths.json", 'r') as f:
                config = json.load(f)

                cls.root_path = cls.path_exists(config["path"]["root"], create_path=False)
                cls.data_path = cls.path_exists(config["path"]['data'][location], create_path=True)
                cls.raw_path = cls.path_exists((os.path.join(cls.data_path, config["path"]['raw'])), create_path=True)
                cls.preprocessed_path = cls.path_exists((os.path.join(cls.data_path, config["path"]['preprocessed'])), create_path=True)
                cls.output_path = cls.path_exists(config["path"]['output'][location], create_path=True)
                cls.models_path = cls.path_exists(os.path.join(cls.output_path, config["path"]['models']), create_path=True)
                cls.templates_path = cls.path_exists(config["path"]['templates'], create_path=True)
                cls.scripts_path = cls.path_exists(config["path"]['scripts'], create_path=True)
                cls.access_key = config["path"]["access_key"]
                cls.hub_token = config["path"]["hub_token"]
                cls.dataset_name = config["name"]["dataset"]
                cls.author_metadata_name = config["name"]["author_metadata"]
                cls.dataset_path = os.path.join(cls.preprocessed_path, cls.dataset_name)
                cls.author_metadata_path = os.path.join(cls.preprocessed_path, cls.author_metadata_name)
                cls.model_basename = config["name"]["model"]
                cls.tokenizer_basename = config["name"]["tokenizer"]
                cls.classifier_basename = config["name"]["classifier"]

        return cls._instance

    @classmethod
    def get_experiment_id(cls):
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls()
        return cls._instance

    @classmethod
    def change_location(cls, location: str):
        os.environ["LOCATION"] = location
        cls._instance = None
        return cls.get_instance()

    @classmethod
    def path_exists(cls, path: Union[str], create_path=False):
        """Checks if a path exist. If it does, it will return it.

        Args:
            path (str): Path to create or check existence for.
            create_path (bool, optional): If True, will create the path if it doesn't exist.
                Defaults to False.
        """
        if os.path.exists(path) == True:
            return path

        if os.path.exists(path) is False and create_path == True:
            os.makedirs(path)

            return path
        else:
            raise ValueError("The following path doesn\'t exist: {}".format(path))

_path_manager_instance = PathManager.get_instance()

if __name__ == "__main__":

    pm = PathManager.get_instance()
    print(pm.get_experiment_id())
