import csv
import time
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from models import Mapping
from db import Base, engine, Session


class Utils:
    def __init__(self) -> None:
        pass

    @staticmethod
    def read(csv_file, database_model):
        """Returns a list of database objects after reading the CSV file.

        Parameters:
            csv_file (str): Path to the CSV file to read.
            database_model (Type[Base]): The database model to use to create the objects.

        Returns:
            List[database_model]: A list of database objects.
        """
        with open(csv_file, newline='') as file:
            filereader = csv.DictReader(file)
            data = []

            for row in filereader:
                current_data_obj = database_model(**row)
                data.append(current_data_obj)
        return data

    @staticmethod
    def write_deviation_results_to_db(results):
        """Writes the classification results to the database.

        Parameters:
            results (List[Dict]): A list of dictionaries with the result of a classification test. Each dictionary
                should have the following keys: 'point', 'classification', and 'delta_y'.
        """
        execute_map = []

        for item in results:
            point = item['point']
            classification = item['classification']
            delta_y = item["delta_y"]

            # Examine for any classification point and rename function accordingly.
            # Dash represents missing classification
            classification_name = None
            if classification is not None:
                classification_name = classification.name.replace("y", "N")
            else:
                classification_name = "-"
                delta_y = -1

            # Alternative dictionary is established for stocking the mapping data
            current_mapping = {"x": point["x"], "y": point["y"], "delta": delta_y,
                               "no_of_ideal_func": classification_name}

            obj = Mapping(**current_mapping)
            execute_map.append(obj)
        session = sessionmaker(bind=engine)
        local_session = session()
        local_session.add_all(execute_map)
        local_session.commit()


class DBUtils:
    @staticmethod
    def create_db():
        """Creates the database tables for the IdeaFunction, TrainingData, and Mapping models.
        """
        from models import IdeaFunction, TrainingData, Mapping
        Base.metadata.create_all(bind=engine)

    @staticmethod
    def populate_db(csv_file, db_model):
        """Populates the database with objects created from the data in the CSV file.

        Parameters:
            csv_file (str): Path to the CSV file to read.
            db_model (Type[Base]): The database model to use to create the objects.
        """
        data = Utils.read(csv_file, db_model)
        DBUtils.load_into_db(data)

    @staticmethod
    def load_into_db(data):
        """Loads a list of objects into the database.

        Parameters:
            data (List[Base]): The list of objects to load into the database.
        """
        session = sessionmaker(bind=engine)
        local_session = session()
        local_session.add_all(data)
        local_session.commit()
