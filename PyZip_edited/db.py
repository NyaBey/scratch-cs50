from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import os

#path to current directory
base_dir =os.path.dirname(os.path.realpath(__file__))

#create the db URL
db_uri = "sqlite:///" + os.path.join(base_dir,'main.db')


#create an engine
engine = create_engine(db_uri)

#create the session to help us do crud
Session = sessionmaker()

#create the base class to help us create model classes
Base = declarative_base()