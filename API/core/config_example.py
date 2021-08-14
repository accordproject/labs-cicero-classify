import os

from dotenv import load_dotenv
from starlette.datastructures import CommaSeparatedStrings, Secret

load_dotenv(".env")

MAX_CONNECTIONS_COUNT = int(os.getenv("MAX_CONNECTIONS_COUNT", 10))
MIN_CONNECTIONS_COUNT = int(os.getenv("MIN_CONNECTIONS_COUNT", 10))
#SECRET_KEY = Secret(os.getenv("SECRET_KEY", "secret key for project"))

PROJECT_NAME = os.getenv("PROJECT_NAME", "Accord Project ML model API")
PROJECT_VERSION = os.getenv("PROJECT_VERSION", "0.1.1")
ALLOWED_HOSTS = CommaSeparatedStrings(os.getenv("ALLOWED_HOSTS", "*"))

API_PORT = 13537
API_HOST = "0.0.0.0"
API_WORKER = 5

DEBUG = False

#chage to yours
MONGODB_PORT = 27017
MONGODB_HOST = "localhost"

MONGODB_USERNAME = ""
MONGODB_PASSWORD = ""

MONGODB_URL = f"mongodb://{MONGODB_USERNAME}:{MONGODB_PASSWORD}@{MONGODB_HOST}:{MONGODB_PORT}"
        
DATABASE_NAME = "Accord_Project"
Feedback_Label_Collection = "label_data_feedback"
Feedback_Template_Collection = "template_data_feedback"
Feedback_Suggestion_Collection = "suggestion_data_feedback"