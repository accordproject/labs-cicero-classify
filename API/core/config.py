import os

try:
    from dotenv import load_dotenv
    load_dotenv(".env")
except ModuleNotFoundError:
    # No dotenv
    pass
MAX_CONNECTIONS_COUNT = int(os.getenv("MAX_CONNECTIONS_COUNT", 10))
MIN_CONNECTIONS_COUNT = int(os.getenv("MIN_CONNECTIONS_COUNT", 10))
#SECRET_KEY = Secret(os.getenv("SECRET_KEY", "secret key for project"))

PROJECT_NAME = os.getenv("PROJECT_NAME", "Accord Project ML model API")
PROJECT_VERSION = os.getenv("PROJECT_VERSION", "0.1.1")
ALLOWED_HOSTS = ["*"]

API_PORT = 13537
API_HOST = "0.0.0.0"
API_WORKER = 5

DEBUG = False

# MongoDB
MONGODB_PORT = os.getenv("MONGODB_PORT", 27017)
MONGODB_HOST = os.getenv("MONGODB_HOST", "localhost")
MONGODB_USERNAME = os.getenv("MONGO_USER", "user")
MONGODB_PASSWORD = os.getenv("MONGO_PASSWORD", "")

MONGODB_URL = f"mongodb://{MONGODB_USERNAME}:{MONGODB_PASSWORD}@{MONGODB_HOST}:{MONGODB_PORT}"
        
DATABASE_NAME = "Accord_Project"
NER_LABEL_COLLECTION = "labeled_dataset"
Feedback_Template_Collection = "template_data_feedback"
Feedback_Suggestion_Collection = "suggestion_data_feedback"
LABEL_COLLECTION = "Labels"
LABEL_TRAIN_JOB_COLLECTION = "NER_label_training_jobs"
CONFIG_COLLECTION="config"
SLEEP_INTERVAL_SECOND = 3

TRAINER_LOG_COLLECTION = "trainer_log"

# NER Trainer
NER_TRAIN_BATCH_SIZE = 256
NER_TRAIN_DEFAULT_FILTER = {}
NER_TRAIN_DEVIDE_ID = 0
NER_ADAPTERS_PATH = "."

# Anaconda
ANACONDA_ENV_NAME = "adapter"


# ENV
PATH = os.getcwd()

# CACHE
NER_TRAINER_DATA_CATCH_FILE=f"{PATH}/cache/NER_TRAINER_DATA_CATCH.csv"
NER_ADAPTERS_TRAINER_NAME = "NER_adapter_trainer"
NER_TRAINER_RUNNER_NAME = "NER_trainer_runner"

# API Version
API_V1_PREFIX="/api/v1"