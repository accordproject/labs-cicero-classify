print("NER Trainer Runner Start!")
import pymongo
import datetime
import os
import subprocess
import signal
import time
from core.config import MONGODB_URL, DATABASE_NAME, CONFIG_COLLECTION, LABEL_TRAIN_JOB_COLLECTION, API_PORT, API_HOST, API_WORKER, SLEEP_INTERVAL_SECOND, ANACONDA_ENV_NAME, PATH, NER_ADAPTERS_TRAINER_NAME, NER_TRAINER_RUNNER_NAME
from utils.trainer_communicate import update_pid

update_pid(NER_TRAINER_RUNNER_NAME, os.getpid())


client = pymongo.MongoClient(MONGODB_URL)

from utils.logs import config_log, change_service_status

config_col = client[DATABASE_NAME][CONFIG_COLLECTION]



trainer = config_col.find_one({"name": NER_ADAPTERS_TRAINER_NAME})

if not trainer:
    trainer_obj = {
        "name": NER_ADAPTERS_TRAINER_NAME,
        "status": "down",
        "restart_required": False,
        "last_data_cache_timestamp": datetime.datetime.strptime("1999", "%Y"),
        "logs": [],
    }
    config_col.insert_one(trainer_obj)

def run_trainer():
    return_status = "Finish."
    command = f"""eval "$(conda shell.bash hook)";
        conda activate {ANACONDA_ENV_NAME};
        python {PATH}/NER_trainer.py;"""
    config_log(NER_ADAPTERS_TRAINER_NAME, "Start {NER_ADAPTERS_TRAINER_NAME} from runner")
    change_service_status(NER_ADAPTERS_TRAINER_NAME, "up")
    process = subprocess.Popen(command, shell=True)
    while True:
        try:
            trainer = config_col.find_one({
                "name": NER_ADAPTERS_TRAINER_NAME
            }, {
                "restart_required": True
            })
            if trainer["restart_required"]:
                config_col.update_one({
                    "name": NER_ADAPTERS_TRAINER_NAME
                },{
                    "$set": {"restart_required": False}
                })
                return_status = "Trainer Restart Required."
                time.sleep(3)
                break
            try:
                poll = process.poll()
                poll = int(poll)
                if poll == 0:
                    return_status = f"Finish with code {poll}"
                elif poll == 4:
                    return_status = f"Error with code {poll}, CUDA out of memory!"
                    #time.sleep(600) #if out of memory, don't keep try it...
                else:
                    return_status = f"Error with code {poll}"
                #print(f"Process Stop Running, return code {poll}")
                break
            except TypeError:
                #print("Process Still Running")
                time.sleep(SLEEP_INTERVAL_SECOND)
                #time.sleep(3)
                continue
            
        except KeyboardInterrupt:
            print("KeyboardInterrupt, stop program")
            return_status = "KeyboardInterrupt"
            break
    trainer = config_col.find_one({
        "name": NER_ADAPTERS_TRAINER_NAME
    }, {"pid": True})
    try:
        os.kill(trainer["pid"], signal.SIGINT)
    except:
        pass
    config_log("trainer", f"Stop {NER_ADAPTERS_TRAINER_NAME} because {return_status}")
    change_service_status(NER_ADAPTERS_TRAINER_NAME, "down")
    process.wait()
    print(f"Stop {NER_ADAPTERS_TRAINER_NAME} because {return_status}")
    return return_status

training_job_col = client[DATABASE_NAME][LABEL_TRAIN_JOB_COLLECTION]
while True:
    try:
        import re
        result = training_job_col.find({
            "status":  re.compile("(training|waiting)")
        })

        result = list(result)

        if result:
            return_status = run_trainer()
            if return_status == "KeyboardInterrupt":
                break
            else:
                continue
        time.sleep(SLEEP_INTERVAL_SECOND)
    except KeyboardInterrupt:
        print("NER Trainer Runner Stop!")
        break