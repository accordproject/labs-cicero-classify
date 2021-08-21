print("Trainer Runner Start!")
import pymongo
import datetime
import os
import subprocess
import signal
import time
from core.config import MONGODB_URL, DATABASE_NAME, CONFIG_COLLECTION, LABEL_RETRAIN_QUEUE_COLLECTION, API_PORT, API_HOST, API_WORKER, SLEEP_INTERVAL_SECOND, ANACONDA_ENV_NAME, PATH

client = pymongo.MongoClient(MONGODB_URL)

from utils.logs import config_log, change_service_status

config_col = client[DATABASE_NAME][CONFIG_COLLECTION]
trainer = config_col.find_one({"name": "trainer"})
trainer_obj = {
        "name": "trainer",
        "status": "down",
        "logs": [],
    }
if not trainer:
    config_col.insert_one(trainer_obj)

def run_trainer():
    return_status = "Finish."
    command = f"""eval "$(conda shell.bash hook)";
        conda activate {ANACONDA_ENV_NAME};
        python {PATH}/trainer.py;"""
    config_log("trainer", "Start Trainer")
    change_service_status("trainer", "up")
    process = subprocess.Popen(command, shell=True)
    while True:
        try:
            try:
                poll = process.poll()
                int(poll)
                if poll == 0:
                    return_status = f"Finish with code {poll}"
                if poll == 4:
                    return_status = f"Error with code {poll}, CUDA out of memory!"
                    time.sleep(600) #if out of memory, don't keep try it...
                else:
                    return_status = f"Error with code {poll}"
                #print(f"Process Stop Running, return code {poll}")
                break
            except TypeError:
                #print("Process Still Running")
                time.sleep(3)
                continue
            
        except KeyboardInterrupt:
            print("KeyboardInterrupt, stop program")
            return_status = "KeyboardInterrupt"
            break

    config_log("trainer", f"Stop trainer because {return_status}")
    change_service_status("trainer", "down")
    process.send_signal(signal.SIGINT)
    process.wait()
    print(f"Stop trainer because {return_status}")
    return return_status

training_queue_col = client[DATABASE_NAME][LABEL_RETRAIN_QUEUE_COLLECTION]
while True:

    import re
    result = training_queue_col.find({
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