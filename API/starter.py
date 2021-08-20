import os
import subprocess
import signal
import time
from core.config import MONGODB_URL, CONFIG_COLLECTION, API_PORT, API_HOST, API_WORKER
import pymongo

if os.path.isdir("./mongodb") == False:
    os.mkdir("./mongodb")


MONGODB_HOST = "0.0.0.0"
slient_commands = [
    #f"mongod --port {MONGODB_PORT} --bind_ip {MONGODB_HOST} --dbpath {MONGODB_PATH}",
]
    
commands = [
    #f"",
    #f"uvicorn app:app --port {API_PORT} --host {API_HOST} --workers {API_WORKER}",
]

if DEBUG:
    pass


processes = []

for command in slient_commands:
    print(command)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    processes.append(process)


for command in commands:
    print(command)
    process = subprocess.Popen(command, shell=True)
    processes.append(process)


while True:
    try:
        time.sleep(1)
    except KeyboardInterrupt:
        for process in processes:
            process.send_signal(signal.SIGINT)
        for process in processes:
            process.wait()
        break
