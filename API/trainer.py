import time
from telegram_notifier import send_message as telegram_bot_sendtext

import os
try:
    for i in range(3):
        time.sleep(1)
        telegram_bot_sendtext(f"2 Running: {i}")
        print("""telegram_bot_sendtext(f"2 Running: {i}")""")
except KeyboardInterrupt:
    telegram_bot_sendtext(f"2 Stop: {i}")
    sys.exit(2)
import sys
sys.exit(0)