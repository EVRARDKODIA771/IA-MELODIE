# utils/logger.py
import sys
import datetime

def log(message):
    """
    Log simple sur stdout avec timestamp
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", file=sys.stdout)
