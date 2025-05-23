import os
import sys
import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')
logger = logging.getLogger(__name__)
logs_dir = "logs"
log_filepath = os.path.join(logs_dir, "running_logs.log")
os.makedirs(logs_dir, exist_ok=True)

logging.basicConfig(
    filename=log_filepath,
    format="[%(asctime)s]: %(levelname)s: %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("MLopsLogger")