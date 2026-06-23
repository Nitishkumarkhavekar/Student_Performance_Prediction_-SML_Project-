import os
import logging
from datetime import datetime

# Create log file name with current timestamp
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"

# Full path for log file
log_file = os.path.join(os.getcwd(), LOG_FILE)

# Create directory if needed
os.makedirs(os.path.dirname(log_file), exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=log_file,
    encoding='utf-8',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Log messages
logging.debug("Detailed diagnostic information.")
logging.info("Confirmation that things are working.")
logging.warning("An unexpected issue occurred.")