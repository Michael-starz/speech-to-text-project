import os
import logging
from logging.handlers import RotatingFileHandler


os.makedirs("logs", exist_ok=True) # Creates logs directory if it doesn't exist

# Define logger
logger = logging.getLogger("transcription_logger")
logger.setLevel(logging.INFO)

# Rotating file handler: 1MB per file, keep 5 backups
file_handler = RotatingFileHandler(
    "logs/app.log",
    maxBytes=1 * 1024 * 1024,
    backupCount=5
)
file_handler.setLevel(logging.INFO)

# Console handler for Dev mode
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)

# Format log
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Avoids duplicate logs if this module is re-imported
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
