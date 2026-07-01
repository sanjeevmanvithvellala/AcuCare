import os
import logging
from logging.handlers import RotatingFileHandler

def setup_logger(name: str = "healthcare_platform", log_file: str = "logs/app.log", level=logging.INFO) -> logging.Logger:
    """Sets up a logger that outputs to both console and a rotating log file."""
    # Ensure logs directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
        
    logger = logging.getLogger(name)
    
    # If logger is already configured, return it
    if logger.handlers:
        return logger
        
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (rotating logs, max 5MB, keep 5 backups)
    try:
        file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not create file log handler: {e}")
        
    return logger

# Create a default logger instance
logger = setup_logger()
