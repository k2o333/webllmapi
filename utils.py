import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up logging configuration for the application.
    
    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file (Optional[str]): Path to log file. If None, logs only to console
        max_bytes (int): Maximum size of log file before rotation
        backup_count (int): Number of backup files to keep
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
    
    # Get logger
    logger = logging.getLogger("wrapper_api")
    
    # Convert string log level to logging constant
    log_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file is specified)
    if log_file:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    # Log initial message
    logger.info(f"Logging setup completed. Level: {logging.getLevelName(log_level)}")
    if log_file:
        logger.info(f"Log file: {log_file}")
    
    return logger

def get_env_log_level() -> str:
    """
    Get log level from environment variable or return default.
    
    Returns:
        str: Log level string
    """
    return os.getenv("LOG_LEVEL", "INFO").upper()

# Example usage
if __name__ == "__main__":
    # Setup logging with both console and file output
    logger = setup_logging(
        log_level=get_env_log_level(),
        log_file="logs/wrapper_api.log"
    )
    
    # Test logging
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")