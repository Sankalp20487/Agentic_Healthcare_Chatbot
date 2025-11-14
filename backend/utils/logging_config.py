"""
Production logging configuration for Healthcare Chat Assistant.
Provides structured logging to both console and file with appropriate levels.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Configure structured logging for the application.
    
    Creates a logger with two handlers:
    - Console handler for user-facing logs (INFO level)
    - File handler for detailed debugging (DEBUG level)
    
    Args:
        log_level: Logging level for console output (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Configured logger instance
    """
    # Create logs directory
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("healthcare_assistant")
    logger.setLevel(logging.DEBUG)  # Capture everything, filter by handlers
    
    # Prevent duplicate handlers if called multiple times
    if logger.handlers:
        return logger
    
    # Console Handler (INFO level) - Clean output for users
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    
    # File Handler (DEBUG level) - Detailed logs for debugging
    log_file = os.path.join(
        log_dir, 
        f"assistant_{datetime.now().strftime('%Y%m%d')}.log"
    )
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_format)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    logger.info(f"Logging initialized - Console: {log_level}, File: DEBUG")
    logger.info(f"Log file: {log_file}")
    
    return logger