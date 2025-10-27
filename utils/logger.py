# utils/logger.py
"""
Enhanced logging utility for AgentOps.
Provides consistent logging across all components with file and console output.
"""

import logging
import os
from datetime import datetime
from typing import Optional


# Global logger configuration
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Create log file with timestamp
LOG_FILE = os.path.join(
    LOG_DIR,
    f"agentops_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get or create a logger with both file and console handlers.
    
    Args:
        name: Logger name (usually module name)
        level: Logging level (default: INFO)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(level)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # File handler (detailed)
        file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        
        # Console handler (simpler)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(simple_formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
    
    return logger


def set_log_level(logger: logging.Logger, level: int):
    """
    Set logging level for a logger.
    
    Args:
        logger: Logger instance
        level: New logging level
    """
    logger.setLevel(level)
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(level)


# Module-level logger for this file
logger = get_logger(__name__)
logger.info(f"Logging initialized. Log file: {LOG_FILE}")
