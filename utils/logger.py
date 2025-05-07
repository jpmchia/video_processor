#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Logging utilities
"""

import logging
import os
import sys
from datetime import datetime


def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Set up a logger with console and file handlers
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is specified
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Create file handler
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(level)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Log start message
    logger.info(f"Logger initialized: {name}")
    
    return logger


class JupyterHandler(logging.Handler):
    """
    Custom logging handler for Jupyter notebooks
    """
    def __init__(self, level=logging.INFO):
        super().__init__(level)
        self.log_widget = None
        self.logs = []
    
    def set_widget(self, widget):
        """Set the widget to display logs in"""
        self.log_widget = widget
        self.update_widget()
    
    def emit(self, record):
        """Process a log record"""
        log_entry = self.format(record)
        self.logs.append(log_entry)
        self.update_widget()
    
    def update_widget(self):
        """Update the widget with logs"""
        if self.log_widget:
            # Format logs with appropriate colors based on level
            formatted_logs = []
            for log in self.logs[-100:]:  # Keep only the last 100 logs to avoid performance issues
                if "ERROR" in log:
                    formatted_logs.append(f'<div style="color: #ff5252;">{log}</div>')
                elif "WARNING" in log:
                    formatted_logs.append(f'<div style="color: #ffab40;">{log}</div>')
                elif "INFO" in log:
                    formatted_logs.append(f'<div style="color: #4caf50;">{log}</div>')
                else:
                    formatted_logs.append(f'<div>{log}</div>')
            
            # Update widget value
            self.log_widget.value = f'<div style="height: 200px; overflow-y: auto; font-family: monospace;">{"".join(formatted_logs)}</div>'
