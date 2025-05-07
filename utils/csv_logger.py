#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CSV logging functionality for tracking processed videos
"""

import os
import csv
import json
import logging


def initialize_processing_log(subfolder_path, video_files):
    """
    Initialize or update the processing log CSV file for a subfolder.
    Creates a new CSV file if it doesn't exist, or updates it with new files.
    
    Args:
        subfolder_path: Path to the subfolder containing videos
        video_files: List of video file paths to process
        
    Returns:
        dict: Dictionary mapping filenames to their processing status
    """
    logger = logging.getLogger("video_processor")
    log_file = os.path.join(subfolder_path, "processing_log.csv")
    processed_files = {}
    
    # Check if log file exists
    if os.path.exists(log_file):
        logger.info(f"Found existing processing log: {log_file}")
        # Read existing log
        with open(log_file, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                filename = row['filename']
                processed_files[filename] = row
                # Convert segment files from JSON string to list if it exists
                if row.get('segment_files') and row['segment_files'].strip():
                    try:
                        processed_files[filename]['segment_files'] = json.loads(row['segment_files'])
                    except json.JSONDecodeError:
                        processed_files[filename]['segment_files'] = []
    
    # Create file with headers if it doesn't exist
    if not os.path.exists(log_file):
        logger.info(f"Creating new processing log: {log_file}")
        with open(log_file, 'w', newline='') as csvfile:
            fieldnames = ['filename', 'processed_datetime', 'objects', 'motion', 'segment_files']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    
    # Add new files to the log if they don't exist
    new_files_added = False
    with open(log_file, 'a', newline='') as csvfile:
        fieldnames = ['filename', 'processed_datetime', 'objects', 'motion', 'segment_files']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        for video_path in video_files:
            filename = os.path.basename(video_path)
            if filename not in processed_files:
                row = {
                    'filename': filename,
                    'processed_datetime': '',
                    'objects': '',
                    'motion': '',
                    'segment_files': ''
                }
                writer.writerow(row)
                processed_files[filename] = row
                new_files_added = True
    
    if new_files_added:
        logger.info(f"Added new files to processing log")
    
    return processed_files


def update_processing_log(subfolder_path, filename, data):
    """
    Update the processing log for a specific file
    
    Args:
        subfolder_path: Path to the subfolder containing videos
        filename: Name of the video file that was processed
        data: Dictionary containing updated data (processed_datetime, objects, motion, segment_files)
    """
    logger = logging.getLogger("video_processor")
    log_file = os.path.join(subfolder_path, "processing_log.csv")
    temp_file = os.path.join(subfolder_path, "processing_log_temp.csv")
    
    # Convert segment_files list to JSON string if it exists
    if 'segment_files' in data and isinstance(data['segment_files'], list):
        data['segment_files'] = json.dumps(data['segment_files'])
    
    # Read the existing file and update the specific row
    updated = False
    with open(log_file, 'r', newline='') as infile, open(temp_file, 'w', newline='') as outfile:
        fieldnames = ['filename', 'processed_datetime', 'objects', 'motion', 'segment_files']
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in reader:
            if row['filename'] == filename:
                # Update the row with new data
                for key, value in data.items():
                    row[key] = value
                updated = True
            writer.writerow(row)
    
    # If the file wasn't found (shouldn't happen), add it
    if not updated:
        with open(temp_file, 'a', newline='') as outfile:
            fieldnames = ['filename', 'processed_datetime', 'objects', 'motion', 'segment_files']
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            row = {
                'filename': filename,
                'processed_datetime': '',
                'objects': '',
                'motion': '',
                'segment_files': ''
            }
            # Update with provided data
            for key, value in data.items():
                row[key] = value
            writer.writerow(row)
    
    # Replace the original file with the updated one
    os.replace(temp_file, log_file)
    
    if data.get('processed_datetime'):
        logger.info(f"Updated processing log for {filename}: processed successfully")
    else:
        logger.info(f"Updated processing log for {filename}: marked for retry")
