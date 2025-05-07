#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Core video processing functionality
"""

import os
import time
import torch
import logging
import numpy as np
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import gc
import psutil

from IPython.display import display, HTML
import ipywidgets as widgets

from video_processor.core.video import process_video
from video_processor.utils.csv_logger import initialize_processing_log, update_processing_log
from video_processor.utils.memory import check_memory_usage


def process_subfolder(subfolder_path, output_base_dir, model, config=None, max_workers=None, 
                     memory_limit_percent=85, video_progress_widgets=None, dark_mode=False):
    """Process all videos in a subfolder with parallel processing
    
    Args:
        subfolder_path: Path to the subfolder containing videos
        output_base_dir: Base directory for output clips
        model: YOLO model to use for detection
        config: Configuration dictionary for processing
        max_workers: Maximum number of parallel workers (None for auto)
        memory_limit_percent: Memory usage percentage to trigger cleanup
        video_progress_widgets: Dictionary of progress widgets for each video
        dark_mode: Whether to use dark mode for UI elements
        
    Returns:
        List of extracted video clips
    """
    global processing_cancelled
    processing_cancelled = False
    
    # Set up logger
    logger = logging.getLogger("video_processor")
    
    # Create output directory
    rel_path = os.path.basename(subfolder_path)
    output_dir = os.path.join(output_base_dir, rel_path)
    os.makedirs(output_dir, exist_ok=True)

    # Get list of video files in the subfolder
    video_files = [os.path.join(subfolder_path, f) for f in os.listdir(subfolder_path) 
                  if f.lower().endswith('.mp4')]

    if not video_files:
        logger.warning(f"No video files found in {subfolder_path}")
        return []

    # Initialize or update the processing log
    processed_files_log = initialize_processing_log(subfolder_path, video_files)

    # Filter out already processed files
    files_to_process = []
    for video_path in video_files:
        filename = os.path.basename(video_path)
        if filename in processed_files_log and processed_files_log[filename].get('processed_datetime'):
            logger.info(f"Skipping already processed file: {filename}")
        else:
            files_to_process.append(video_path)

    if not files_to_process:
        logger.info(f"All files in {subfolder_path} have already been processed")
        return []

    # Update video_files to only include files that need processing
    video_files = files_to_process

    # Sort video files by name for consistent processing
    video_files.sort()
    
    # Determine optimal number of workers if not specified
    if max_workers is None:
        import multiprocessing
        # Use half of available CPUs to avoid overloading the system
        # but at least 1 and at most 4 (to avoid excessive memory usage)
        cpu_count = multiprocessing.cpu_count()
        
        # If CUDA is available, we can use more workers since GPU will handle the heavy lifting
        if torch.cuda.is_available():
            max_workers = max(2, min(6, cpu_count // 2))
        else:
            max_workers = max(1, min(4, cpu_count // 2))
    
    # Print detailed information about found videos
    logger.info(f"Found {len(video_files)} video files in {subfolder_path}")
    logger.info(f"Will process with {max_workers} parallel workers")
    
    # Default colors for UI elements
    ui_colors = {
        'bg_primary': '#f8f9fa',
        'bg_secondary': '#e9ecef',
        'bg_success': '#d1e7dd',
        'bg_info': '#cff4fc',
        'bg_warning': '#fff3cd',
        'bg_danger': '#f8d7da',
        'text_primary': '#212529',
        'text_secondary': '#6c757d',
        'text_info': '#055160',
        'text_success': '#0f5132',
        'text_warning': '#664d03',
        'text_danger': '#842029',
    }
    
    # Function to update progress for a specific video
    def update_video_progress(video_index, progress, detection_count, segment_count, progress_widget=None):
        # Update the progress widget if available
        if progress_widget:
            progress_widget.value = progress
        
        # Check if we're in a Jupyter environment
        try:
            from IPython.display import display, HTML
            in_jupyter = True
        except (ImportError, NameError):
            in_jupyter = False
        
        if in_jupyter:
            # Update detections and motion cells
            display(HTML(f"<script>"
                        f"var detectionsCell = document.getElementById('detections-{video_index}');"
                        f"if (detectionsCell) detectionsCell.innerHTML = '{detection_count}';"
                        f"var motionCell = document.getElementById('motion-{video_index}');"
                        f"if (motionCell) motionCell.innerHTML = '{segment_count}';"
                        f"</script>"))
    
    # Process a single video
    def process_single_video(args):
        i, video_path = args
        video_name = os.path.basename(video_path)
        
        # Check if in Jupyter environment
        try:
            from IPython.display import display, HTML
            in_jupyter = True
        except (ImportError, NameError):
            in_jupyter = False
        
        # Update status in the table
        if in_jupyter:
            display(HTML(f"<script>"
                         f"var statusCell = document.getElementById('status-{i}');"
                         f"if (statusCell) statusCell.innerHTML = 'Processing';"
                         f"var row = document.getElementById('video-row-{i}');"
                         f"if (row) {{"
                         f"  row.style.backgroundColor = '{ui_colors['bg_info']}';"
                         f"  // Update text color for all cells in the row"
                         f"  var cells = row.getElementsByTagName('td');"
                         f"  for (var j = 0; j < cells.length; j++) {{"
                         f"    cells[j].style.color = '#ffffff';"
                         f"  }}"
                         f"}}"
                         f"</script>"))
        
        try:
            # Create output directory for this video
            video_output_dir = os.path.join(output_base_dir, os.path.basename(subfolder_path))
            os.makedirs(video_output_dir, exist_ok=True)
            
            # Get the progress widget for this video if available
            progress_widget = None
            if video_progress_widgets and video_name in video_progress_widgets:
                progress_widget = video_progress_widgets[video_name]
            
            # Process the video with progress tracking
            clips = process_video(
                video_path, 
                video_output_dir,
                model,
                progress_callback=lambda progress, detections, segments: update_video_progress(i, progress, detections, segments, progress_widget),
                dark_mode=dark_mode,
                **config
            )
            
            # Get the total number of objects detected and segments created
            total_objects = 0
            total_motion = 0
            segment_filenames = []
            
            # Extract segment filenames and counts
            for clip_info in clips:
                if isinstance(clip_info, dict):
                    if 'objects' in clip_info:
                        total_objects += clip_info.get('objects', 0)
                    if 'motion' in clip_info:
                        total_motion += clip_info.get('motion', 0)
                    if 'filename' in clip_info:
                        segment_filenames.append(os.path.basename(clip_info['filename']))
            
            # Update the processing log with the results
            update_processing_log(subfolder_path, video_name, {
                'processed_datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'objects': str(total_objects),
                'motion': str(total_motion),
                'segment_files': segment_filenames
            })
            
            # Update status to complete
            if in_jupyter:
                display(HTML(f"<script>"
                              f"var statusCell = document.getElementById('status-{i}');"
                              f"if (statusCell) statusCell.innerHTML = 'Complete';"
                              f"var row = document.getElementById('video-row-{i}');"
                              f"if (row) {{"
                              f"  row.style.backgroundColor = '{ui_colors['bg_success']}';"
                              f"  // Update text color for all cells in the row"
                              f"  var cells = row.getElementsByTagName('td');"
                              f"  for (var j = 0; j < cells.length; j++) {{"
                              f"    cells[j].style.color = '#ffffff';"
                              f"  }}"
                              f"}}"
                              f"</script>"))
            
            return i, video_name, clips, None
        except Exception as e:
            # Update status to error
            if in_jupyter:
                display(HTML(f"<script>"
                              f"var statusCell = document.getElementById('status-{i}');"
                              f"if (statusCell) statusCell.innerHTML = 'Error';"
                              f"var row = document.getElementById('video-row-{i}');"
                              f"if (row) {{"
                              f"  row.style.backgroundColor = '{ui_colors['bg_danger']}';"
                              f"  // Update text color for all cells in the row"
                              f"  var cells = row.getElementsByTagName('td');"
                              f"  for (var j = 0; j < cells.length; j++) {{"
                              f"    cells[j].style.color = '#ffffff';"
                              f"  }}"
                              f"}}"
                              f"</script>"))
            
            # Record the error in the processing log with current timestamp
            # We don't mark it as fully processed so it can be retried later
            update_processing_log(subfolder_path, video_name, {
                'processed_datetime': '',  # Leave empty to allow retry
                'objects': '0',
                'motion': '0',
                'segment_files': []
            })
            
            logger.error(f"Error processing {video_name}: {str(e)}")
            return i, video_name, [], str(e)
    
    # Track completed videos for progress updates
    completed_count = 0
    active_videos = []
    
    # For ETA calculation
    start_time = time.time()
    video_processing_times = {}
    
    # For memory monitoring
    last_memory_check = time.time()
    memory_check_interval = 5  # seconds
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all video processing tasks
        future_to_video = {}
        
        # Only submit tasks if not cancelled
        if not processing_cancelled:
            for i, video_path in enumerate(video_files):
                future = executor.submit(process_single_video, (i, video_path))
                future_to_video[future] = (i, os.path.basename(video_path))
        
        # Process results as they complete
        all_clips = []
        errors = []
        
        for future in concurrent.futures.as_completed(future_to_video):
            if processing_cancelled:
                # Cancel all pending futures
                for f in future_to_video:
                    f.cancel()
                break
            
            i, video_name, clips, error = future.result()
            completed_count += 1
            
            # Record processing time for this video for ETA calculation
            video_processing_times[video_name] = time.time() - start_time
            
            # Update active videos list
            if video_name in active_videos:
                active_videos.remove(video_name)
            
            # Add clips to the result list
            all_clips.extend(clips)
            
            # Record any errors
            if error:
                errors.append((video_name, error))
            
            # Check memory usage periodically and perform cleanup if needed
            current_time = time.time()
            if current_time - last_memory_check > memory_check_interval:
                last_memory_check = current_time
                if check_memory_usage(memory_limit_percent):
                    logger.warning(f"Memory usage exceeded {memory_limit_percent}%, performing garbage collection")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    
    # Print summary
    logger.info(f"Processed {completed_count} videos, extracted {len(all_clips)} segments")
    if errors:
        logger.warning(f"Encountered {len(errors)} errors:")
        for video_name, error in errors:
            logger.warning(f"  {video_name}: {error}")
    
    return all_clips
