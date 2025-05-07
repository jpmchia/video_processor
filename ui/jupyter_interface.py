#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Jupyter notebook interface for video processing
"""

import os
import sys
import time
import torch
import logging
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets

from video_processor.core.processor import process_subfolder
from video_processor.core.model_loader import load_model
from video_processor.utils.logger import setup_logger, JupyterHandler
from video_processor.utils.memory import check_memory_usage, cleanup_memory


def main_jupyter():
    """
    Main function for Jupyter notebook interface
    """
    # Set up dark mode colors
    colors = {
        'bg_primary': '#1e1e1e',
        'bg_secondary': '#252526',
        'bg_success': '#0d5323',
        'bg_info': '#063b59',
        'bg_warning': '#664d03',
        'bg_danger': '#842029',
        'text_primary': '#e0e0e0',
        'text_secondary': '#a0a0a0',
        'text_info': '#4fc3f7',
        'text_success': '#81c784',
        'text_warning': '#ffb74d',
        'text_danger': '#e57373',
        'border': '#424242'
    }
    
    # Global variables
    global processing_cancelled
    processing_cancelled = False
    
    # Set up dark mode styling
    def setup_dark_mode(dark_mode=False):
        if dark_mode:
            display(HTML("""
            <style>
                .jp-OutputArea-output, .jp-Cell-outputWrapper {
                    background-color: #1e1e1e !important;
                    color: #e0e0e0 !important;
                }
                
                /* Style for all jupyter widgets */
                .jupyter-widgets {
                    background-color: #1e1e1e !important;
                    color: #e0e0e0 !important;
                }
                
                /* Target HTML elements inside widgets but preserve their colors */
                .jupyter-widgets body,
                .jupyter-widgets div,
                .jupyter-widgets span,
                .jupyter-widgets p,
                .jupyter-widgets h1,
                .jupyter-widgets h2,
                .jupyter-widgets h3,
                .jupyter-widgets h4,
                .jupyter-widgets h5,
                .jupyter-widgets h6,
                .jupyter-widgets ul,
                .jupyter-widgets ol,
                .jupyter-widgets dl,
                .jupyter-widgets pre,
                .jupyter-widgets form,
                .jupyter-widgets table,
                .jupyter-widgets th,
                .jupyter-widgets td {
                    color: inherit;
                }
                
                /* Preserve colors for status indicators */
                .jupyter-widgets .text-success {
                    color: #81c784 !important;
                }
                
                .jupyter-widgets .text-info {
                    color: #4fc3f7 !important;
                }
                
                .jupyter-widgets .text-warning {
                    color: #ffb74d !important;
                }
                
                .jupyter-widgets .text-danger {
                    color: #e57373 !important;
                }
            </style>
            """))
    
    # Create widgets for the interface
    header = widgets.HTML(
        value="<h2>Video Processing with YOLO</h2>",
        layout=widgets.Layout(margin='10px 0')
    )
    
    # Base directory input
    base_dir_input = widgets.Text(
        value="/mnt/j/Bugs/LAIDG0025410XEDT",
        description="Base Directory:",
        layout=widgets.Layout(width='80%')
    )
    
    # Output directory input
    output_dir_input = widgets.Text(
        value="/mnt/j/Bugs/Processed",
        description="Output Directory:",
        layout=widgets.Layout(width='80%')
    )
    
    # Model selection
    model_dropdown = widgets.Dropdown(
        options=['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt'],
        value='yolo11n.pt',
        description='YOLO Model:',
        layout=widgets.Layout(width='50%')
    )
    
    # Confidence threshold
    confidence_slider = widgets.FloatSlider(
        value=0.35,
        min=0.1,
        max=0.9,
        step=0.05,
        description='Confidence:',
        layout=widgets.Layout(width='50%')
    )
    
    # Dark mode toggle
    dark_mode_toggle = widgets.Checkbox(
        value=True,
        description='Dark Mode',
        layout=widgets.Layout(width='20%')
    )
    
    # Subfolder dropdown (will be populated later)
    subfolder_dropdown = widgets.Dropdown(
        options=[],
        description='Subfolder:',
        layout=widgets.Layout(width='80%')
    )
    
    # Refresh button for subfolders
    refresh_button = widgets.Button(
        description='Refresh Subfolders',
        icon='refresh',
        button_style='info',
        layout=widgets.Layout(width='20%')
    )
    
    # Process button
    process_button = widgets.Button(
        description='Process Subfolder',
        icon='play',
        button_style='success',
        layout=widgets.Layout(width='20%')
    )
    
    # Cancel button
    cancel_button = widgets.Button(
        description='Cancel',
        icon='stop',
        button_style='danger',
        disabled=True,
        layout=widgets.Layout(width='20%')
    )
    
    # Output area
    output = widgets.Output()
    
    # Function to update subfolder dropdown
    def update_subfolder_dropdown(base_dir):
        if os.path.isdir(base_dir):
            subfolders = [f for f in os.listdir(base_dir) 
                         if os.path.isdir(os.path.join(base_dir, f))]
            subfolder_dropdown.options = subfolders
            if subfolders:
                subfolder_dropdown.value = subfolders[0]
        else:
            subfolder_dropdown.options = []
    
    # Function to handle base directory change
    def on_base_dir_change(change):
        update_subfolder_dropdown(change['new'])
    
    # Function to handle refresh button click
    def on_refresh_button_click(b):
        update_subfolder_dropdown(base_dir_input.value)
    
    # Function to handle dark mode toggle
    def on_dark_mode_toggle(change):
        setup_dark_mode(change['new'])
    
    # Function to handle cancel button click
    def on_cancel_button_click(b):
        global processing_cancelled
        processing_cancelled = True
        cancel_button.description = 'Cancelling...'
        cancel_button.icon = 'hourglass'
        cancel_button.button_style = 'warning'
    
    # Function to handle process button click
    def on_process_button_click(b):
        global processing_cancelled
        processing_cancelled = False  # Reset cancellation flag
        
        # Update button states
        process_button.disabled = True
        cancel_button.disabled = False
        cancel_button.description = 'Cancel Processing'
        cancel_button.icon = 'stop'
        cancel_button.button_style = 'danger'
        
        with output:
            clear_output()
            subfolder_path = os.path.join(base_dir_input.value, subfolder_dropdown.value)
            output_base_dir = output_dir_input.value
            
            # Apply dark mode if enabled
            dark_mode = dark_mode_toggle.value
            setup_dark_mode(dark_mode)
            
            # Create a log output widget (needs to be created before it's used in the logger)
            log_widget = widgets.HTML(
                value="<div style='height: 200px; overflow-y: auto; font-family: monospace;'><div>Log messages will appear here...</div></div>",
                layout=widgets.Layout(background_color='#1e1e1e' if dark_mode else 'white')
            )
            
            # Configure logging to avoid duplicates and redirect to widgets
            logger = setup_logger("video_processor", "video_processing.log")
            
            # Set up logging handler
            jupyter_handler = JupyterHandler()
            jupyter_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(jupyter_handler)
            
            # Set the log widget
            jupyter_handler.set_widget(log_widget)
            
            logger.info("======= Subfolder Processing With YOLO =======")
            
            # Configuration settings adjusted to reduce NMS timeouts
            config = {
                "confidence": confidence_slider.value,
                "buffer_seconds": 5,
                "min_object_area_ratio": 0.002,  # Slightly larger min object size (0.2% of frame)
                "target_classes": [0, 1, 2, 3, 5, 7],  # Focus on people and vehicles
                "roi_coords": None,        # No ROI restriction
                "motion_threshold": 0.015, # Moderate motion threshold
                "skip_frames": 15,        # Base frame skip rate (will be adjusted adaptively)
                "resize_factor": 0.5,      # Resize frames to 50% for faster processing
                "adaptive_skip": True,     # Dynamically adjust skip_frames based on video FPS
                "debug": True              # Enable detailed logging
            }
            
            # Load YOLOv11 model with CUDA and optimize for faster NMS
            model_path = model_dropdown.value
            
            logger.info(f"Loading model {model_path} with optimized settings...")
            
            # Add a progress indicator for model loading using widgets
            try:
                # Create a progress widget
                progress = widgets.FloatProgress(
                    value=0,
                    min=0,
                    max=100,
                    description='Loading model:',
                    bar_style='info',
                    style={'description_width': 'initial'}
                )
                display(progress)
                
                # Update progress
                progress.value = 10
                
                # Load the model
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = load_model(model_path, device)
                
                # Update progress
                progress.value = 100
                progress.bar_style = 'success'
                progress.description = 'Model loaded!'
                
                logger.info(f"Model loaded successfully on {device}")
            except Exception as e:
                logger.error(f"Failed to load YOLO model: {str(e)}")
                process_button.disabled = False
                cancel_button.disabled = True
                return
            
            # Get video files in the subfolder
            video_files = [os.path.join(subfolder_path, f) for f in os.listdir(subfolder_path) 
                          if f.lower().endswith('.mp4')]
            
            if not video_files:
                logger.warning(f"No video files found in {subfolder_path}")
                process_button.disabled = False
                cancel_button.disabled = True
                return
            
            # Sort video files by name
            video_files.sort()
            
            # Calculate total duration of all videos
            total_duration = 0
            video_info_rows = []
            
            for i, video_path in enumerate(video_files):
                try:
                    # Get video info
                    cap = cv2.VideoCapture(video_path)
                    if cap.isOpened():
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        duration = total_frames / fps if fps > 0 else 0
                        total_duration += duration
                        
                        # Format duration as MM:SS
                        minutes, seconds = divmod(duration, 60)
                        duration_str = f"{int(minutes):02d}:{int(seconds):02d}"
                        
                        # Create a row with additional status columns
                        video_info_rows.append(f"<tr id='video-row-{len(video_info_rows)}'>"
                                               f"<td>{os.path.basename(video_path)}</td>"
                                               f"<td>{width}x{height}</td>"
                                               f"<td>{fps:.2f}</td>"
                                               f"<td>{duration_str}</td>"
                                               f"<td id='status-{len(video_info_rows)}'>Pending</td>"
                                               f"<td id='detections-{len(video_info_rows)}'>0</td>"
                                               f"<td id='motion-{len(video_info_rows)}'>0</td>"
                                               f"<td id='progress-cell-{len(video_info_rows)}'></td>"
                                               f"</tr>")
                        cap.release()
                    else:
                        video_info_rows.append(f"<tr><td>{os.path.basename(video_path)}</td><td colspan='7'>Could not read video properties</td></tr>")
                except Exception as e:
                    video_info_rows.append(f"<tr><td>{os.path.basename(video_path)}</td><td colspan='7'>Error: {str(e)}</td></tr>")
            
            # Format total duration
            total_minutes, total_seconds = divmod(total_duration, 60)
            total_hours, total_minutes = divmod(total_minutes, 60)
            total_duration_str = f"{int(total_hours):02d}:{int(total_minutes):02d}:{int(total_seconds):02d}"
            
            # Get colors from the main function if available
            bg_secondary = colors['bg_secondary'] if dark_mode else '#f8f9fa'
            text_primary = colors['text_primary'] if dark_mode else '#333333'
            border_color = colors['border'] if dark_mode else '#dee2e6'
            header_bg = colors['bg_primary'] if dark_mode else '#e9ecef'
            
            # Display subfolder info with video details
            subfolder_info_html = f"""
            <div style='background-color: {bg_secondary}; color: {text_primary}; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
              <h4 style='color: {text_primary};'>Processing Subfolder</h4>
              <ul style='color: {text_primary};'>
                <li>Subfolder: {os.path.basename(subfolder_path)}</li>
                <li>Path: {subfolder_path}</li>
                <li>Output: {os.path.join(output_base_dir, os.path.basename(subfolder_path))}</li>
                <li>Videos: {len(video_files)} files</li>
                <li>Total Duration: {total_duration_str}</li>
              </ul>
              
              <h5 style='color: {text_primary};'>Video Files:</h5>
              <table style='width:100%; border-collapse: collapse; color: {text_primary};'>
                <thead>
                  <tr style='background-color: {header_bg};'>
                    <th style='padding: 8px; text-align: left; border: 1px solid {border_color}; color: {text_primary};'>Filename</th>
                    <th style='padding: 8px; text-align: left; border: 1px solid {border_color}; color: {text_primary};'>Resolution</th>
                    <th style='padding: 8px; text-align: left; border: 1px solid {border_color}; color: {text_primary};'>FPS</th>
                    <th style='padding: 8px; text-align: left; border: 1px solid {border_color}; color: {text_primary};'>Duration</th>
                    <th style='padding: 8px; text-align: left; border: 1px solid {border_color}; color: {text_primary};'>Status</th>
                    <th style='padding: 8px; text-align: left; border: 1px solid {border_color}; color: {text_primary};'>Detections</th>
                    <th style='padding: 8px; text-align: left; border: 1px solid {border_color}; color: {text_primary};'>Motion</th>
                    <th style='padding: 8px; text-align: left; border: 1px solid {border_color}; color: {text_primary};'>% complete</th>
                  </tr>
                </thead>
                <tbody>
                  {''.join(video_info_rows)}
                </tbody>
              </table>
            </div>
            """
            display(HTML(subfolder_info_html))
            
            # Create a dictionary to store progress widgets for each video
            video_progress_widgets = {}
            video_status_widgets = {}
            
            # Define worker and memory limit parameters
            max_workers = None  # Auto-detect optimal number of workers
            memory_limit_percent = 85  # Trigger memory cleanup at 85% usage
            
            # Create progress widgets for each video and add them to the table
            for i, video_path in enumerate(video_files):
                # Create a progress widget for this video
                video_name = os.path.basename(video_path)
                progress_widget = widgets.IntProgress(
                    value=0,
                    min=0,
                    max=100,
                    description='',
                    bar_style='info',
                    style={'description_width': '0px', 'bar_color': '#3498db'}
                )
                
                # Store the widget in our dictionary
                video_progress_widgets[video_name] = progress_widget
                
                # Add the widget to the table
                display(HTML(f"<script>"
                             f"document.getElementById('progress-cell-{i}').innerHTML = '';"
                             f"</script>"))
                
                # Display the widget in the cell
                with widgets.Output() as out:
                    display(progress_widget)
                display(HTML(f"<script>"
                             f"var cell = document.getElementById('progress-cell-{i}');"
                             f"if (cell) cell.appendChild(document.getElementsByClassName('jupyter-widgets')[document.getElementsByClassName('jupyter-widgets').length-1]);"
                             f"</script>"))
            
            # Process the subfolder with the progress widgets
            clips = process_subfolder(
                subfolder_path,
                output_base_dir,
                model,
                config,
                max_workers=max_workers,
                memory_limit_percent=memory_limit_percent,
                video_progress_widgets=video_progress_widgets,
                dark_mode=dark_mode
            )
            
            # Get success background color if available
            success_bg = colors['bg_success'] if dark_mode else '#dff0d8'
            success_text = colors['text_primary'] if dark_mode else '#333333'
            
            # Show completion message with a widget
            if processing_cancelled:
                completion_widget = widgets.HTML(
                    value=f"<div style='background-color: {colors['bg_warning'] if dark_mode else '#fff3cd'}; color: {colors['text_warning'] if dark_mode else '#664d03'}; padding: 10px; border-radius: 5px; margin-top: 10px;'>"
                          f"<h4>Processing Cancelled</h4>"
                          f"<p>Processing was cancelled. Some videos may not have been processed.</p>"
                          f"</div>"
                )
            else:
                completion_widget = widgets.HTML(
                    value=f"<div style='background-color: {success_bg}; color: {success_text}; padding: 10px; border-radius: 5px; margin-top: 10px;'>"
                          f"<h4>Processing Complete</h4>"
                          f"<p>Processed {len(video_files)} videos and extracted {len(clips)} segments.</p>"
                          f"<p>Output directory: {os.path.join(output_base_dir, os.path.basename(subfolder_path))}</p>"
                          f"</div>"
                )
            
            display(completion_widget)
            display(log_widget)
            
            # Reset button states
            process_button.disabled = False
            cancel_button.disabled = True
    
    # Connect event handlers
    base_dir_input.observe(on_base_dir_change, names='value')
    refresh_button.on_click(on_refresh_button_click)
    dark_mode_toggle.observe(on_dark_mode_toggle, names='value')
    process_button.on_click(on_process_button_click)
    cancel_button.on_click(on_cancel_button_click)
    
    # Initial update of subfolder dropdown
    update_subfolder_dropdown(base_dir_input.value)
    
    # Apply initial dark mode if enabled
    setup_dark_mode(dark_mode_toggle.value)
    
    # Display the interface
    display(header)
    display(widgets.HBox([base_dir_input, refresh_button]))
    display(output_dir_input)
    display(widgets.HBox([model_dropdown, confidence_slider, dark_mode_toggle]))
    display(widgets.HBox([subfolder_dropdown]))
    display(widgets.HBox([process_button, cancel_button]))
    display(output)
