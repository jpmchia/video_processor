#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Core video processing functionality for individual videos
"""

import cv2
import torch
import numpy as np
import time
import os
import logging
from datetime import timedelta
from moviepy.editor import VideoFileClip, concatenate_videoclips

from video_processor.utils.memory import check_memory_usage


def process_video(video_path, output_dir, model, 
                  confidence=0.45,
                  buffer_seconds=5,
                  min_object_area_ratio=0.001,
                  target_classes=None,
                  roi_coords=None,
                  motion_threshold=0.01,
                  skip_frames=10,
                  resize_factor=0.5,
                  adaptive_skip=True,
                  debug=False,
                  progress_callback=None,
                  dark_mode=False):
    """
    Process a video file to detect objects and extract segments with motion
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted clips
        model: YOLO model to use for detection
        confidence: Confidence threshold for detections
        buffer_seconds: Number of seconds to include before and after detection
        min_object_area_ratio: Minimum object area as ratio of frame area
        target_classes: List of class IDs to detect, None for all
        roi_coords: Region of interest coordinates [x1,y1,x2,y2], None for full frame
        motion_threshold: Threshold for motion detection
        skip_frames: Number of frames to skip between detections
        resize_factor: Factor to resize frames by before processing
        adaptive_skip: Whether to adaptively adjust skip_frames based on video FPS
        debug: Whether to enable debug output
        progress_callback: Callback function for progress updates
        dark_mode: Whether to use dark mode for UI elements
        
    Returns:
        List of dictionaries with clip information
    """
    logger = logging.getLogger("video_processor")
    logger.info(f"Processing video: {os.path.basename(video_path)}")
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video file: {video_path}")
        return []
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    logger.info(f"Video properties: {frame_width}x{frame_height}, {fps:.2f}fps, {duration:.2f}s")
    
    # Adjust skip_frames based on FPS if adaptive_skip is enabled
    if adaptive_skip:
        # Higher FPS videos can skip more frames
        if fps > 30:
            skip_frames = int(skip_frames * (fps / 30))
        elif fps < 15:
            skip_frames = max(1, int(skip_frames * (fps / 30)))
        
        logger.info(f"Adaptive frame skipping: {skip_frames} frames")
    
    # Calculate buffer frames
    buffer_frames = int(buffer_seconds * fps)
    
    # Initialize variables for motion detection
    prev_frame = None
    motion_history = []
    
    # Initialize variables for object detection
    detection_frames = []
    current_frame_idx = 0
    
    # Initialize segment tracking
    segments = []
    current_segment = None
    
    # Calculate minimum object area in pixels
    min_object_area = min_object_area_ratio * frame_width * frame_height
    
    # For progress tracking
    detection_count = 0
    segment_count = 0
    
    # Process frames
    while cap.isOpened():
        # Check if we need to skip frames
        if current_frame_idx % skip_frames != 0:
            # Skip this frame
            cap.grab()
            current_frame_idx += 1
            continue
        
        # Read the next frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Update progress
        progress = min(100, int((current_frame_idx / total_frames) * 100))
        if progress_callback:
            progress_callback(progress, detection_count, segment_count)
        
        # Resize frame for faster processing
        if resize_factor != 1.0:
            frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
        
        # Apply ROI if specified
        if roi_coords:
            x1, y1, x2, y2 = roi_coords
            frame_roi = frame[y1:y2, x1:x2]
        else:
            frame_roi = frame
        
        # Convert frame to grayscale for motion detection
        gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
        
        # Detect motion
        motion_detected = False
        if prev_frame is not None:
            # Calculate absolute difference between current and previous frame
            frame_diff = cv2.absdiff(gray, prev_frame)
            
            # Apply threshold to difference
            _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
            
            # Calculate motion score (ratio of changed pixels)
            motion_score = np.sum(thresh) / (thresh.shape[0] * thresh.shape[1] * 255)
            
            # Add to motion history
            motion_history.append(motion_score)
            if len(motion_history) > 10:
                motion_history.pop(0)
            
            # Check if motion exceeds threshold
            avg_motion = sum(motion_history) / len(motion_history)
            motion_detected = avg_motion > motion_threshold
        
        # Update previous frame
        prev_frame = gray
        
        # Detect objects using YOLO
        objects_detected = False
        if current_frame_idx % (skip_frames * 3) == 0:  # Run object detection less frequently
            # Use mixed precision for faster inference if available
            with torch.cuda.amp.autocast() if torch.cuda.is_available() else torch.no_grad():
                results = model(frame)
            
            # Process detections
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box information
                    cls = int(box.cls.item())
                    conf = box.conf.item()
                    
                    # Check if class is in target classes
                    if target_classes is not None and cls not in target_classes:
                        continue
                    
                    # Check if confidence exceeds threshold
                    if conf < confidence:
                        continue
                    
                    # Get bounding box
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # Calculate object area
                    object_area = (x2 - x1) * (y2 - y1)
                    
                    # Check if object area exceeds minimum
                    if object_area < min_object_area:
                        continue
                    
                    # Object detected
                    objects_detected = True
                    detection_count += 1
                    break
                
                if objects_detected:
                    break
        
        # Update segment tracking
        if motion_detected or objects_detected:
            if current_segment is None:
                # Start a new segment
                segment_start = max(0, current_frame_idx - buffer_frames)
                current_segment = {
                    'start': segment_start,
                    'end': current_frame_idx,
                    'motion': motion_detected,
                    'objects': objects_detected
                }
            else:
                # Extend current segment
                current_segment['end'] = current_frame_idx
                current_segment['motion'] = current_segment['motion'] or motion_detected
                current_segment['objects'] = current_segment['objects'] or objects_detected
        elif current_segment is not None:
            # Check if segment has ended
            if current_frame_idx - current_segment['end'] > buffer_frames:
                # Add buffer to end
                current_segment['end'] += buffer_frames
                
                # Add segment to list
                segments.append(current_segment)
                segment_count += 1
                
                # Reset current segment
                current_segment = None
        
        # Increment frame index
        current_frame_idx += 1
        
        # Check memory usage and perform cleanup if needed
        if current_frame_idx % 100 == 0:
            if check_memory_usage(90):  # 90% threshold for cleanup during processing
                logger.warning("Memory usage high, performing garbage collection")
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    # Add final segment if there is one
    if current_segment is not None:
        # Add buffer to end
        current_segment['end'] = min(current_segment['end'] + buffer_frames, total_frames - 1)
        
        # Add segment to list
        segments.append(current_segment)
        segment_count += 1
    
    # Release video capture
    cap.release()
    
    # Merge overlapping segments
    segments.sort(key=lambda x: x['start'])
    merged_segments = []
    
    for segment in segments:
        if not merged_segments:
            merged_segments.append(segment)
        else:
            last_segment = merged_segments[-1]
            
            # Check if segments overlap
            if segment['start'] <= last_segment['end']:
                # Merge segments
                last_segment['end'] = max(last_segment['end'], segment['end'])
                last_segment['motion'] = last_segment['motion'] or segment['motion']
                last_segment['objects'] = last_segment['objects'] or segment['objects']
            else:
                # Add new segment
                merged_segments.append(segment)
    
    # Extract segments
    extracted_clips = []
    
    for i, segment in enumerate(merged_segments):
        try:
            # Calculate segment duration
            segment_duration = (segment['end'] - segment['start']) / fps
            
            # Skip very short segments
            if segment_duration < 1.0:
                continue
            
            # Calculate segment times
            start_time = segment['start'] / fps
            end_time = segment['end'] / fps
            
            # Format times as MM:SS
            start_time_str = str(timedelta(seconds=int(start_time)))
            end_time_str = str(timedelta(seconds=int(end_time)))
            
            # Create output filename
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            segment_name = f"{video_name}_seg{i+1:03d}_{start_time_str.replace(':', '')}_{end_time_str.replace(':', '')}"
            
            if segment['objects']:
                segment_name += "_obj"
            if segment['motion']:
                segment_name += "_mot"
            
            output_path = os.path.join(output_dir, f"{segment_name}.mp4")
            
            # Extract segment using moviepy
            with VideoFileClip(video_path) as video:
                # Extract the segment
                segment_clip = video.subclip(start_time, end_time)
                
                # Write the segment to file with optimized settings
                segment_clip.write_videofile(
                    output_path,
                    codec='libx264',
                    audio_codec='aac',
                    preset='ultrafast',
                    threads=2,
                    ffmpeg_params=['-tune', 'fastdecode'],
                    logger=None  # Suppress moviepy output
                )
            
            # Add to extracted clips
            extracted_clips.append({
                'filename': output_path,
                'start_time': start_time,
                'end_time': end_time,
                'duration': segment_duration,
                'objects': 1 if segment['objects'] else 0,
                'motion': 1 if segment['motion'] else 0
            })
            
            logger.info(f"Extracted segment: {os.path.basename(output_path)}")
        
        except Exception as e:
            logger.error(f"Error extracting segment {i+1}: {str(e)}")
    
    # Final progress update
    if progress_callback:
        progress_callback(100, detection_count, segment_count)
    
    logger.info(f"Completed processing {os.path.basename(video_path)}: {len(extracted_clips)} segments extracted")
    
    return extracted_clips
