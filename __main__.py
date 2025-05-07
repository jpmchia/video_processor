#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Video Processor - Main entry point
"""

import argparse
import logging
import os
import sys

from video_processor.core.processor import process_subfolder
from video_processor.core.model_loader import load_model
from video_processor.ui.jupyter_interface import main_jupyter
from video_processor.utils.logger import setup_logger


def main_cli():
    """Command line interface for the script"""
    parser = argparse.ArgumentParser(description="Process videos in a subfolder using YOLO")
    parser.add_argument("--base_dir", type=str, default="/data/Input", 
                        help="Base directory containing subfolders with videos")
    parser.add_argument("--output_dir", type=str, default="/data/Output",
                        help="Output directory for processed video clips")
    parser.add_argument("--subfolder", type=str, default=None,
                        help="Specific subfolder to process (if not provided, will list available subfolders)")
    parser.add_argument("--model", type=str, default="yolo11n.pt",
                        help="Path to YOLO model")
    parser.add_argument("--confidence", type=float, default=0.35,
                        help="Confidence threshold for detections")
    parser.add_argument("--max_workers", type=int, default=None,
                        help="Maximum number of worker threads (default: auto)")
    parser.add_argument("--dark_mode", action="store_true",
                        help="Enable dark mode for UI elements")
    
    args = parser.parse_args()
    
    # Setup logger
    logger = setup_logger("video_processor", "video_processing.log")
    
    # If no subfolder is specified, list available subfolders
    if not args.subfolder:
        subfolders = [f for f in os.listdir(args.base_dir) 
                     if os.path.isdir(os.path.join(args.base_dir, f))]
        if not subfolders:
            logger.error(f"No subfolders found in {args.base_dir}")
            return
        
        logger.info("Available subfolders:")
        for i, folder in enumerate(subfolders):
            logger.info(f"{i+1}. {folder}")
        
        logger.info("Please specify a subfolder using the --subfolder argument")
        return
    
    # Construct full path to subfolder
    subfolder_path = os.path.join(args.base_dir, args.subfolder)
    if not os.path.isdir(subfolder_path):
        logger.error(f"Subfolder not found: {subfolder_path}")
        return
    
    # Load model
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_model(args.model, device)
        logger.info(f"Model loaded successfully on {device}")
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {str(e)}")
        return
    
    # Configuration
    config = {
        "confidence": args.confidence,
        "buffer_seconds": 5,
        "min_object_area_ratio": 0.002,
        "target_classes": [0, 1, 2, 3, 5, 7],
        "roi_coords": None,
        "motion_threshold": 0.015,
        "skip_frames": 15,
        "resize_factor": 0.5,
        "adaptive_skip": True,
        "debug": True
    }
    
    # Process the subfolder
    logger.info(f"Processing subfolder: {args.subfolder}")
    clips = process_subfolder(
        subfolder_path,
        args.output_dir,
        model,
        config,
        max_workers=args.max_workers,
        dark_mode=args.dark_mode
    )
    
    logger.info(f"Processing complete! Extracted {len(clips)} video segments.")


if __name__ == "__main__":
    # Check if we're in a Jupyter notebook
    try:
        import IPython
        if IPython.get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
            # We're in a Jupyter notebook, use the Jupyter interface
            main_jupyter()
        else:
            # We're in a regular Python environment, use the CLI
            main_cli()
    except (ImportError, AttributeError):
        # We're in a regular Python environment, use the CLI
        main_cli()
