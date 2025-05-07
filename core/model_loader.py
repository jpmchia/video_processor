#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model loading functionality
"""

import torch
import logging
from ultralytics import YOLO


def load_model(model_path, device=None):
    """
    Load a YOLO model for object detection
    
    Args:
        model_path: Path to the YOLO model file
        device: Device to load the model on ('cuda' or 'cpu')
        
    Returns:
        Loaded YOLO model
    """
    logger = logging.getLogger("video_processor")
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Loading YOLO model from {model_path} on {device}")
    
    try:
        # Load the model
        model = YOLO(model_path)
        
        # Move model to device
        model.to(device)
        
        # Set model parameters for faster inference
        if device == "cuda":
            # Use half precision for faster inference
            model.model.half()
        
        logger.info(f"Model loaded successfully: {model_path}")
        return model
    
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise
