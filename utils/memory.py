#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Memory management utilities
"""

import psutil
import gc
import torch
import logging


def check_memory_usage(threshold_percent=85):
    """
    Check if memory usage exceeds the threshold
    
    Args:
        threshold_percent: Memory usage threshold percentage
        
    Returns:
        bool: True if memory usage exceeds threshold, False otherwise
    """
    # Get memory usage
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    
    # Check if GPU is available and get GPU memory usage
    gpu_memory_percent = 0
    if torch.cuda.is_available():
        try:
            # Get GPU memory usage
            gpu_memory_used = torch.cuda.memory_allocated()
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_percent = (gpu_memory_used / gpu_memory_total) * 100
        except Exception:
            # If we can't get GPU memory usage, assume it's 0
            pass
    
    # Use the higher of CPU and GPU memory usage
    memory_percent = max(memory_percent, gpu_memory_percent)
    
    # Return True if memory usage exceeds threshold
    return memory_percent > threshold_percent


def cleanup_memory():
    """
    Perform memory cleanup
    
    Returns:
        tuple: (CPU memory before, CPU memory after, GPU memory before, GPU memory after)
    """
    logger = logging.getLogger("video_processor")
    
    # Get memory usage before cleanup
    memory_before = psutil.virtual_memory().percent
    gpu_memory_before = 0
    
    if torch.cuda.is_available():
        try:
            gpu_memory_before = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory * 100
        except Exception:
            pass
    
    # Perform garbage collection
    gc.collect()
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Get memory usage after cleanup
    memory_after = psutil.virtual_memory().percent
    gpu_memory_after = 0
    
    if torch.cuda.is_available():
        try:
            gpu_memory_after = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory * 100
        except Exception:
            pass
    
    # Log memory usage
    logger.info(f"Memory cleanup: CPU {memory_before:.1f}% -> {memory_after:.1f}%, "
               f"GPU {gpu_memory_before:.1f}% -> {gpu_memory_after:.1f}%")
    
    return (memory_before, memory_after, gpu_memory_before, gpu_memory_after)
