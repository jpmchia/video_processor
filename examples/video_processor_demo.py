#!/usr/bin/env python3
"""
Video Processor with YOLO Demo

This script demonstrates how to use the Video Processor with YOLO models for object detection in videos.
It can be run directly or used as a reference for your own scripts.

Features:
- Automatic model downloading
- Object detection in videos
- Parallel video processing
- Progress tracking
- Memory management
"""

import os
import sys
import time
from pathlib import Path

# Add the project root to the path if needed
project_root = Path(__file__).absolute().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("=" * 50)
print("Video Processor with YOLO Demo")
print("=" * 50)

# 1. Setup
print("\n## 1. Setup")
print("Importing modules and setting up environment...")

# Check if we're running in a Jupyter notebook
try:
    import IPython
    is_jupyter = IPython.get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
    print(f"Running in Jupyter: {is_jupyter}")
except (ImportError, AttributeError):
    is_jupyter = False
    print("Not running in Jupyter")

# 2. Model Management
print("\n## 2. Model Management")
print("Exploring available models and downloading one for our demo...")

# Import our model manager
from models import get_model, model_manager

# List available models
print("\nAvailable pre-trained models:")
for model_name in model_manager.list_available_models():
    print(f"  - {model_name}")

# Show where models will be stored
print(f"\nModels will be stored in: {model_manager.models_dir}")

# Download and load a model
# We'll use yolov8n.pt which is small and fast for demonstration purposes
model_name = "yolov8n.pt"
print(f"\nDownloading and loading model: {model_name}")
print("(This will only download if the model isn't already cached)")

start_time = time.time()
model = get_model(model_name)
elapsed = time.time() - start_time

print(f"Model loaded in {elapsed:.2f} seconds")
print(f"Model type: {type(model).__name__}")
print(f"Model task: {model.task}")

# 3. Video Processing Configuration
print("\n## 3. Video Processing Configuration")
print("Setting up the configuration parameters...")

# Configuration for video processing
config = {
    "confidence": 0.35,        # Confidence threshold for detections
    "buffer_seconds": 5,       # Buffer seconds before and after detection
    "min_object_area_ratio": 0.002,  # Minimum object area relative to frame
    "target_classes": [0, 1, 2, 3, 5, 7],  # Person, bicycle, car, motorcycle, bus, truck
    "roi_coords": None,        # Region of interest (None for full frame)
    "motion_threshold": 0.015, # Motion detection threshold
    "skip_frames": 15,         # Number of frames to skip between detections
    "resize_factor": 0.5,      # Resize factor for processing (smaller = faster)
    "adaptive_skip": True,     # Adaptively adjust frame skipping
    "debug": True              # Enable debug information
}

print("\nVideo processing configuration:")
for key, value in config.items():
    print(f"  - {key}: {value}")

# 4. Using the Jupyter Interface
print("\n## 4. Using the Jupyter Interface")
print("If you're running in a Jupyter notebook, you can use the interactive interface.")

if is_jupyter:
    print("Jupyter detected, you can uncomment the following line to launch the interface:")
    print("# from video_processor.ui.jupyter_interface import main_jupyter")
    print("# main_jupyter()")
else:
    print("Not running in Jupyter, skipping interactive interface.")

# 5. Manual Video Processing Example
print("\n## 5. Manual Video Processing Example")
print("Example of how to process videos programmatically...")

# Define input and output directories
# Replace these with your actual video directories
input_dir = "/path/to/your/videos"
output_dir = "/path/to/output"

# Check if the input directory exists
if not os.path.exists(input_dir):
    print(f"\nInput directory {input_dir} does not exist.")
    print("Please update the path to point to your video directory.")
    print("Example code for processing videos:")
    print("""
    from video_processor.core.processor import process_subfolder
    
    # Process the videos
    clips = process_subfolder(
        input_dir,
        output_dir,
        model,
        config,
        max_workers=2,  # Adjust based on your system
        memory_limit_percent=85
    )
    
    print(f"Processed videos and extracted {len(clips)} segments")
    """)
else:
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nFound input directory, ready to process videos.")
    print("Uncomment the following code to process videos:")
    print("""
    from video_processor.core.processor import process_subfolder
    
    # Process the videos
    clips = process_subfolder(
        input_dir,
        output_dir,
        model,
        config,
        max_workers=2,  # Adjust based on your system
        memory_limit_percent=85
    )
    
    print(f"Processed videos and extracted {len(clips)} segments")
    """)

# 6. Visualizing Results
print("\n## 6. Visualizing Results")
print("Example of how to visualize detection results...")

print("""
# Example code for visualizing results on an image:
image_path = "/path/to/your/image.jpg"

if os.path.exists(image_path):
    # Run inference on the image
    results = model(image_path)
    
    # Print detection information
    print(f"Detected {len(results[0].boxes)} objects")
    for i, box in enumerate(results[0].boxes):
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        name = results[0].names[cls]
        print(f"  {i+1}. {name} (confidence: {conf:.2f})")
    
    # If in Jupyter, display the image
    if is_jupyter:
        from IPython.display import display
        from PIL import Image
        import numpy as np
        
        # Get the image with annotations
        annotated_img = results[0].plot()
        
        # Convert to PIL Image and display
        display(Image.fromarray(annotated_img))
""")

# 7. Conclusion
print("\n## 7. Conclusion")
print("In this demo, we've demonstrated how to:")
print("1. Set up the Video Processor environment")
print("2. Download and load YOLO models")
print("3. Configure video processing parameters")
print("4. Use the Jupyter interface for interactive processing")
print("5. Process videos programmatically")
print("6. Visualize detection results")
print("\nThe Video Processor is a powerful tool for automatically detecting objects in videos and extracting relevant segments.")
