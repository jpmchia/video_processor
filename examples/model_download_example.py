#!/usr/bin/env python3
"""
Example script demonstrating the automatic model downloading functionality.
This script shows how to use the ModelManager to download and use YOLO models.
"""

import os
import sys
import time
from pathlib import Path

# Add the project root to the path so we can import the package
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import directly from the models module
from models import get_model, model_manager


def main():
    """Demonstrate model downloading and basic usage."""
    print("Video Processor - Model Download Example")
    print("-" * 50)
    
    # Display the models directory
    print(f"Models will be stored in: {model_manager.models_dir}")
    print()
    
    # List available models
    print("Available pre-trained models:")
    for model_name in model_manager.list_available_models():
        print(f"  - {model_name}")
    print()
    
    # Download and use a model
    model_name = "yolov8n.pt"  # Small and fast model for demonstration
    print(f"Downloading and loading model: {model_name}")
    print("(This will only download if the model isn't already cached)")
    
    start_time = time.time()
    model = get_model(model_name)
    elapsed = time.time() - start_time
    
    print(f"Model loaded in {elapsed:.2f} seconds")
    print(f"Model type: {type(model).__name__}")
    print(f"Model task: {model.task}")
    print()
    
    # Try a second model (YOLO11 nano)
    model_name = "yolo11n.pt"
    print(f"Downloading and loading another model: {model_name}")
    
    start_time = time.time()
    model = get_model(model_name)
    elapsed = time.time() - start_time
    
    print(f"Model loaded in {elapsed:.2f} seconds")
    print(f"Model type: {type(model).__name__}")
    print(f"Model task: {model.task}")
    print()
    
    # Show that models are cached
    print("Loading the first model again (should be instant from cache):")
    start_time = time.time()
    model = get_model("yolov8n.pt")
    elapsed = time.time() - start_time
    print(f"Model loaded from cache in {elapsed:.6f} seconds")
    
    # Show currently loaded models in cache:
    print("\nCurrently loaded models in cache:")
    for name, model in model_manager.loaded_models.items():
        print(f"  - {name}")
    
    # Check if models exist in the models directory
    print("\nModels saved in the models directory:")
    for model_name in model_manager.loaded_models.keys():
        model_path = model_manager.models_dir / model_name
        if model_path.exists():
            print(f"  - {model_name} ({model_path})")
        else:
            print(f"  - {model_name} (Not saved to models directory yet)")
    
    print(f"\nTo use these models in your code, simply import the get_model function:")
    print("from models import get_model")
    print("model = get_model('yolov8n.pt')  # Will use cached model if available")


if __name__ == "__main__":
    main()
