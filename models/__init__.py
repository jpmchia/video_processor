import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from ultralytics import YOLO


class ModelManager:
    """Manages YOLO models for object detection, including automatic downloading."""
    
    def __init__(self, models_dir: Optional[Union[str, Path]] = None):
        """Initialize the ModelManager.
        
        Args:
            models_dir: Optional directory to store downloaded models. If None, creates a 'models/weights' directory.
        """
        if models_dir is None:
            # Create a default models directory in the project
            project_root = Path(__file__).parent.parent
            models_dir = project_root / 'models' / 'weights'
        
        self.models_dir = Path(models_dir)
        
        # Create the models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.loaded_models: Dict[str, YOLO] = {}
    
    def get_model(self, model_name: str) -> YOLO:
        """Get a YOLO model, downloading it if necessary.
        
        Args:
            model_name: Name of the model (e.g., 'yolov8n.pt', 'yolov8s.pt', 'yolo11n.pt')
            
        Returns:
            YOLO model instance
        """
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        # Determine the model path
        model_path = self.models_dir / model_name
        
        # Check if we already have the model in our custom directory
        if model_path.exists():
            # Load from our custom directory
            model = YOLO(str(model_path), task='detect')
        else:
            # Load and cache the model (ultralytics will download it if needed)
            model = YOLO(model_name, task='detect')
            
            # Copy the model file to our custom directory if possible
            if hasattr(model, 'ckpt_path') and Path(model.ckpt_path).exists():
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                # Use shutil.copy2 to preserve metadata
                import shutil
                shutil.copy2(model.ckpt_path, model_path)
        
        self.loaded_models[model_name] = model
        return model
    
    def list_available_models(self) -> List[str]:
        """List available pre-trained YOLO models.
        
        Returns:
            List of available model names
        """
        return [
            # YOLOv8 models
            'yolov8n.pt',  # Nano model
            'yolov8s.pt',  # Small model
            'yolov8m.pt',  # Medium model
            'yolov8l.pt',  # Large model
            'yolov8x.pt',  # Extra large model
            # YOLO11 models
            'yolo11n.pt',  # Nano model
            'yolo11s.pt',  # Small model
            'yolo11m.pt',  # Medium model
            'yolo11l.pt',  # Large model
            'yolo11x.pt',  # Extra large model
        ]


# Create a default model manager instance for easy import
model_manager = ModelManager()

# Function to get a model (convenience function)
def get_model(model_name: str = 'yolov8n.pt') -> YOLO:
    """Get a YOLO model, downloading it if necessary.
    
    Args:
        model_name: Name of the model (default: 'yolov8n.pt')
        
    Returns:
        YOLO model instance
    """
    return model_manager.get_model(model_name)
