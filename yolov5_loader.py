"""
Standalone YOLOv5 model loader to avoid utils naming conflict
"""
import sys
from pathlib import Path

def load_yolov5_model(model_path, conf_threshold=0.5):
    """
    Load YOLOv5 model in an isolated context
    
    Args:
        model_path: Path to YOOv5 weights file
        conf_threshold: Confidence threshold
        
    Returns:
        YOLOv5 model or None if failed
    """
    try:
        # Remove project root from sys.path to avoid utils conflict
        project_root = str(Path(__file__).parent.parent)
        original_path = sys.path.copy()
        
        # Filter out project paths
        sys.path = [p for p in sys.path if not p.startswith(project_root)]
        
        try:
            import torch
            model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                 path=model_path, force_reload=False, 
                                 verbose=False, trust_repo=True)
            model.conf = conf_threshold
            return model
        finally:
            # Restore original path
            sys.path = original_path
            
    except Exception as e:
        # Restore path on error too
        if 'original_path' in locals():
            sys.path = original_path
        raise e
