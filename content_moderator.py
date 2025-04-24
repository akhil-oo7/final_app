from transformers import AutoModelForImageClassification, AutoFeatureExtractor
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

class ContentModerator:
    def __init__(self, model_name="microsoft/resnet-50", train_mode=False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        
        # Use smaller feature extractor
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        
        if not train_mode:
            model_path = os.path.join("models", "best_model")
            if os.path.exists(model_path):
                # Load with lower memory footprint
                self.model = AutoModelForImageClassification.from_pretrained(
                    model_path,
                    num_labels=2,
                    low_cpu_mem_usage=True
                ).to(self.device)
                self.model.eval()
            else:
                raise FileNotFoundError("Trained model not found.")
    
    def analyze_frames(self, frames):
        results = []
        batch_size = 8  # Reduced batch size
        
        # Simple frame processing without Dataset class to save memory
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            inputs = self.feature_extractor(
                [Image.fromarray(frame).convert('RGB') for frame in batch_frames],
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.softmax(outputs.logits, dim=1)
                
                for pred in predictions:
                    violence_prob = pred[1].item()
                    flagged = violence_prob > 0.3
                    
                    results.append({
                        'flagged': flagged,
                        'reason': "Detected violence" if flagged else "No inappropriate content detected",
                        'confidence': violence_prob if flagged else 1 - violence_prob
                    })
            
            # Clear memory
            del inputs
            torch.cuda.empty_cache() if self.device == "cuda" else None
        
        return results