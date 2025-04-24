from transformers import AutoModelForImageClassification, AutoFeatureExtractor
import torch
from PIL import Image
import numpy as np
import os

class ContentModerator:
    def __init__(self, model_name="microsoft/resnet-50", train_mode=False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = None
        self.train_mode = train_mode
    
    def load_model(self):
        if self.model is None and not self.train_mode:
            model_path = os.path.join("models", "best_model")
            try:
                if os.path.exists(model_path):
                    self.model = AutoModelForImageClassification.from_pretrained(
                        model_path,
                        num_labels=2,
                        low_cpu_mem_usage=True
                    )
                else:
                    print("Warning: Fine-tuned model not found, using base model")
                    self.model = AutoModelForImageClassification.from_pretrained(
                        self.model_name,
                        num_labels=2,
                        low_cpu_mem_usage=True
                    )
                self.model.to(self.device)
                self.model.eval()
            except Exception as e:
                raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def analyze_frames(self, frames):
        self.load_model()
        results = []
        batch_size = 8
        
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            results.extend(self._process_batch(batch_frames, i))
            torch.cuda.empty_cache() if self.device == "cuda" else None
        
        return results
    
    def analyze_frames_stream(self, frame_generator, batch_size=8):
        self.load_model()
        results = []
        batch = []
        frame_idx = 0
        
        for frame in frame_generator:
            batch.append(frame)
            if len(batch) == batch_size:
                results.extend(self._process_batch(batch, frame_idx))
                batch = []
                frame_idx += len(batch)
                torch.cuda.empty_cache() if self.device == "cuda" else None
        
        if batch:
            results.extend(self._process_batch(batch, frame_idx))
        
        return results
    
    def _process_batch(self, batch, start_idx):
        inputs = self.feature_extractor(
            [Image.fromarray(frame).convert('RGB') for frame in batch],
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=1)
        
        results = []
        for i, pred in enumerate(predictions):
            violence_prob = pred[1].item()
            flagged = violence_prob > 0.3
            results.append({
                'flagged': flagged,
                'reason': "Detected violence" if flagged else "No inappropriate content detected",
                'confidence': violence_prob if flagged else 1 - violence_prob,
                'frame': (start_idx + i) * 15  # Assuming frame_interval=15
            })
        
        del inputs
        return results