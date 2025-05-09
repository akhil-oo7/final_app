import cv2
import numpy as np
import gc

class VideoProcessor:
    def __init__(self, frame_interval=15, target_size=(224, 224)):
        self.frame_interval = frame_interval
        self.target_size = target_size
    
    def extract_frames(self, video_path, max_frames=None):
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError("Could not open video file")
        
        try:
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % self.frame_interval == 0:
                    try:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_resized = cv2.resize(frame_rgb, self.target_size)
                        frames.append(frame_resized)
                        
                        if max_frames and len(frames) >= max_frames:
                            break
                    except Exception as e:
                        print(f"Error processing frame {frame_count}: {str(e)}")
                        continue
                
                frame_count += 1
            
            return frames
        
        finally:
            cap.release()
            gc.collect()
    
    def extract_frames_stream(self, video_path, max_frames=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")
        
        try:
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % self.frame_interval == 0:
                    try:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_resized = cv2.resize(frame_rgb, self.target_size)
                        yield frame_resized
                        
                        if max_frames and frame_count // self.frame_interval >= max_frames:
                            break
                    except Exception as e:
                        print(f"Error processing frame {frame_count}: {str(e)}")
                        continue
                
                frame_count += 1
        
        finally:
            cap.release()
            gc.collect()