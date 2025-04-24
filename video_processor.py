import cv2
import numpy as np
import gc

class VideoProcessor:
    def __init__(self, frame_interval=30, target_size=(224, 224)):
        self.frame_interval = frame_interval
        self.target_size = target_size
    
    def extract_frames(self, video_path, max_frames=None, start_frame=0):
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError("Could not open video file")
        
        try:
            # Set starting position
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            frames_extracted = 0
            while True:
                ret, frame = cap.read()
                if not ret or (max_frames and frames_extracted >= max_frames):
                    break
                
                if frames_extracted % self.frame_interval == 0:
                    try:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_resized = cv2.resize(frame_rgb, self.target_size)
                        frames.append(frame_resized)
                    except Exception as e:
                        print(f"Error processing frame: {str(e)}")
                        continue
                
                frames_extracted += 1
                
        finally:
            cap.release()
            
        return frames