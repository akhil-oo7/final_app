from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from video_processor import VideoProcessor
from content_moderator import ContentModerator
from dotenv import load_dotenv
import gc

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Reduced to 50MB

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize components lazily to save memory
video_processor = None
content_moderator = None

def get_processor():
    global video_processor
    if video_processor is None:
        video_processor = VideoProcessor(frame_interval=60)  # Increased interval
    return video_processor

def get_moderator():
    global content_moderator
    if content_moderator is None:
        content_moderator = ContentModerator(train_mode=False)
    return content_moderator

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    allowed_extensions = {'mp4', 'avi', 'mov', 'mkv'}
    if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return jsonify({'error': 'Invalid file format. Supported formats: mp4, avi, mov, mkv'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process in chunks to save memory
        processor = get_processor()
        moderator = get_moderator()
        
        # Get frame count first
        cap = cv2.VideoCapture(filepath)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Process in chunks of 100 frames
        chunk_size = 100
        results = []
        processed_frames = 0
        
        while processed_frames < total_frames:
            frames = processor.extract_frames(
                filepath, 
                max_frames=chunk_size,
                start_frame=processed_frames
            )
            
            if not frames:
                break
                
            chunk_results = moderator.analyze_frames(frames)
            results.extend(chunk_results)
            
            processed_frames += len(frames)
            del frames
            gc.collect()
        
        # Calculate results
        unsafe_frames = [r for r in results if r['flagged']]
        total_analyzed = len(results)
        
        if total_analyzed == 0:
            return jsonify({'error': 'No frames were processed'}), 400
            
        unsafe_percentage = (len(unsafe_frames) / total_analyzed) * 100
        
        response = {
            'status': 'UNSAFE' if unsafe_frames else 'SAFE',
            'total_frames': total_analyzed,
            'unsafe_frames': len(unsafe_frames),
            'unsafe_percentage': unsafe_percentage,
            'confidence': 1.0 if not unsafe_frames else max(r['confidence'] for r in unsafe_frames),
            'details': []
        }
        
        if unsafe_frames:
            for frame_idx, result in enumerate(results):
                if result['flagged']:
                    response['details'].append({
                        'frame': frame_idx,
                        'reason': result['reason'],
                        'confidence': result['confidence']
                    })
        
        os.remove(filepath)
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)