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
        
        # Initialize processor with smaller frame interval
        processor = VideoProcessor(frame_interval=15)  # Reduced from 60 to 15
        moderator = ContentModerator(train_mode=False)
        
        # Extract ALL frames first (for accurate frame numbering)
        try:
            frames = processor.extract_frames(filepath)
            if not frames:
                return jsonify({
                    'status': 'error',
                    'message': 'No frames could be extracted'
                }), 400
                
            # Analyze all frames at once (for simplicity)
            results = moderator.analyze_frames(frames)
            
            # Calculate results with proper frame indices
            unsafe_frames = []
            for frame_idx, result in enumerate(results):
                if result['flagged']:
                    unsafe_frames.append({
                        'frame': frame_idx * processor.frame_interval,  # Actual frame number
                        'reason': result['reason'],
                        'confidence': result['confidence']
                    })
            
            # Prepare response
            response = {
                'status': 'UNSAFE' if unsafe_frames else 'SAFE',
                'total_frames': len(frames),
                'unsafe_frames': len(unsafe_frames),
                'unsafe_percentage': (len(unsafe_frames)/len(frames))*100,
                'confidence': 1.0 if not unsafe_frames else max(r['confidence'] for r in unsafe_frames),
                'details': unsafe_frames  # Only include flagged frames
            }
            
            return jsonify(response)
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Analysis failed: {str(e)}'
            }), 500
            
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)