from flask import Flask, render_template, request, jsonify
from flask_sse import sse
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
import cv2
from werkzeug.utils import secure_filename
from video_processor import VideoProcessor
from content_moderator import ContentModerator
from dotenv import load_dotenv
import gc

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB limit
app.config['SSE_REDIS_URL'] = os.environ.get('SSE_REDIS_URL')  # Redis URL for flask_sse

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize Flask-Limiter with updated API
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"]
)

app.register_blueprint(sse, url_prefix='/stream')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
@limiter.limit("5 per minute")
def analyze_video():
    if 'video' not in request.files:
        return jsonify({'status': 'error', 'message': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400
    
    allowed_extensions = {'mp4', 'avi', 'mov', 'mkv'}
    if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return jsonify({'status': 'error', 'message': 'Invalid file format. Supported formats: mp4, avi, mov, mkv'}), 400
    
    if file.content_length and file.content_length > app.config['MAX_CONTENT_LENGTH']:
        return jsonify({'status': 'error', 'message': 'File size exceeds 50MB limit'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        processor = VideoProcessor(frame_interval=15)
        moderator = ContentModerator(train_mode=False)
        
        frame_generator = processor.extract_frames_stream(filepath)
        total_frames = int(cv2.VideoCapture(filepath).get(cv2.CAP_PROP_FRAME_COUNT) / 15)  # Approximate
        processed_frames = 0
        
        results = []
        for batch_results in moderator.analyze_frames_stream(frame_generator):
            results.extend(batch_results)
            processed_frames += len(batch_results)
            if total_frames > 0:
                progress = min((processed_frames / total_frames) * 100, 100)
                sse.publish({"progress": progress}, type='progress')
        
        if not results:
            return jsonify({
                'status': 'error',
                'message': 'No frames could be extracted'
            }), 400
        
        unsafe_frames = [r for r in results if r['flagged']]
        response = {
            'status': 'UNSAFE' if unsafe_frames else 'SAFE',
            'total_frames': len(results),
            'unsafe_frames': len(unsafe_frames),
            'unsafe_percentage': (len(unsafe_frames) / len(results)) * 100 if results else 0,
            'confidence': 1.0 if not unsafe_frames else max(r['confidence'] for r in unsafe_frames),
            'details': unsafe_frames
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Analysis failed: {str(e)}'
        }), 500
    
    finally:
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        gc.collect()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)