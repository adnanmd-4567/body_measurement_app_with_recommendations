import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # DON'T CHANGE THIS !!!

from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import os
import mediapipe as mp
import math

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src/static/uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the image to get measurements
        measurements = process_image_with_mediapipe(filepath)
        
        return jsonify(measurements)
    
    return jsonify({'error': 'File type not allowed'}), 400

def calculate_distance(landmark1, landmark2):
    """Calculate Euclidean distance between two landmarks"""
    return math.sqrt(
        (landmark1.x - landmark2.x) ** 2 +
        (landmark1.y - landmark2.y) ** 2
    )

def process_image_with_mediapipe(image_path):
    """
    Process the uploaded image using MediaPipe Pose to estimate body measurements.
    This provides more accurate measurements than simple contour detection.
    """
    try:
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            return {'error': 'Failed to read image'}
        
        # Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, _ = img.shape
        
        # Process the image with MediaPipe Pose
        results = pose.process(img_rgb)
        
        if not results.pose_landmarks:
            return {'error': 'No pose landmarks detected. Please use a clearer full-body image.'}
        
        landmarks = results.pose_landmarks.landmark
        
        # Create a visualization image for debugging
        debug_img = img.copy()
        mp_drawing.draw_landmarks(
            debug_img, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS
        )
        debug_path = os.path.join(app.config['UPLOAD_FOLDER'], 'debug_' + os.path.basename(image_path))
        cv2.imwrite(debug_path, debug_img)
        
        # Calculate pixel to cm ratio based on height
        # Assuming average adult height is around 170 cm
        # Use the distance from nose to ankle as reference
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        
        # Use the average of both ankles for better accuracy
        ankle_y = (left_ankle.y + right_ankle.y) / 2
        body_height_pixels = (ankle_y - nose.y) * height
        
        # Assuming the visible body is about 85% of total height (170cm)
        pixel_to_cm = (170 * 0.85) / body_height_pixels
        
        # Calculate shoulder width
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        shoulder_width = calculate_distance(left_shoulder, right_shoulder) * width * pixel_to_cm
        
        # Calculate chest width (approximated using shoulders and a factor)
        # The chest is typically slightly narrower than shoulders
        chest_width = shoulder_width * 0.95
        
        # Calculate waist width (approximated using hip points and a factor)
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        hip_width = calculate_distance(left_hip, right_hip) * width * pixel_to_cm
        
        # Waist is typically narrower than hips
        waist_width = hip_width * 0.9
        
        # Calculate height
        height_cm = body_height_pixels * pixel_to_cm / 0.85  # Adjust back to full height
        
        # Determine clothing sizes based on measurements
        # These are more realistic size charts
        
        # Shirt size (based on chest)
        if chest_width < 31:
            shirt_size = "XS"
        elif chest_width < 33:
            shirt_size = "S"
        elif chest_width < 35:
            shirt_size = "M"
        elif chest_width < 37:
            shirt_size = "L"
        elif chest_width < 39:
            shirt_size = "XL"
        else:
            shirt_size = "XXL"
        
        # Pants size (based on waist)
        if waist_width < 16:
            pants_size = "XS (28-30)"
        elif waist_width < 17:
            pants_size = "S (30-32)"
        elif waist_width < 19:
            pants_size = "M (32-34)"
        elif waist_width < 21:
            pants_size = "L (34-36)"
        elif waist_width < 22:
            pants_size = "XL (36-38)"
        else:
            pants_size = "XXL (38+)"
        
        return {
            'measurements': {
                'height': round(height_cm, 1),
                'shoulder': round((shoulder_width + 10), 1),
                'chest': round((chest_width * 2.85), 1),
                'waist': round((waist_width * 4.3), 1),
                'hips': round((hip_width * 4.63), 1)
            },
            'sizes': {
                'shirt': shirt_size,
                'pants': pants_size
            },
            'debug_image': os.path.basename(debug_path)
        }
    
    except Exception as e:
        return {'error': str(e)}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
