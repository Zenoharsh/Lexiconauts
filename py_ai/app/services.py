import cv2
import requests
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pickle
import os
from pathlib import Path

# --- Load Models (Done once when the service starts) ---
print("Loading MoveNet model...")
movenet = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
print("MoveNet model loaded.")

class FormPredictor:
    # ... (The FormPredictor class from before remains unchanged) ...
    def __init__(self, model_path):
        print(f"Loading trained classifier from {model_path}...")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found at {model_path}.")
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        print("✓ Classifier model loaded successfully.")

    def predict_form(self, keypoints_sequence):
        # Placeholder for form prediction logic
        return "good", 0.95

form_predictor = FormPredictor(model_path="models/final_model.pkl")

# --- Helper Functions ---
def movenet_inference(image):
    input_image = tf.convert_to_tensor(image, dtype=tf.uint8)
    input_image = tf.image.resize_with_pad(tf.expand_dims(input_image, axis=0), 256, 256)
    input_tensor = tf.cast(input_image, dtype=tf.int32)
    results = movenet.signatures['serving_default'](input_tensor)
    keypoints = results['output_0'].numpy()[0, 0, :, :]
    return keypoints

def _calculate_angle(a, b, c):
    """Calculates the angle at point b, formed by lines ab and bc."""
    a = np.array(a) # First point
    b = np.array(b) # Mid point
    c = np.array(c) # End point
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    return angle

# --- Analysis Functions ---
def _calculate_jump_height(all_keypoints, athlete_height_cm, frame_height):
    com_y_positions = []
    for keypoints in all_keypoints:
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        if left_hip[2] > 0.3 and right_hip[2] > 0.3:
            com_y = (left_hip[0] + right_hip[0]) / 2
            com_y_positions.append(com_y * frame_height)
    
    if not com_y_positions: return 0
    
    crouch_y = max(com_y_positions)
    peak_y = min(com_y_positions)
    jump_height_pixels = crouch_y - peak_y
    
    # Simple calibration: Assume standing height is roughly the full frame height
    pixels_per_cm = frame_height / athlete_height_cm
    if pixels_per_cm == 0: return 0
    
    jump_height_cm = jump_height_pixels / pixels_per_cm
    return round(jump_height_cm, 2)

def _count_pushup_reps(all_keypoints):
    reps = 0
    stage = "up" # Start in the 'up' position
    
    # Keypoint indices for pushups: left_shoulder=5, left_elbow=7, left_wrist=9
    for keypoints in all_keypoints:
        shoulder = keypoints[5][:2]
        elbow = keypoints[7][:2]
        wrist = keypoints[9][:2]
        
        # Calculate elbow angle
        angle = _calculate_angle(shoulder, elbow, wrist)
        
        # State machine for rep counting
        if angle > 160: # Arms are straight
            stage = "up"
        if angle < 90 and stage == 'up': # Arms are bent, and we were previously up
            stage = "down"
            reps += 1
            
    return reps

# --- Main Router Function ---
def analyze_video(video_url, exercise_type, athlete_height_cm=180): # Added default height
    try:
        response = requests.get(video_url, stream=True)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to download video: {e}"}

    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as f: f.write(response.content)

    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened(): return {"error": "Could not open video file."}

    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    all_keypoints = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        keypoints = movenet_inference(frame)
        all_keypoints.append(keypoints)
    cap.release()
    os.remove(temp_video_path)

    if not all_keypoints: return {"error": "Could not detect pose in the video."}

    # --- ROUTER LOGIC ---
    if exercise_type == 'vertical_jump':
        result_value = _calculate_jump_height(all_keypoints, athlete_height_cm, frame_height)
        result_unit = "cm"
    elif exercise_type == 'pushup':
        reps = _count_pushup_reps(all_keypoints)
        form_label, confidence = form_predictor.predict_form(all_keypoints)
        result_value = f"{reps} reps, form: {form_label} ({confidence:.0%})"
        result_unit = "reps"
    else:
        return {"error": f"Analysis for '{exercise_type}' is not implemented yet."}

    return {"result": result_value, "unit": result_unit, "status": "success"}