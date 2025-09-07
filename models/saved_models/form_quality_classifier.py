# ==============================================================================
# COMPLETE EXERCISE FORM QUALITY CLASSIFIER WITH DATASET INTEGRATION
# File: complete_form_classifier.py
# ==============================================================================

import numpy as np
import pandas as pd
import cv2
import json
import os
import pickle
import requests
import zipfile
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
import joblib

# Deep Learning (optional)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. Using sklearn models only.")

# Pose Detection
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    print("MediaPipe not available. Using pre-extracted features only.")

class ExerciseDatasetManager:
    """Manages multiple exercise datasets and provides unified interface"""
    
    def __init__(self, data_dir="./exercise_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.datasets = {}
        
        # Available dataset configurations
        self.available_datasets = {
            "fitness_pose": {
                "url": "https://www.kaggle.com/api/v1/datasets/download/hasyimabdillah/workoutfitness-computer-vision",
                "type": "kaggle",
                "exercises": ["squat", "pushup", "situp", "plank"]
            },
            "synthetic_data": {
                "type": "generated",
                "exercises": ["squat", "pushup", "situp", "plank", "lunge", "jump"]
            },
            "ui_prmd": {
                "url": "https://github.com/abdullahmahmood/UI-PRMD",
                "type": "github",
                "exercises": ["squat", "deadlift", "bench_press", "overhead_press"]
            }
        }

    def download_kaggle_dataset(self, dataset_name):
        """Download dataset from Kaggle (requires kaggle API setup)"""
        try:
            import kaggle
            dataset_path = self.data_dir / dataset_name
            dataset_path.mkdir(exist_ok=True)
            
            if dataset_name == "fitness_pose":
                kaggle.api.dataset_download_files(
                    'hasyimabdillah/workoutfitness-computer-vision',
                    path=str(dataset_path),
                    unzip=True
                )
            print(f"Downloaded {dataset_name} successfully!")
            return True
        except Exception as e:
            print(f"Error downloading {dataset_name}: {e}")
            print("Please download manually from Kaggle or ensure kaggle API is set up")
            return False

    def generate_synthetic_dataset(self, num_samples_per_exercise=1000):
        """Generate synthetic exercise data for testing"""
        synthetic_data = []
        exercises = ["squat", "pushup", "situp", "plank", "lunge", "jump"]
        
        for exercise in exercises:
            for _ in range(num_samples_per_exercise):
                # Generate realistic-looking features based on exercise type
                features = self._generate_exercise_features(exercise)
                quality = np.random.choice(["good", "poor"], p=[0.6, 0.4])
                
                # Add noise based on quality
                if quality == "poor":
                    features += np.random.normal(0, 0.3, features.shape)
                
                synthetic_data.append({
                    "exercise": exercise,
                    "features": features,
                    "quality": quality,
                    "quality_score": 8.5 if quality == "good" else 4.2
                })
        
        # Save synthetic dataset
        synthetic_path = self.data_dir / "synthetic_dataset.pkl"
        with open(synthetic_path, 'wb') as f:
            pickle.dump(synthetic_data, f)
        
        print(f"Generated {len(synthetic_data)} synthetic samples")
        return synthetic_data

    def _generate_exercise_features(self, exercise):
        """Generate realistic features for each exercise type"""
        if exercise == "squat":
            return np.array([
                np.random.normal(85, 10),    # knee_angle_mean
                np.random.normal(15, 5),     # knee_angle_std
                np.random.normal(95, 8),     # back_angle_mean
                np.random.normal(10, 3),     # back_angle_std
                np.random.normal(5, 2),      # knee_tracking_error
                np.random.normal(2.5, 0.5),  # rep_duration
            ])
        elif exercise == "pushup":
            return np.array([
                np.random.normal(90, 12),    # elbow_angle_mean
                np.random.normal(20, 8),     # elbow_angle_std
                np.random.normal(175, 5),    # body_alignment
                np.random.normal(8, 3),      # body_alignment_std
                np.random.normal(2.0, 0.3),  # rep_duration
                np.random.normal(15, 5),     # range_of_motion
            ])
        elif exercise == "situp":
            return np.array([
                np.random.normal(45, 10),    # torso_angle_mean
                np.random.normal(12, 4),     # torso_angle_std
                np.random.normal(20, 8),     # neck_strain
                np.random.normal(5, 2),      # hip_movement
                np.random.normal(1.8, 0.2),  # rep_duration
            ])
        # Add more exercises as needed
        else:
            return np.random.normal(50, 10, 6)  # Generic features

    def load_all_datasets(self):
        """Load and combine all available datasets"""
        all_data = []
        
        # Try to load Kaggle dataset
        kaggle_data = self.load_kaggle_dataset()
        if kaggle_data:
            all_data.extend(kaggle_data)
        
        # Generate synthetic data
        synthetic_data = self.generate_synthetic_dataset()
        all_data.extend(synthetic_data)
        
        print(f"Total loaded samples: {len(all_data)}")
        return all_data

    def load_kaggle_dataset(self):
        """Load and process Kaggle fitness dataset"""
        dataset_path = self.data_dir / "fitness_pose"
        if not dataset_path.exists():
            if not self.download_kaggle_dataset("fitness_pose"):
                return None
        
        # Process the dataset (this is a placeholder - actual implementation depends on dataset structure)
        data = []
        try:
            # Look for CSV files or image directories
            csv_files = list(dataset_path.glob("*.csv"))
            if csv_files:
                df = pd.read_csv(csv_files[0])
                # Convert CSV data to our format
                for _, row in df.iterrows():
                    data.append({
                        "exercise": row.get("exercise", "unknown"),
                        "features": np.array([row.get(f"feature_{i}", 0) for i in range(10)]),
                        "quality": row.get("quality", "good"),
                        "quality_score": row.get("score", 7.0)
                    })
        except Exception as e:
            print(f"Error loading Kaggle dataset: {e}")
            return None
        
        return data


class ExerciseFormClassifier:
    """Complete exercise form quality classifier with multiple models"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_extractors = {
            'pushup': self._extract_pushup_features,
            'squat': self._extract_squat_features,
            'situp': self._extract_situp_features,
            'plank': self._extract_plank_features,
            'lunge': self._extract_lunge_features,
            'jump': self._extract_jump_features,
        }
        
        # Exercise-specific parameters for form evaluation
        self.exercise_params = {
            'pushup': {
                'elbow_angle_min': 70, 'elbow_angle_max': 110,
                'body_alignment_min': 160, 'rep_time_min': 1.0, 'rep_time_max': 5.0
            },
            'squat': {
                'knee_angle_min': 70, 'knee_angle_max': 110,
                'back_angle_min': 70, 'back_angle_max': 110,
                'knee_tracking_threshold': 20
            },
            'situp': {
                'torso_angle_min': 30, 'torso_angle_max': 60,
                'neck_strain_threshold': 50, 'hip_stability_threshold': 20
            }
        }
        
        # Initialize pose detection if available
        if MP_AVAILABLE:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5
            )
            self.mp_draw = mp.solutions.drawing_utils

    def _extract_pushup_features(self, keypoints_sequence):
        """Extract pushup-specific features from keypoint sequence"""
        elbow_angles = []
        body_alignments = []
        ranges_of_motion = []
        
        for keypoints in keypoints_sequence:
            if len(keypoints) < 17:  # Standard pose keypoints
                continue
                
            # Calculate elbow angle (shoulder-elbow-wrist)
            left_elbow_angle = self._calculate_angle_from_keypoints(keypoints, 5, 7, 9)
            right_elbow_angle = self._calculate_angle_from_keypoints(keypoints, 6, 8, 10)
            
            if left_elbow_angle and right_elbow_angle:
                elbow_angles.append((left_elbow_angle + right_elbow_angle) / 2)
            
            # Calculate body alignment (shoulder-hip-ankle)
            body_alignment = self._calculate_angle_from_keypoints(keypoints, 5, 11, 15)
            if body_alignment:
                body_alignments.append(body_alignment)
        
        if not elbow_angles:
            return np.array([])
        
        features = [
            np.mean(elbow_angles), np.std(elbow_angles),
            np.mean(body_alignments), np.std(body_alignments),
            max(elbow_angles) - min(elbow_angles),  # Range of motion
            len(keypoints_sequence) / 30.0  # Estimated duration (assuming 30 FPS)
        ]
        return np.array(features)

    def _extract_squat_features(self, keypoints_sequence):
        """Extract squat-specific features"""
        knee_angles = []
        back_angles = []
        knee_tracking_errors = []
        
        for keypoints in keypoints_sequence:
            if len(keypoints) < 17:
                continue
            
            # Knee angle (hip-knee-ankle)
            left_knee = self._calculate_angle_from_keypoints(keypoints, 11, 13, 15)
            right_knee = self._calculate_angle_from_keypoints(keypoints, 12, 14, 16)
            
            if left_knee and right_knee:
                knee_angles.append((left_knee + right_knee) / 2)
            
            # Back angle (shoulder-hip-knee)
            back_angle = self._calculate_angle_from_keypoints(keypoints, 5, 11, 13)
            if back_angle:
                back_angles.append(back_angle)
            
            # Knee tracking (knee should track over ankle)
            if len(keypoints) >= 16:
                knee_ankle_distance = abs(keypoints[13][0] - keypoints[15][0])  # Left side
                knee_tracking_errors.append(knee_ankle_distance)
        
        if not knee_angles:
            return np.array([])
        
        features = [
            np.mean(knee_angles), np.std(knee_angles),
            np.mean(back_angles), np.std(back_angles),
            np.mean(knee_tracking_errors) if knee_tracking_errors else 0,
            len(keypoints_sequence) / 30.0
        ]
        return np.array(features)

    def _extract_situp_features(self, keypoints_sequence):
        """Extract sit-up specific features"""
        torso_angles = []
        neck_strains = []
        hip_movements = []
        
        for keypoints in keypoints_sequence:
            if len(keypoints) < 12:
                continue
            
            # Torso angle (hip-shoulder-head)
            torso_angle = self._calculate_angle_from_keypoints(keypoints, 11, 5, 0)
            if torso_angle:
                torso_angles.append(torso_angle)
            
            # Neck strain (head-shoulder distance)
            if keypoints[0][2] > 0.3 and keypoints[5][2] > 0.3:  # Confidence check
                neck_strain = np.linalg.norm(
                    np.array(keypoints[0][:2]) - np.array(keypoints[5][:2])
                )
                neck_strains.append(neck_strain)
            
            # Hip stability (hip position variance)
            if keypoints[11][2] > 0.3:
                hip_movements.append(keypoints[11][1])  # Y coordinate
        
        if not torso_angles:
            return np.array([])
        
        features = [
            np.mean(torso_angles), np.std(torso_angles),
            np.mean(neck_strains) if neck_strains else 0,
            np.std(hip_movements) if len(hip_movements) > 1 else 0,
            len(keypoints_sequence) / 30.0
        ]
        return np.array(features)

    def _extract_plank_features(self, keypoints_sequence):
        """Extract plank-specific features"""
        body_alignments = []
        hip_sags = []
        stability_scores = []
        
        for keypoints in keypoints_sequence:
            if len(keypoints) < 16:
                continue
            
            # Body alignment (shoulder-hip-ankle)
            alignment = self._calculate_angle_from_keypoints(keypoints, 5, 11, 15)
            if alignment:
                body_alignments.append(alignment)
            
            # Hip sag (measure deviation from straight line)
            if all(keypoints[i][2] > 0.3 for i in [5, 11, 15]):  # Confidence check
                shoulder_y = keypoints[5][1]
                hip_y = keypoints[11][1]
                ankle_y = keypoints[15][1]
                
                # Calculate how much hip deviates from shoulder-ankle line
                expected_hip_y = (shoulder_y + ankle_y) / 2
                hip_sag = abs(hip_y - expected_hip_y)
                hip_sags.append(hip_sag)
        
        if not body_alignments:
            return np.array([])
        
        features = [
            np.mean(body_alignments), np.std(body_alignments),
            np.mean(hip_sags) if hip_sags else 0,
            np.std(body_alignments),  # Stability measure
            len(keypoints_sequence) / 30.0
        ]
        return np.array(features)

    def _extract_lunge_features(self, keypoints_sequence):
        """Extract lunge-specific features"""
        front_knee_angles = []
        back_knee_angles = []
        balance_scores = []
        
        for keypoints in keypoints_sequence:
            if len(keypoints) < 17:
                continue
            
            # Front and back knee angles
            left_knee = self._calculate_angle_from_keypoints(keypoints, 11, 13, 15)
            right_knee = self._calculate_angle_from_keypoints(keypoints, 12, 14, 16)
            
            if left_knee and right_knee:
                # Assume left leg is front (this could be detected automatically)
                front_knee_angles.append(left_knee)
                back_knee_angles.append(right_knee)
            
            # Balance (center of mass stability)
            if keypoints[11][2] > 0.3 and keypoints[12][2] > 0.3:
                com_x = (keypoints[11][0] + keypoints[12][0]) / 2
                balance_scores.append(com_x)
        
        if not front_knee_angles:
            return np.array([])
        
        features = [
            np.mean(front_knee_angles), np.min(front_knee_angles),
            np.mean(back_knee_angles), np.std(back_knee_angles),
            np.std(balance_scores) if len(balance_scores) > 1 else 0,
            len(keypoints_sequence) / 30.0
        ]
        return np.array(features)

    def _extract_jump_features(self, keypoints_sequence):
        """Extract jumping-specific features"""
        takeoff_angles = []
        landing_angles = []
        jump_heights = []
        
        # Detect takeoff and landing phases
        hip_heights = [kp[11][1] for kp in keypoints_sequence if len(kp) > 11 and kp[11][2] > 0.3]
        
        if len(hip_heights) < 5:
            return np.array([])
        
        # Find takeoff (lowest point at start) and peak (highest point)
        min_height = min(hip_heights)
        max_height = max(hip_heights)
        jump_height = max_height - min_height
        
        for i, keypoints in enumerate(keypoints_sequence):
            if len(keypoints) < 16:
                continue
            
            # Analyze knee angles during different phases
            knee_angle = self._calculate_angle_from_keypoints(keypoints, 11, 13, 15)
            if knee_angle:
                if i < len(keypoints_sequence) // 3:  # Takeoff phase
                    takeoff_angles.append(knee_angle)
                elif i > 2 * len(keypoints_sequence) // 3:  # Landing phase
                    landing_angles.append(knee_angle)
        
        features = [
            np.mean(takeoff_angles) if takeoff_angles else 90,
            np.min(takeoff_angles) if takeoff_angles else 90,
            np.mean(landing_angles) if landing_angles else 90,
            jump_height,
            len(keypoints_sequence) / 30.0
        ]
        return np.array(features)

    def _calculate_angle_from_keypoints(self, keypoints, idx1, idx2, idx3):
        """Calculate angle between three keypoints with confidence check"""
        if (idx1 >= len(keypoints) or idx2 >= len(keypoints) or idx3 >= len(keypoints)):
            return None
            
        if any(keypoints[i][2] < 0.3 for i in [idx1, idx2, idx3]):
            return None  # Low confidence keypoints
        
        p1 = np.array(keypoints[idx1][:2])
        p2 = np.array(keypoints[idx2][:2])
        p3 = np.array(keypoints[idx3][:2])
        
        # Calculate vectors
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Calculate angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle) * 180 / np.pi
        
        return angle

    def extract_features_from_video(self, video_path, exercise_type):
        """Extract features from video file using pose detection"""
        if not MP_AVAILABLE:
            raise RuntimeError("MediaPipe not available for video processing")
        
        cap = cv2.VideoCapture(video_path)
        keypoints_sequence = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # Extract keypoint coordinates
                keypoints = []
                for landmark in results.pose_landmarks.landmark:
                    keypoints.append([landmark.x, landmark.y, landmark.visibility])
                keypoints_sequence.append(keypoints)
        
        cap.release()
        
        # Extract exercise-specific features
        if exercise_type in self.feature_extractors:
            features = self.feature_extractors[exercise_type](keypoints_sequence)
        else:
            # Generic feature extraction
            features = self._extract_generic_features(keypoints_sequence)
        
        return features

    def _extract_generic_features(self, keypoints_sequence):
        """Generic feature extraction for unknown exercises"""
        if not keypoints_sequence:
            return np.array([])
        
        # Basic movement analysis
        movements = []
        for keypoints in keypoints_sequence:
            if len(keypoints) >= 17:
                # Calculate overall body movement
                center_mass = np.mean([kp[:2] for kp in keypoints if kp[2] > 0.3], axis=0)
                movements.append(center_mass)
        
        if len(movements) < 2:
            return np.array([0] * 5)
        
        movements = np.array(movements)
        features = [
            np.std(movements[:, 0]),  # X movement variance
            np.std(movements[:, 1]),  # Y movement variance
            np.mean(np.diff(movements, axis=0)),  # Average movement speed
            len(keypoints_sequence) / 30.0,  # Duration estimate
            len([kp for kp in keypoints_sequence[0] if kp[2] > 0.5])  # Visible keypoints
        ]
        return np.array(features)

    def train_models(self, training_data):
        """Train multiple models for exercise form classification"""
        print("Training exercise form classification models...")
        
        # Organize data by exercise type
        exercise_data = {}
        for sample in training_data:
            exercise = sample['exercise']
            if exercise not in exercise_data:
                exercise_data[exercise] = {'features': [], 'labels': []}
            
            exercise_data[exercise]['features'].append(sample['features'])
            exercise_data[exercise]['labels'].append(sample['quality'])
        
        # Train models for each exercise
        for exercise, data in exercise_data.items():
            print(f"Training model for {exercise}...")
            
            X = np.array(data['features'])
            y = np.array(data['labels'])
            
            if len(X) < 10:  # Not enough data
                print(f"Insufficient data for {exercise}, skipping...")
                continue
            
            # Handle feature extraction if needed
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            elif X.ndim > 2:
                X = X.reshape(len(X), -1)
            
            # Remove samples with empty features
            valid_samples = [i for i, features in enumerate(X) if len(features) > 0 and not np.isnan(features).any()]
            if len(valid_samples) < 10:
                print(f"Insufficient valid samples for {exercise}, skipping...")
                continue
            
            X = X[valid_samples]
            y = y[valid_samples]
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Encode labels
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            # Train multiple models and select the best
            models = {
                'rf': RandomForestClassifier(n_estimators=100, random_state=42),
                'gb': GradientBoostingClassifier(random_state=42),
                'mlp': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
            }
            
            best_model = None
            best_score = 0
            best_model_name = ""
            
            for model_name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    score = model.score(X_test, y_test)
                    print(f"  {model_name}: {score:.3f}")
                    
                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_model_name = model_name
                except Exception as e:
                    print(f"  {model_name}: Error - {e}")
            
            if best_model:
                self.models[exercise] = best_model
                self.scalers[exercise] = scaler
                self.label_encoders[exercise] = label_encoder
                print(f"  Best model for {exercise}: {best_model_name} ({best_score:.3f})")
            
            # Detailed evaluation
            if best_model:
                y_pred = best_model.predict(X_test)
                print(f"  Classification Report for {exercise}:")
                print(classification_report(y_test, y_pred, 
                                           target_names=label_encoder.classes_, zero_division=0))

    def predict_form_quality(self, features, exercise_type):
        """Predict form quality for given features"""
        if exercise_type not in self.models:
            return {"error": f"No model trained for {exercise_type}"}
        
        # Prepare features
        features = np.array(features).reshape(1, -1)
        
        # Check for invalid features
        if np.isnan(features).any() or features.shape[1] == 0:
            return {"quality": "unknown", "confidence": 0.0, "score": 0.0}
        
        # Scale features
        features_scaled = self.scalers[exercise_type].transform(features)
        
        # Predict
        model = self.models[exercise_type]
        prediction = model.predict(features_scaled)[0]
        confidence = max(model.predict_proba(features_scaled)[0])
        
        # Decode prediction
        quality = self.label_encoders[exercise_type].inverse_transform([prediction])[0]
        
        # Calculate quality score (1-10 scale)
        quality_score = 8.5 if quality == "good" else 4.0
        
        return {
            "quality": quality,
            "confidence": confidence,
            "score": quality_score,
            "exercise": exercise_type
        }

    def analyze_video(self, video_path, exercise_type):
        """Complete video analysis pipeline"""
        try:
            # Extract features from video
            features = self.extract_features_from_video(video_path, exercise_type)
            
            if len(features) == 0:
                return {"error": "Could not extract features from video"}
            
            # Predict form quality
            result = self.predict_form_quality(features, exercise_type)
            
            # Add detailed analysis
            result["features"] = features.tolist()
            result["analysis"] = self._generate_form_feedback(features, exercise_type)
            
            return result
            
        except Exception as e:
            return {"error": str(e)}

    def _generate_form_feedback(self, features, exercise_type):
        """Generate specific feedback based on exercise parameters"""
        feedback = []
        
        if exercise_type in self.exercise_params:
            params = self.exercise_params[exercise_type]
            
            if exercise_type == "pushup" and len(features) >= 6:
                elbow_angle_mean = features[0]
                body_alignment = features[2]
                
                if elbow_angle_mean < params['elbow_angle_min']:
                    feedback.append("Elbows too close to body - spread them wider")
                elif elbow_angle_mean > params['elbow_angle_max']:
                    feedback.append("Not going deep enough - lower your chest more")
                
                if body_alignment < params['body_alignment_min']:
                    feedback.append("Keep your body straight - avoid sagging hips")
            
            elif exercise_type == "squat" and len(features) >= 6:
                knee_angle_mean = features[0]
                knee_tracking_error = features[4]
                
                if knee_angle_mean > params['knee_angle_max']:
                    feedback.append("Squat deeper - aim for thighs parallel to ground")
                
                if knee_tracking_error > params['knee_tracking_threshold']:
                    feedback.append("Keep knees aligned over ankles - avoid knee valgus")
        
        return feedback if feedback else ["Form looks good! Keep it up!"]

    def save_models(self, filepath="exercise_models.pkl"):
        """Save trained models to file"""
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'label_encoders': self.label_encoders,
            'exercise_params': self.exercise_params
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Models saved to {filepath}")

    def load_models(self, filepath="exercise_models.pkl"):
        """Load trained models from file"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.label_encoders = model_data['label_encoders']
            self.exercise_params = model_data.get('exercise_params', self.exercise_params)
            
            print(f"Models loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

    def visualize_results(self, results_data, save_path="form_analysis_results.png"):
        """Create visualizations of form analysis results"""
        if not results_data:
            print("No data to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Exercise Form Analysis Results', fontsize=16)
        
        # Extract data for visualization
        exercises = [r.get('exercise', 'unknown') for r in results_data]
        qualities = [r.get('quality', 'unknown') for r in results_data]
        scores = [r.get('score', 0) for r in results_data]
        confidences = [r.get('confidence', 0) for r in results_data]
        
        # 1. Exercise distribution
        exercise_counts = pd.Series(exercises).value_counts()
        axes[0, 0].pie(exercise_counts.values, labels=exercise_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Exercise Distribution')
        
        # 2. Quality distribution
        quality_counts = pd.Series(qualities).value_counts()
        axes[0, 1].bar(quality_counts.index, quality_counts.values, 
                       color=['green' if q == 'good' else 'red' for q in quality_counts.index])
        axes[0, 1].set_title('Form Quality Distribution')
        axes[0, 1].set_ylabel('Count')
        
        # 3. Quality scores by exercise
        df_results = pd.DataFrame({'exercise': exercises, 'score': scores, 'quality': qualities})
        for exercise in df_results['exercise'].unique():
            exercise_data = df_results[df_results['exercise'] == exercise]
            axes[1, 0].scatter(exercise_data['exercise'], exercise_data['score'], 
                              c=['green' if q == 'good' else 'red' for q in exercise_data['quality']],
                              alpha=0.6)
        axes[1, 0].set_title('Quality Scores by Exercise')
        axes[1, 0].set_ylabel('Quality Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Confidence distribution
        axes[1, 1].hist(confidences, bins=20, color='blue', alpha=0.7)
        axes[1, 1].set_title('Prediction Confidence Distribution')
        axes[1, 1].set_xlabel('Confidence')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Results visualization saved to {save_path}")


def main():
    """Main execution function with complete pipeline"""
    print("=== Exercise Form Quality Classifier ===")
    print("Initializing system...")
    
    # Initialize components
    dataset_manager = ExerciseDatasetManager()
    classifier = ExerciseFormClassifier()
    
    print("\n1. Loading datasets...")
    training_data = dataset_manager.load_all_datasets()
    
    if not training_data:
        print("No training data available. Generating synthetic data...")
        training_data = dataset_manager.generate_synthetic_dataset(500)
    
    print(f"Loaded {len(training_data)} training samples")
    
    print("\n2. Training models...")
    classifier.train_models(training_data)
    
    print("\n3. Saving models...")
    classifier.save_models("exercise_form_models.pkl")
    
    print("\n4. Testing predictions...")
    # Test with sample data
    test_results = []
    for i, sample in enumerate(training_data[:20]):  # Test first 20 samples
        if len(sample['features']) > 0:
            result = classifier.predict_form_quality(sample['features'], sample['exercise'])
            result['actual_quality'] = sample['quality']
            test_results.append(result)
    
    print(f"\nTest Results Summary:")
    correct_predictions = sum(1 for r in test_results 
                             if r.get('quality') == r.get('actual_quality'))
    accuracy = correct_predictions / len(test_results) if test_results else 0
    print(f"Accuracy on test samples: {accuracy:.2%}")
    
    print("\n5. Generating visualizations...")
    classifier.visualize_results(test_results)
    
    # Example usage with video (if MediaPipe is available)
    if MP_AVAILABLE:
        print("\n6. Video analysis example:")
        print("To analyze a video file:")
        print("result = classifier.analyze_video('path/to/video.mp4', 'pushup')")
        print("print(result)")
    
    print("\n=== Setup Complete ===")
    print("Models are trained and ready for use!")
    print("\nUsage examples:")
    print("1. Load existing models: classifier.load_models('exercise_form_models.pkl')")
    print("2. Predict form quality: classifier.predict_form_quality(features, 'pushup')")
    print("3. Analyze video: classifier.analyze_video('video.mp4', 'pushup')")
    
    return classifier, dataset_manager


# Additional utility functions
def create_demo_data():
    """Create demo data for testing without external datasets"""
    demo_data = []
    exercises = ['pushup', 'squat', 'situp', 'plank']
    
    for exercise in exercises:
        for quality in ['good', 'poor']:
            for _ in range(50):  # 50 samples per exercise-quality combination
                # Generate realistic features based on exercise
                if exercise == 'pushup':
                    features = np.array([
                        np.random.normal(90 if quality == 'good' else 60, 10),  # elbow_angle_mean
                        np.random.normal(15, 5),  # elbow_angle_std
                        np.random.normal(175 if quality == 'good' else 145, 10),  # body_alignment
                        np.random.normal(8, 3),   # body_alignment_std
                        np.random.normal(25 if quality == 'good' else 15, 5),   # range_of_motion
                        np.random.normal(2.0, 0.3)  # rep_duration
                    ])
                elif exercise == 'squat':
                    features = np.array([
                        np.random.normal(85 if quality == 'good' else 110, 10),  # knee_angle_mean
                        np.random.normal(15, 5),  # knee_angle_std
                        np.random.normal(95 if quality == 'good' else 75, 8),   # back_angle_mean
                        np.random.normal(10, 3),  # back_angle_std
                        np.random.normal(5 if quality == 'good' else 25, 5),    # knee_tracking_error
                        np.random.normal(2.5, 0.5)  # rep_duration
                    ])
                elif exercise == 'situp':
                    features = np.array([
                        np.random.normal(45 if quality == 'good' else 25, 10),  # torso_angle_mean
                        np.random.normal(12, 4),  # torso_angle_std
                        np.random.normal(15 if quality == 'good' else 35, 8),   # neck_strain
                        np.random.normal(5 if quality == 'good' else 15, 5),    # hip_movement
                        np.random.normal(1.8, 0.2)  # rep_duration
                    ])
                elif exercise == 'plank':
                    features = np.array([
                        np.random.normal(175 if quality == 'good' else 155, 5),  # body_alignment_mean
                        np.random.normal(5 if quality == 'good' else 15, 3),     # body_alignment_std
                        np.random.normal(5 if quality == 'good' else 20, 5),     # hip_sag
                        np.random.normal(3 if quality == 'good' else 10, 2),     # stability_score
                        np.random.normal(30, 10)  # hold_duration
                    ])
                
                demo_data.append({
                    'exercise': exercise,
                    'features': features,
                    'quality': quality,
                    'quality_score': 8.5 if quality == 'good' else 4.2
                })
    
    return demo_data


def run_quick_demo():
    """Run a quick demonstration without external dependencies"""
    print("=== Quick Demo Mode ===")
    print("Running demonstration with synthetic data...")
    
    # Create classifier and demo data
    classifier = ExerciseFormClassifier()
    demo_data = create_demo_data()
    
    print(f"Generated {len(demo_data)} demo samples")
    
    # Train models
    classifier.train_models(demo_data)
    
    # Test predictions
    print("\nTesting predictions...")
    test_sample = demo_data[0]
    result = classifier.predict_form_quality(test_sample['features'], test_sample['exercise'])
    
    print(f"Test Exercise: {test_sample['exercise']}")
    print(f"Actual Quality: {test_sample['quality']}")
    print(f"Predicted Quality: {result['quality']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Quality Score: {result['score']:.1f}/10")
    
    if result.get('analysis'):
        print("Form Analysis:")
        for feedback in result['analysis']:
            print(f"  • {feedback}")
    
    return classifier


# Advanced features
class RealTimeFormAnalyzer:
    """Real-time form analysis using webcam"""
    
    def __init__(self, classifier):
        self.classifier = classifier
        self.pose_buffer = deque(maxlen=90)  # 3 seconds at 30 FPS
        if MP_AVAILABLE:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5
            )
            self.mp_draw = mp.solutions.drawing_utils

    def analyze_webcam(self, exercise_type='pushup'):
        """Real-time webcam analysis"""
        if not MP_AVAILABLE:
            print("MediaPipe not available for real-time analysis")
            return
        
        cap = cv2.VideoCapture(0)
        
        print(f"Starting real-time {exercise_type} analysis...")
        print("Press 'q' to quit, 'r' to reset buffer, 's' to analyze current buffer")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process pose
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # Draw pose landmarks
                self.mp_draw.draw_landmarks(
                    frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
                )
                
                # Extract keypoints
                keypoints = []
                for landmark in results.pose_landmarks.landmark:
                    keypoints.append([landmark.x, landmark.y, landmark.visibility])
                
                self.pose_buffer.append(keypoints)
                
                # Show buffer status
                buffer_text = f"Buffer: {len(self.pose_buffer)}/90 frames"
                cv2.putText(frame, buffer_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show exercise type
                cv2.putText(frame, f"Exercise: {exercise_type}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            cv2.imshow('Real-time Form Analysis', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.pose_buffer.clear()
                print("Buffer reset")
            elif key == ord('s') and len(self.pose_buffer) > 30:
                # Analyze current buffer
                result = self._analyze_buffer(exercise_type)
                print(f"Analysis Result: {result}")
        
        cap.release()
        cv2.destroyAllWindows()

    def _analyze_buffer(self, exercise_type):
        """Analyze current pose buffer"""
        if len(self.pose_buffer) < 30:
            return {"error": "Insufficient data in buffer"}
        
        buffer_list = list(self.pose_buffer)
        
        # Extract features using classifier's feature extractors
        if exercise_type in self.classifier.feature_extractors:
            features = self.classifier.feature_extractors[exercise_type](buffer_list)
        else:
            features = self.classifier._extract_generic_features(buffer_list)
        
        if len(features) == 0:
            return {"error": "Could not extract features"}
        
        # Predict form quality
        result = self.classifier.predict_form_quality(features, exercise_type)
        return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Exercise Form Quality Classifier')
    parser.add_argument('--mode', choices=['full', 'demo', 'realtime'], default='demo',
                       help='Run mode: full (with datasets), demo (synthetic), realtime (webcam)')
    parser.add_argument('--exercise', default='pushup',
                       help='Exercise type for real-time analysis')
    
    args = parser.parse_args()
    
    if args.mode == 'full':
        classifier, dataset_manager = main()
    elif args.mode == 'demo':
        classifier = run_quick_demo()
    elif args.mode == 'realtime':
        # First run demo to get trained classifier
        classifier = run_quick_demo()
        
        # Then start real-time analysis
        analyzer = RealTimeFormAnalyzer(classifier)
        analyzer.analyze_webcam(args.exercise)
    
    print("\nScript completed successfully!")