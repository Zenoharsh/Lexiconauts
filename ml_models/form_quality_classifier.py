# ==============================================================================
# MODEL 3: EXERCISE FORM QUALITY CLASSIFIER
# File: form_quality_classifier.py
# ==============================================================================

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pickle
import json
import os
import cv2
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns

class ExerciseFormClassifier:
    """Custom ML model for exercise form quality assessment"""
    
    def __init__(self):
        self.models = {}  # Store models for different exercises
        self.scalers = {}  # Store scalers for feature normalization
        self.feature_extractors = {
            'pushup': self._extract_pushup_features,
            'squat': self._extract_squat_features,
            'situp': self._extract_situp_features,
            'plank': self._extract_plank_features,
            'lunge': self._extract_lunge_features,
            'jump': self._extract_jump_features
        }
        
        # Exercise-specific thresholds and parameters
        self.exercise_params = {
            'pushup': {
                'elbow_angle_min': 70,    # Minimum acceptable elbow angle
                'elbow_angle_max': 110,   # Maximum elbow angle at bottom
                'body_alignment_min': 160,  # Minimum body straightness
                'rep_time_min': 1.0,      # Minimum time per rep (seconds)
                'rep_time_max': 5.0       # Maximum time per rep
            },
            'squat': {
                'knee_angle_min': 70,     # Minimum knee bend
                'knee_angle_max': 110,    # Maximum acceptable knee angle
                'back_angle_min': 70,     # Minimum back angle (leaning forward)
                'back_angle_max': 110,    # Maximum back angle
                'knee_tracking_threshold': 20  # Max knee deviation from ankle
            },
            'situp': {
                'torso_angle_min': 30,    # Minimum torso lift
                'torso_angle_max': 60,    # Maximum torso angle
                'neck_strain_threshold': 50,  # Max head-shoulder distance
                'hip_stability_threshold': 20  # Max hip movement
            }
        }
    
    def _calculate_angle_from_keypoints(self, keypoints, idx1, idx2, idx3):
        """Calculate angle between three keypoints with confidence check"""
        if any(keypoints[i][2] < 0.3 for i in [idx1, idx2, idx3]):
            return None  # Low confidence keypoints
        
        p1 = keypoints[idx1][:2]  # y, x
        p2 = keypoints[idx2][:2]
        p3 = keypoints[idx3][:2]
        
        # Convert to numpy arrays and swap coordinates for calculation
        a = np.array([p1[1], p1[0]])  # x, y
        b = np.array([p2[1], p2[0]])
        c = np.array([p3[1], p3[0]])
        
        # Calculate vectors
        ba = a - b
        bc = c - b
        
        # Avoid division by zero
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        
        if norm_ba == 0 or norm_bc == 0:
            return None
        
        # Calculate angle
        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Prevent numerical errors
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)
    
    def _calculate_distance(self, p1, p2):
        """Calculate normalized distance between two keypoints"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _extract_pushup_features(self, keypoints_sequence):
        """Extract comprehensive features for pushup form analysis"""
        features = []
        valid_frames = 0
        
        elbow_angles = []
        body_alignments = []
        wrist_positions = []
        hip_heights = []
        
        for keypoints in keypoints_sequence:
            frame_features = []
            
            # Primary angles
            left_elbow_angle = self._calculate_angle_from_keypoints(
                keypoints, 5, 7, 9)  # left_shoulder, left_elbow, left_wrist
            right_elbow_angle = self._calculate_angle_from_keypoints(
                keypoints, 6, 8, 10)  # right_shoulder, right_elbow, right_wrist
            
            if left_elbow_angle is None or right_elbow_angle is None:
                continue
            
            elbow_angles.extend([left_elbow_angle, right_elbow_angle])
            
            # Body alignment (shoulder-hip-ankle line)
            left_body_angle = self._calculate_angle_from_keypoints(
                keypoints, 5, 11, 15)  # left_shoulder, left_hip, left_ankle
            right_body_angle = self._calculate_angle_from_keypoints(
                keypoints, 6, 12, 16)  # right_shoulder, right_hip, right_ankle
            
            if left_body_angle and right_body_angle:
                body_alignments.extend([left_body_angle, right_body_angle])
                avg_body_alignment = (left_body_angle + right_body_angle) / 2
            else:
                avg_body_alignment = 180  # Default neutral
            
            # Hip position consistency
            hip_y = (keypoints[11][0] + keypoints[12][0]) / 2
            shoulder_y = (keypoints[5][0] + keypoints[6][0]) / 2
            hip_drop = abs(hip_y - shoulder_y)
            hip_heights.append(hip_drop)
            
            # Wrist position relative to shoulders (should be under shoulders)
            left_wrist_offset = abs(keypoints[9][1] - keypoints[5][1])  # x-axis deviation
            right_wrist_offset = abs(keypoints[10][1] - keypoints[6][1])
            wrist_positions.extend([left_wrist_offset, right_wrist_offset])
            
            # Symmetry checks
            elbow_symmetry = abs(left_elbow_angle - right_elbow_angle)
            
            frame_features = [
                left_elbow_angle, right_elbow_angle,
                avg_body_alignment, hip_drop,
                elbow_symmetry, left_wrist_offset, right_wrist_offset
            ]
            
            features.extend(frame_features)
            valid_frames += 1
        
        if valid_frames == 0:
            return np.array([])
        
        # Statistical features across the movement
        statistical_features = []
        
        if elbow_angles:
            statistical_features.extend([
                np.mean(elbow_angles), np.std(elbow_angles),
                np.min(elbow_angles), np.max(elbow_angles),
                np.ptp(elbow_angles)  # range
            ])
        else:
            statistical_features.extend([0, 0, 0, 0, 0])
        
        if body_alignments:
            statistical_features.extend([
                np.mean(body_alignments), np.std(body_alignments),
                np.mean([abs(180 - angle) for angle in body_alignments])  # deviation from straight
            ])
        else:
            statistical_features.extend([180, 0, 0])
        
        if hip_heights:
            statistical_features.extend([
                np.mean(hip_heights), np.std(hip_heights),
                np.max(hip_heights)  # max hip sag
            ])
        else:
            statistical_features.extend([0, 0, 0])
        
        # Movement quality indicators
        if len(elbow_angles) > 4:
            # Range of motion consistency
            rom_consistency = 1.0 - (np.std(elbow_angles) / (np.mean(elbow_angles) + 1e-6))
            statistical_features.append(rom_consistency)
            
            # Movement smoothness (how consistent is the angle change)
            angle_changes = np.diff(elbow_angles[:len(elbow_angles)//2])  # Take first half for analysis
            smoothness = 1.0 / (np.std(angle_changes) + 1e-6)
            statistical_features.append(min(smoothness, 10))  # Cap at reasonable value
        else:
            statistical_features.extend([0, 0])
        
        return np.array(features + statistical_features)
    
    def _extract_squat_features(self, keypoints_sequence):
        """Extract features for squat form analysis"""
        features = []
        valid_frames = 0
        
        knee_angles = []
        back_angles = []
        knee_tracking_errors = []
        
        for keypoints in keypoints_sequence:
            # Knee angles (both legs)
            left_knee_angle = self._calculate_angle_from_keypoints(
                keypoints, 11, 13, 15)  # left_hip, left_knee, left_ankle
            right_knee_angle = self._calculate_angle_from_keypoints(
                keypoints, 12, 14, 16)  # right_hip, right_knee, right_ankle
            
            if left_knee_angle is None or right_knee_angle is None:
                continue
            
            knee_angles.extend([left_knee_angle, right_knee_angle])
            
            # Back angle (torso alignment) - approximate using shoulder to hip
            torso_angle = self._calculate_angle_from_keypoints(
                keypoints, 5, 11, 13)  # shoulder, hip, knee (approximation)
            
            if torso_angle:
                back_angles.append(torso_angle)
            
            # Knee tracking (knees should track over toes)
            left_knee_x = keypoints[13][1]
            left_ankle_x = keypoints[15][1]
            right_knee_x = keypoints[14][1]
            right_ankle_x = keypoints[16][1]
            
            left_knee_track = abs(left_knee_x - left_ankle_x)
            right_knee_track = abs(right_knee_x - right_ankle_x)
            knee_tracking_errors.extend([left_knee_track, right_knee_track])
            
            # Hip width consistency (should remain stable)
            hip_width = abs(keypoints[11][1] - keypoints[12][1])
            
            # Ankle stability
            ankle_width = abs(keypoints[15][1] - keypoints[16][1])
            
            frame_features = [
                left_knee_angle, right_knee_angle, torso_angle or 90,
                left_knee_track, right_knee_track, hip_width, ankle_width,
                abs(left_knee_angle - right_knee_angle)  # knee symmetry
            ]
            
            features.extend(frame_features)
            valid_frames += 1
        
        if valid_frames == 0:
            return np.array([])
        
        # Statistical features
        statistical_features = []
        
        if knee_angles:
            statistical_features.extend([
                np.mean(knee_angles), np.std(knee_angles),
                np.min(knee_angles), np.max(knee_angles)
            ])
        else:
            statistical_features.extend([90, 0, 90, 90])
        
        if back_angles:
            statistical_features.extend([
                np.mean(back_angles), np.std(back_angles)
            ])
        else:
            statistical_features.extend([90, 0])
        
        if knee_tracking_errors:
            statistical_features.extend([
                np.mean(knee_tracking_errors), np.max(knee_tracking_errors)
            ])
        else:
            statistical_features.extend([0, 0])
        
        return np.array(features + statistical_features)
    
    def _extract_situp_features(self, keypoints_sequence):
        """Extract features for sit-up form analysis"""
        features = []
        valid_frames = 0
        
        torso_angles = []
        neck_strains = []
        hip_movements = []
        
        for keypoints in keypoints_sequence:
            # Torso angle (how much the person sits up)
            torso_angle = self._calculate_angle_from_keypoints(
                keypoints, 5, 11, 13)  # shoulder, hip, knee
            
            if torso_angle is None:
                continue
            
            torso_angles.append(torso_angle)
            
            # Neck strain check (head position relative to shoulders)
            nose_y = keypoints[0][0]
            shoulder_y = (keypoints[5][0] + keypoints[6][0]) / 2
            neck_strain = abs(nose_y - shoulder_y)
            neck_strains.append(neck_strain)
            
            # Hip stability (hips shouldn't move much)
            left_hip_y = keypoints[11][0]
            right_hip_y = keypoints[12][0]
            hip_stability = abs(left_hip_y - right_hip_y)
            hip_movements.append(hip_stability)
            
            # Knee position (should stay bent and stable)
            knee_angle = self._calculate_angle_from_keypoints(
                keypoints, 11, 13, 15) or 90
            
            frame_features = [torso_angle, neck_strain, hip_stability, knee_angle]
            features.extend(frame_features)
            valid_frames += 1
        
        if valid_frames == 0:
            return np.array([])
        
        # Statistical features
        statistical_features = []
        
        if torso_angles:
            statistical_features.extend([
                np.mean(torso_angles), np.std(torso_angles),
                np.min(torso_angles), np.max(torso_angles),
                np.ptp(torso_angles)  # range of motion
            ])
        else:
            statistical_features.extend([90, 0, 90, 90, 0])
        
        if neck_strains:
            statistical_features.extend([
                np.mean(neck_strains), np.max(neck_strains)  # avg and max neck strain
            ])
        else:
            statistical_features.extend([0, 0])
        
        if hip_movements:
            statistical_features.extend([
                np.mean(hip_movements), np.std(hip_movements)  # hip stability metrics
            ])
        else:
            statistical_features.extend([0, 0])
        
        return np.array(features + statistical_features)
    
    def _extract_plank_features(self, keypoints_sequence):
        """Extract features for plank form analysis"""
        features = []
        valid_frames = 0
        
        body_angles = []
        hip_sags = []
        elbow_alignments = []
        
        for keypoints in keypoints_sequence:
            # Body alignment (straight line from head to ankles)
            body_angle = self._calculate_angle_from_keypoints(
                keypoints, 0, 11, 15)  # nose, hip, ankle
            
            if body_angle is None:
                continue
            
            body_angles.append(body_angle)
            
            # Hip height consistency (shouldn't sag or pike)
            hip_y = (keypoints[11][0] + keypoints[12][0]) / 2
            shoulder_y = (keypoints[5][0] + keypoints[6][0]) / 2
            ankle_y = (keypoints[15][0] + keypoints[16][0]) / 2
            
            # Check if hip is in line with shoulder and ankle
            expected_hip_y = (shoulder_y + ankle_y) / 2
            hip_sag = abs(hip_y - expected_hip_y)
            hip_sags.append(hip_sag)
            
            # Elbow position (should be under shoulders for proper form)
            left_elbow_x = keypoints[7][1]
            left_shoulder_x = keypoints[5][1]
            right_elbow_x = keypoints[8][1]
            right_shoulder_x = keypoints[6][1]
            
            elbow_alignment = (abs(left_elbow_x - left_shoulder_x) + 
                             abs(right_elbow_x - right_shoulder_x)) / 2
            elbow_alignments.append(elbow_alignment)
            
            frame_features = [body_angle, hip_sag, elbow_alignment]
            features.extend(frame_features)
            valid_frames += 1
        
        if valid_frames == 0:
            return np.array([])
        
        # Statistical features for plank (stability is key)
        statistical_features = []
        
        if body_angles:
            # For plank, consistency is more important than range
            statistical_features.extend([
                np.mean(body_angles), np.std(body_angles),  # stability metrics
                abs(np.mean(body_angles) - 180)  # deviation from straight line
            ])
        else:
            statistical_features.extend([180, 0, 0])
        
        if hip_sags:
            statistical_features.extend([
                np.mean(hip_sags), np.max(hip_sags), np.std(hip_sags)
            ])
        else:
            statistical_features.extend([0, 0, 0])
        
        if elbow_alignments:
            statistical_features.extend([
                np.mean(elbow_alignments), np.std(elbow_alignments)
            ])
        else:
            statistical_features.extend([0, 0])
        
        return np.array(features + statistical_features)
    
    def _extract_lunge_features(self, keypoints_sequence):
        """Extract features for lunge form analysis"""
        features = []
        valid_frames = 0
        
        front_knee_angles = []
        back_knee_angles = []
        torso_angles = []
        
        for keypoints in keypoints_sequence:
            # Assume left leg is front leg (can be adjusted based on detection)
            front_knee_angle = self._calculate_angle_from_keypoints(
                keypoints, 11, 13, 15)  # left_hip, left_knee, left_ankle
            back_knee_angle = self._calculate_angle_from_keypoints(
                keypoints, 12, 14, 16)  # right_hip, right_knee, right_ankle
            
            if front_knee_angle is None or back_knee_angle is None:
                continue
            
            front_knee_angles.append(front_knee_angle)
            back_knee_angles.append(back_knee_angle)
            
            # Torso should remain upright
            torso_angle = self._calculate_angle_from_keypoints(
                keypoints, 5, 11, 13)  # shoulder, hip, knee
            if torso_angle:
                torso_angles.append(torso_angle)
            
            # Front knee should track over ankle
            front_knee_x = keypoints[13][1]
            front_ankle_x = keypoints[15][1]
            knee_tracking = abs(front_knee_x - front_ankle_x)
            
            frame_features = [front_knee_angle, back_knee_angle, torso_angle or 90, knee_tracking]
            features.extend(frame_features)
            valid_frames += 1
        
        if valid_frames == 0:
            return np.array([])
        
        # Statistical features
        statistical_features = []
        
        if front_knee_angles:
            statistical_features.extend([
                np.mean(front_knee_angles), np.min(front_knee_angles)
            ])
        else:
            statistical_features.extend([90, 90])
        
        if back_knee_angles:
            statistical_features.extend([
                np.mean(back_knee_angles), np.min(back_knee_angles)
            ])
        else:
            statistical_features.extend([90, 90])
        
        if torso_angles:
            statistical_features.extend([
                np.mean(torso_angles), np.std(torso_angles)
            ])
        else:
            statistical_features.extend([90, 0])
        
        return np.array(features + statistical_features)
    
    def _extract_jump_features(self, keypoints_sequence):
        """Extract features for jumping form analysis"""
        features = []
        valid_frames = 0
        
        knee_angles = []
        takeoff_angles = []
        landing_angles = []
        body_alignments = []
        
        # Identify jump phases (preparation, takeoff, flight, landing)
        hip_heights = []
        for keypoints in keypoints_sequence:
            hip_y = (keypoints[11][0] + keypoints[12][0]) / 2
            hip_heights.append(hip_y)
        
        if len(hip_heights) < 10:
            return np.array([])
        
        # Find takeoff and landing points
        hip_heights = np.array(hip_heights)
        min_idx = np.argmin(hip_heights)  # Highest point (lowest y-coordinate)
        
        # Analyze each phase
        preparation_end = max(0, min_idx - 5)
        landing_start = min(len(hip_heights) - 1, min_idx + 5)
        
        for i, keypoints in enumerate(keypoints_sequence):
            knee_angle = self._calculate_angle_from_keypoints(
                keypoints, 11, 13, 15)  # left_hip, left_knee, left_ankle
            
            if knee_angle is None:
                continue
            
            knee_angles.append(knee_angle)
            
            # Body alignment during jump
            body_angle = self._calculate_angle_from_keypoints(
                keypoints, 5, 11, 15)  # shoulder, hip, ankle
            if body_angle:
                body_alignments.append(body_angle)
                
                # Categorize by jump phase
                if i <= preparation_end:
                    takeoff_angles.append(knee_angle)
                elif i >= landing_start:
                    landing_angles.append(knee_angle)
            
            frame_features = [knee_angle, body_angle or 180]
            features.extend(frame_features)
            valid_frames += 1
        
        if valid_frames == 0:
            return np.array([])
        
        # Statistical features
        statistical_features = []
        
        if knee_angles:
            statistical_features.extend([
                np.mean(knee_angles), np.std(knee_angles),
                np.min(knee_angles)  # deepest squat
            ])
        else:
            statistical_features.extend([90, 0, 90])
        
        if takeoff_angles:
            statistical_features.append(np.mean(takeoff_angles))
        else:
            statistical_features.append(90)
        
        if landing_angles:
            statistical_features.append(np.mean(landing_angles))
        else:
            statistical_features.append(90)
        
        if body_alignments:
            statistical_features.extend([
                np.mean(body_alignments), 
                abs(np.mean(body_alignments) - 180)  # deviation from straight
            ])
        else:
            statistical_features.extend([180, 0])
        
        return np.array(features + statistical_features)
    
    def generate_synthetic_training_data(self, exercise_type, num_samples=1000):
        """Generate synthetic training data for initial model training"""
        np.random.seed(42)
        
        if exercise_type not in self.feature_extractors:
            raise ValueError(f"No feature extractor for {exercise_type}")
        
        # Get expected feature dimensions by creating a dummy sequence
        dummy_keypoints = np.random.rand(10, 17, 3)  # 10 frames, 17 keypoints, 3 values each
        dummy_keypoints[:, :, 2] = 0.8  # Set confidence scores
        
        try:
            dummy_features = self.feature_extractors[exercise_type](dummy_keypoints)
            feature_dim = len(dummy_features)
        except:
            feature_dim = 50  # Fallback dimension
        
        print(f"Generating {num_samples} synthetic samples for {exercise_type}")
        print(f"Feature dimension: {feature_dim}")
        
        X, y = [], []
        
        # Generate correct form samples
        for _ in range(num_samples // 2):
            if exercise_type == 'pushup':
                features = self._generate_good_pushup_features(feature_dim)
            elif exercise_type == 'squat':
                features = self._generate_good_squat_features(feature_dim)
            elif exercise_type == 'situp':
                features = self._generate_good_situp_features(feature_dim)
            else:
                features = np.random.normal(0, 1, feature_dim)
            
            X.append(features)
            y.append(1)  # Correct form
        
        # Generate incorrect form samples
        for _ in range(num_samples // 2):
            if exercise_type == 'pushup':
                features = self._generate_bad_pushup_features(feature_dim)
            elif exercise_type == 'squat':
                features = self._generate_bad_squat_features(feature_dim)
            elif exercise_type == 'situp':
                features = self._generate_bad_situp_features(feature_dim)
            else:
                features = np.random.normal(0, 2, feature_dim)  # Higher variance for bad form
            
            X.append(features)
            y.append(0)  # Incorrect form
        
        return np.array(X), np.array(y)
    
    def _generate_good_pushup_features(self, feature_dim):
        """Generate features representing good pushup form"""
        features = np.zeros(feature_dim)
        
        # Good elbow angles (90-120 degrees at bottom)
        if feature_dim > 10:
            features[0] = np.random.normal(100, 8)  # Left elbow angle
            features[1] = np.random.normal(100, 8)  # Right elbow angle
            features[2] = np.random.normal(175, 5)  # Body alignment (near straight)
            features[3] = np.random.normal(5, 2)    # Small hip drop
            features[4] = np.random.normal(3, 1)    # Good elbow symmetry
            
            # Statistical features (good range of motion, consistency)
            if feature_dim > 20:
                features[10] = np.random.normal(105, 5)  # Mean elbow angle
                features[11] = np.random.normal(15, 3)   # Std of elbow angles
                features[12] = np.random.normal(85, 5)   # Min elbow angle
                features[13] = np.random.normal(160, 5)  # Max elbow angle
                features[14] = np.random.normal(75, 10)  # Range
        
        # Fill remaining dimensions with random noise
        if feature_dim > len(features):
            remaining = feature_dim - len(features[features != 0])
            features[-remaining:] = np.random.normal(0, 0.4, remaining)
        
        return features
    
    def _generate_bad_situp_features(self, feature_dim):
        """Generate features representing poor sit-up form"""
        features = np.zeros(feature_dim)
        
        if feature_dim > 8:
            issue = np.random.choice(['neck_strain', 'partial_range', 'hip_movement'])
            
            if issue == 'neck_strain':
                features[1] = np.random.normal(25, 8)   # High neck strain
            elif issue == 'partial_range':
                features[0] = np.random.normal(20, 5)   # Doesn't sit up enough
            else:  # hip_movement
                features[2] = np.random.normal(15, 5)   # Hips moving too much
        
        if feature_dim > len(features):
            remaining = feature_dim - len(features[features != 0])
            features[-remaining:] = np.random.normal(0, 1.0, remaining)
        
        return features
    
    def train_classifier(self, exercise_type, training_data=None, labels=None, use_synthetic=True):
        """Train form classifier for specific exercise
        
        Args:
            exercise_type: Type of exercise ('pushup', 'squat', etc.)
            training_data: List of keypoints sequences (optional if using synthetic)
            labels: List of labels (0=incorrect, 1=correct)
            use_synthetic: Whether to use synthetic data if real data is insufficient
        """
        print(f"Training {exercise_type} form classifier...")
        
        if exercise_type not in self.feature_extractors:
            raise ValueError(f"No feature extractor for {exercise_type}")
        
        X = []
        y = []
        
        # Use real data if provided
        if training_data is not None and labels is not None:
            feature_extractor = self.feature_extractors[exercise_type]
            
            print(f"Processing {len(training_data)} real training samples...")
            for keypoints_seq, label in zip(training_data, labels):
                try:
                    features = feature_extractor(keypoints_seq)
                    if len(features) > 0:  # Valid features extracted
                        X.append(features)
                        y.append(label)
                except Exception as e:
                    print(f"Error extracting features: {e}")
                    continue
            
            print(f"Successfully processed {len(X)} samples from real data")
        
        # Add synthetic data if needed
        if len(X) < 100 or use_synthetic:
            print("Adding synthetic training data...")
            X_synthetic, y_synthetic = self.generate_synthetic_training_data(
                exercise_type, num_samples=max(500, 1000 - len(X)))
            
            if len(X) > 0:
                # Ensure feature dimensions match
                real_dim = len(X[0])
                synthetic_dim = len(X_synthetic[0])
                
                if real_dim != synthetic_dim:
                    if real_dim < synthetic_dim:
                        # Pad real features
                        X = [np.pad(features, (0, synthetic_dim - real_dim)) for features in X]
                    else:
                        # Truncate synthetic features
                        X_synthetic = X_synthetic[:, :real_dim]
                
                X.extend(X_synthetic)
                y.extend(y_synthetic)
            else:
                X = X_synthetic.tolist()
                y = y_synthetic.tolist()
        
        if len(X) == 0:
            raise ValueError("No valid training samples available")
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Handle variable feature lengths by padding/truncating
        max_length = max(len(features) if hasattr(features, '__len__') else 0 for features in X)
        X_processed = []
        
        for features in X:
            if len(features) < max_length:
                padded = np.pad(features, (0, max_length - len(features)))
            else:
                padded = features[:max_length]
            X_processed.append(padded)
        
        X = np.array(X_processed)
        
        print(f"Final training set: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Feature scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train Random Forest classifier
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        clf.fit(X_train, y_train)
        
        # Evaluate
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Training completed! Test accuracy: {accuracy:.3f}")
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=['Incorrect Form', 'Correct Form']))
        
        # Cross-validation
        cv_scores = cross_val_score(clf, X_scaled, y, cv=5)
        print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Feature importance
        feature_importance = clf.feature_importances_
        top_features = np.argsort(feature_importance)[-10:][::-1]
        print(f"\nTop 10 most important features:")
        for i, idx in enumerate(top_features):
            print(f"{i+1}. Feature {idx}: {feature_importance[idx]:.3f}")
        
        # Store the trained model and scaler
        self.models[exercise_type] = {
            'classifier': clf,
            'scaler': scaler,
            'feature_length': max_length,
            'accuracy': accuracy,
            'cv_accuracy': cv_scores.mean(),
            'feature_importance': feature_importance
        }
        
        return clf
    
    def predict_form_quality(self, exercise_type, keypoints_sequence, return_detailed=False):
        """Predict form quality for a given exercise sequence
        
        Args:
            exercise_type: Type of exercise
            keypoints_sequence: Sequence of keypoints from video
            return_detailed: Whether to return detailed analysis
            
        Returns:
            dict: Contains prediction, confidence, and feedback
        """
        if exercise_type not in self.models:
            return {'error': f'No trained model for {exercise_type}'}
        
        model_info = self.models[exercise_type]
        classifier = model_info['classifier']
        scaler = model_info['scaler']
        expected_length = model_info['feature_length']
        
        try:
            # Extract features
            feature_extractor = self.feature_extractors[exercise_type]
            features = feature_extractor(keypoints_sequence)
            
            if len(features) == 0:
                return {'error': 'Could not extract features from keypoints'}
            
            # Handle feature length
            if len(features) < expected_length:
                features = np.pad(features, (0, expected_length - len(features)))
            else:
                features = features[:expected_length]
            
            # Scale features
            features_scaled = scaler.transform([features])
            
            # Predict
            prediction = classifier.predict(features_scaled)[0]
            probabilities = classifier.predict_proba(features_scaled)[0]
            confidence = max(probabilities)
            
            result = {
                'prediction': 'correct' if prediction == 1 else 'incorrect',
                'confidence': float(confidence),
                'probabilities': {
                    'incorrect': float(probabilities[0]),
                    'correct': float(probabilities[1])
                }
            }
            
            # Add specific feedback based on prediction and exercise type
            if prediction == 0:  # Incorrect form
                result['feedback'] = self._generate_specific_feedback(
                    exercise_type, features, keypoints_sequence)
            else:
                result['feedback'] = ['Great form! Keep it up!']
            
            # Add detailed analysis if requested
            if return_detailed:
                result['detailed_analysis'] = self._analyze_exercise_details(
                    exercise_type, keypoints_sequence)
            
            return result
            
        except Exception as e:
            return {'error': f'Prediction failed: {str(e)}'}
    
    def _generate_specific_feedback(self, exercise_type, features, keypoints_sequence):
        """Generate specific feedback based on detected form issues"""
        feedback = []
        params = self.exercise_params.get(exercise_type, {})
        
        if exercise_type == 'pushup':
            # Analyze specific pushup issues
            if len(features) > 5:
                avg_elbow_angle = (features[0] + features[1]) / 2 if len(features) > 1 else features[0]
                body_alignment = features[2] if len(features) > 2 else 180
                elbow_symmetry = features[4] if len(features) > 4 else 0
                
                if avg_elbow_angle > 130:
                    feedback.append("Go deeper - your arms should bend to about 90 degrees")
                elif avg_elbow_angle < 70:
                    feedback.append("Don't go too low - risk of shoulder injury")
                
                if body_alignment < 160:
                    feedback.append("Keep your body straight - avoid hip sagging")
                elif body_alignment > 200:
                    feedback.append("Don't pike your hips up - maintain a straight line")
                
                if elbow_symmetry > 15:
                    feedback.append("Keep both arms moving symmetrically")
        
        elif exercise_type == 'squat':
            if len(features) > 5:
                avg_knee_angle = (features[0] + features[1]) / 2 if len(features) > 1 else features[0]
                back_angle = features[2] if len(features) > 2 else 90
                knee_tracking = (features[3] + features[4]) / 2 if len(features) > 4 else 0
                
                if avg_knee_angle > 110:
                    feedback.append("Squat deeper - aim for thighs parallel to ground")
                
                if back_angle < 70:
                    feedback.append("Keep your chest up - avoid excessive forward lean")
                
                if knee_tracking > 15:
                    feedback.append("Keep knees tracking over toes - avoid knee cave")
        
        elif exercise_type == 'situp':
            if len(features) > 3:
                torso_angle = features[0] if len(features) > 0 else 90
                neck_strain = features[1] if len(features) > 1 else 0
                
                if torso_angle < 30:
                    feedback.append("Sit up higher - bring chest towards knees")
                
                if neck_strain > 20:
                    feedback.append("Avoid pulling on your neck - focus on core muscles")
        
        if not feedback:
            feedback.append("Focus on maintaining consistent form throughout the movement")
        
        return feedback
    
    def _analyze_exercise_details(self, exercise_type, keypoints_sequence):
        """Provide detailed biomechanical analysis"""
        analysis = {
            'total_frames': len(keypoints_sequence),
            'exercise_type': exercise_type,
            'quality_indicators': {}
        }
        
        # Calculate frame-by-frame quality metrics
        frame_qualities = []
        
        for i, keypoints in enumerate(keypoints_sequence):
            frame_quality = self._assess_frame_quality(exercise_type, keypoints)
            frame_qualities.append(frame_quality)
        
        # Overall quality metrics
        if frame_qualities:
            analysis['quality_indicators'] = {
                'consistency': np.std([fq.get('overall_score', 0.5) for fq in frame_qualities]),
                'average_quality': np.mean([fq.get('overall_score', 0.5) for fq in frame_qualities]),
                'worst_frames': sorted(range(len(frame_qualities)), 
                                     key=lambda i: frame_qualities[i].get('overall_score', 0.5))[:3],
                'best_frames': sorted(range(len(frame_qualities)), 
                                    key=lambda i: frame_qualities[i].get('overall_score', 0.5), 
                                    reverse=True)[:3]
            }
        
        return analysis
    
    def _assess_frame_quality(self, exercise_type, keypoints):
        """Assess quality of form in a single frame"""
        quality = {'overall_score': 0.5}  # Default neutral score
        
        try:
            if exercise_type == 'pushup':
                # Check elbow angles
                left_elbow = self._calculate_angle_from_keypoints(keypoints, 5, 7, 9)
                right_elbow = self._calculate_angle_from_keypoints(keypoints, 6, 8, 10)
                
                if left_elbow and right_elbow:
                    avg_elbow = (left_elbow + right_elbow) / 2
                    
                    # Score based on elbow angle (90-120 is ideal)
                    if 90 <= avg_elbow <= 120:
                        elbow_score = 1.0
                    elif 70 <= avg_elbow <= 140:
                        elbow_score = 0.7
                    else:
                        elbow_score = 0.3
                    
                    quality['elbow_score'] = elbow_score
                    quality['elbow_angle'] = avg_elbow
                    quality['overall_score'] = elbow_score * 0.7  # Weight elbow angle heavily
                
                # Check body alignment
                body_angle = self._calculate_angle_from_keypoints(keypoints, 5, 11, 15)
                if body_angle:
                    alignment_score = max(0, 1 - abs(body_angle - 180) / 30)  # Closer to 180 is better
                    quality['alignment_score'] = alignment_score
                    quality['overall_score'] = quality['overall_score'] * 0.7 + alignment_score * 0.3
        
        except Exception:
            pass  # Return default scores if calculation fails
        
        return quality
    
    def batch_evaluate(self, exercise_type, video_sequences_with_labels):
        """Evaluate model performance on a batch of videos
        
        Args:
            exercise_type: Type of exercise
            video_sequences_with_labels: List of (keypoints_sequence, true_label) tuples
        """
        if exercise_type not in self.models:
            print(f"No trained model for {exercise_type}")
            return
        
        predictions = []
        true_labels = []
        confidences = []
        
        print(f"Evaluating {len(video_sequences_with_labels)} videos...")
        
        for i, (keypoints_seq, true_label) in enumerate(video_sequences_with_labels):
            result = self.predict_form_quality(exercise_type, keypoints_seq)
            
            if 'error' not in result:
                predicted_label = 1 if result['prediction'] == 'correct' else 0
                predictions.append(predicted_label)
                true_labels.append(true_label)
                confidences.append(result['confidence'])
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(video_sequences_with_labels)} videos")
        
        if len(predictions) == 0:
            print("No valid predictions made")
            return
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        
        print(f"\nBatch Evaluation Results for {exercise_type}:")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Average Confidence: {np.mean(confidences):.3f}")
        
        print("\nDetailed Classification Report:")
        print(classification_report(true_labels, predictions, 
                                  target_names=['Incorrect', 'Correct']))
        
        # Confusion Matrix
        cm = confusion_matrix(true_labels, predictions)
        print(f"\nConfusion Matrix:")
        print(f"True\\Pred  Incorrect  Correct")
        print(f"Incorrect      {cm[0,0]}        {cm[0,1]}")
        print(f"Correct        {cm[1,0]}        {cm[1,1]}")
        
        return {
            'accuracy': accuracy,
            'predictions': predictions,
            'true_labels': true_labels,
            'confidences': confidences,
            'confusion_matrix': cm.tolist()
        }
    
    def save_model(self, exercise_type, filepath):
        """Save trained model to file"""
        if exercise_type not in self.models:
            raise ValueError(f"No trained model for {exercise_type}")
        
        model_data = {
            'exercise_type': exercise_type,
            'model_info': self.models[exercise_type],
            'exercise_params': self.exercise_params.get(exercise_type, {})
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model for {exercise_type} saved to {filepath}")
    
    def load_model(self, exercise_type, filepath):
        """Load trained model from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models[exercise_type] = model_data['model_info']
        if 'exercise_params' in model_data:
            self.exercise_params[exercise_type] = model_data['exercise_params']
        
        print(f"Model for {exercise_type} loaded from {filepath}")
    
    def export_model_summary(self, exercise_type, output_path):
        """Export model summary and performance metrics"""
        if exercise_type not in self.models:
            print(f"No trained model for {exercise_type}")
            return
        
        model_info = self.models[exercise_type]
        
        summary = {
            'exercise_type': exercise_type,
            'model_type': 'RandomForestClassifier',
            'training_accuracy': model_info.get('accuracy', 'N/A'),
            'cross_validation_accuracy': model_info.get('cv_accuracy', 'N/A'),
            'feature_dimension': model_info.get('feature_length', 'N/A'),
            'model_parameters': {
                'n_estimators': model_info['classifier'].n_estimators,
                'max_depth': model_info['classifier'].max_depth,
                'min_samples_split': model_info['classifier'].min_samples_split
            },
            'feature_importance': model_info.get('feature_importance', []).tolist(),
            'exercise_parameters': self.exercise_params.get(exercise_type, {})
        }
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Model summary exported to {output_path}")

# ==============================================================================
# DEMO AND TESTING FUNCTIONS
# ==============================================================================

def demo_synthetic_training():
    """Demo training with synthetic data"""
    print("=== SYNTHETIC DATA TRAINING DEMO ===")
    
    classifier = ExerciseFormClassifier()
    
    exercises = ['pushup', 'squat', 'situp']
    
    for exercise in exercises:
        print(f"\nTraining {exercise} classifier with synthetic data...")
        try:
            clf = classifier.train_classifier(exercise, use_synthetic=True)
            
            # Test with some synthetic samples
            X_test, y_test = classifier.generate_synthetic_training_data(exercise, 100)
            
            # Predict on test samples
            test_results = []
            for features, true_label in zip(X_test, y_test):
                # Convert features back to keypoints format for prediction
                # This is a simplified approach for demo
                dummy_keypoints = np.random.rand(10, 17, 3)
                dummy_keypoints[:, :, 2] = 0.8  # Set confidence
                
                result = classifier.predict_form_quality(exercise, dummy_keypoints)
                if 'error' not in result:
                    predicted = 1 if result['prediction'] == 'correct' else 0
                    test_results.append((predicted, true_label, result['confidence']))
            
            if test_results:
                predictions, true_labels, confidences = zip(*test_results)
                accuracy = accuracy_score(true_labels, predictions)
                print(f"Test accuracy on synthetic data: {accuracy:.3f}")
                print(f"Average confidence: {np.mean(confidences):.3f}")
            
        except Exception as e:
            print(f"Error training {exercise}: {e}")

def demo_menu():
    """Interactive demo menu for form classification"""
    classifier = ExerciseFormClassifier()
    
    while True:
        print("\n" + "="*50)
        print("EXERCISE FORM QUALITY CLASSIFIER DEMO")
        print("="*50)
        print("1. Train classifier with synthetic data")
        print("2. Test prediction with synthetic sample")
        print("3. Save trained model")
        print("4. Load trained model")
        print("5. Export model summary")
        print("6. Batch evaluate (synthetic)")
        print("7. Compare exercise classifiers")
        print("8. Exit")
        
        choice = input("\nSelect option (1-8): ")
        
        if choice == '1':
            exercise = input("Enter exercise type (pushup/squat/situp/plank/lunge/jump): ").lower()
            if exercise in classifier.feature_extractors:
                try:
                    classifier.train_classifier(exercise, use_synthetic=True)
                    print(f"Training completed for {exercise}")
                except Exception as e:
                    print(f"Training failed: {e}")
            else:
                print("Invalid exercise type")
        
        elif choice == '2':
            exercise = input("Enter exercise type: ").lower()
            if exercise in classifier.models:
                # Generate a test sample
                X_test, y_test = classifier.generate_synthetic_training_data(exercise, 2)
                
                # Create dummy keypoints for prediction
                dummy_keypoints = np.random.rand(15, 17, 3)
                dummy_keypoints[:, :, 2] = 0.8
                
                result = classifier.predict_form_quality(exercise, dummy_keypoints, return_detailed=True)
                
                print(f"\nPrediction Result:")
                print(f"Form Quality: {result.get('prediction', 'error')}")
                print(f"Confidence: {result.get('confidence', 0):.3f}")
                if 'feedback' in result:
                    print(f"Feedback: {result['feedback']}")
                
            else:
                print(f"No trained model for {exercise}")
        
        elif choice == '3':
            exercise = input("Enter exercise type to save: ").lower()
            if exercise in classifier.models:
                filepath = input("Enter save path: ")
                classifier.save_model(exercise, filepath)
            else:
                print(f"No trained model for {exercise}")
        
        elif choice == '4':
            exercise = input("Enter exercise type: ").lower()
            filepath = input("Enter model file path: ")
            try:
                classifier.load_model(exercise, filepath)
                print(f"Model loaded successfully for {exercise}")
            except Exception as e:
                print(f"Failed to load model: {e}")
        
        elif choice == '5':
            exercise = input("Enter exercise type: ").lower()
            if exercise in classifier.models:
                output_path = input("Enter output path for summary: ")
                classifier.export_model_summary(exercise, output_path)
            else:
                print(f"No trained model for {exercise}")
        
        elif choice == '6':
            exercise = input("Enter exercise type: ").lower()
            if exercise in classifier.models:
                # Generate synthetic test data
                X_test, y_test = classifier.generate_synthetic_training_data(exercise, 50)
                
                # Convert to keypoints format (simplified)
                test_sequences = []
                for features, label in zip(X_test, y_test):
                    dummy_keypoints = np.random.rand(12, 17, 3)
                    dummy_keypoints[:, :, 2] = 0.8
                    test_sequences.append((dummy_keypoints, label))
                
                results = classifier.batch_evaluate(exercise, test_sequences)
                
            else:
                print(f"No trained model for {exercise}")
        
        elif choice == '7':
            # Train and compare multiple exercise classifiers
            exercises = ['pushup', 'squat', 'situp']
            results = {}
            
            print("Training and comparing classifiers...")
            for exercise in exercises:
                try:
                    classifier.train_classifier(exercise, use_synthetic=True)
                    model_info = classifier.models[exercise]
                    results[exercise] = {
                        'accuracy': model_info.get('accuracy', 0),
                        'cv_accuracy': model_info.get('cv_accuracy', 0)
                    }
                except Exception as e:
                    print(f"Failed to train {exercise}: {e}")
                    results[exercise] = {'accuracy': 0, 'cv_accuracy': 0}
            
            # Display comparison
            print(f"\n{'Exercise':<10} {'Accuracy':<10} {'CV Accuracy':<12}")
            print("-" * 35)
            for exercise, metrics in results.items():
                print(f"{exercise:<10} {metrics['accuracy']:<10.3f} {metrics['cv_accuracy']:<12.3f}")
        
        elif choice == '8':
            break
        
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    # Check dependencies
    try:
        import sklearn
        import numpy as np
        import pandas as pd
        print("✓ All dependencies available")
        print(f"Scikit-learn version: {sklearn.__version__}")
        print(f"NumPy version: {np.__version__}")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install: pip install scikit-learn pandas numpy matplotlib seaborn")
        exit(1)
    
# Run demo
demo_menu()
features[-remaining:] = np.random.normal(0, 0.5, remaining)
        
return features
def _generate_bad_pushup_features(self, feature_dim):
        """Generate features representing poor pushup form"""
        features = np.zeros(feature_dim)
        
        # Poor form characteristics
        if feature_dim > 10:
            # Random choice of what's wrong
            issue = np.random.choice(['partial_range', 'body_sag', 'asymmetric'])
            
            if issue == 'partial_range':
                features[0] = np.random.normal(140, 10)  # Doesn't go low enough
                features[1] = np.random.normal(140, 10)
                features[2] = np.random.normal(175, 5)   # Body still straight
            elif issue == 'body_sag':
                features[0] = np.random.normal(100, 8)   # Angles OK
                features[1] = np.random.normal(100, 8)
                features[2] = np.random.normal(150, 10)  # Body sagging
                features[3] = np.random.normal(20, 5)    # Large hip drop
            else:  # asymmetric
                features[0] = np.random.normal(90, 8)    # Left side different
                features[1] = np.random.normal(130, 8)   # Right side different
                features[4] = np.random.normal(40, 10)   # Poor symmetry
            
            # Poor statistical features
            if feature_dim > 20:
                features[11] = np.random.normal(30, 10)  # High variability
                features[14] = np.random.normal(40, 15)  # Poor range consistency
        
        # Fill remaining with higher variance noise
        if feature_dim > len(features):
            remaining = feature_dim - len(features[features != 0])
            features[-remaining:] = np.random.normal(0, 1.5, remaining)
        
        return features
    
    def _generate_good_squat_features(self, feature_dim):
        """Generate features representing good squat form"""
        features = np.zeros(feature_dim)
        
        if feature_dim > 10:
            features[0] = np.random.normal(80, 8)   # Good knee angle (deep squat)
            features[1] = np.random.normal(80, 8)   # Symmetric
            features[2] = np.random.normal(85, 5)   # Good back angle
            features[3] = np.random.normal(3, 2)    # Minimal knee tracking error
            features[4] = np.random.normal(3, 2)    # Good tracking both sides
            features[7] = np.random.normal(2, 1)    # Good symmetry
        
        if feature_dim > len(features):
            remaining = feature_dim - len(features[features != 0])
            features[-remaining:] = np.random.normal(0, 0.3, remaining)
        
        return features
    
    def _generate_bad_squat_features(self, feature_dim):
        """Generate features representing poor squat form"""
        features = np.zeros(feature_dim)
        
        if feature_dim > 10:
            issue = np.random.choice(['shallow', 'knee_cave', 'forward_lean'])
            
            if issue == 'shallow':
                features[0] = np.random.normal(130, 10)  # Doesn't squat deep enough
                features[1] = np.random.normal(130, 10)
            elif issue == 'knee_cave':
                features[3] = np.random.normal(25, 5)    # Poor knee tracking
                features[4] = np.random.normal(25, 5)
            else:  # forward_lean
                features[2] = np.random.normal(60, 8)    # Leaning too far forward
            
            features[7] = np.random.normal(15, 5)        # Poor symmetry
        
        if feature_dim > len(features):
            remaining = feature_dim - len(features[features != 0])
            features[-remaining:] = np.random.normal(0, 1.2, remaining)
        
        return features
    
    def _generate_good_situp_features(self, feature_dim):
        """Generate features representing good sit-up form"""
        features = np.zeros(feature_dim)
        
        if feature_dim > 8:
            features[0] = np.random.normal(45, 5)   # Good torso angle
            features[1] = np.random.normal(8, 3)    # Minimal neck strain
            features[2] = np.random.normal(3, 1)    # Stable hips
            features[3] = np.random.normal(90, 5)   # Knees stay bent
        
        if feature_dim > len(features):
            remaining = feature_dim - len