# ==============================================================================
# MODEL 1: MOVENET POSE ESTIMATION
# File: movenet_pose_estimator.py
# ==============================================================================

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from collections import deque
import time

class MoveNetPoseEstimator:
    """Advanced MoveNet implementation for sports assessment"""
    
    def __init__(self, model_type='thunder'):
        """Initialize MoveNet model
        
        Args:
            model_type: 'thunder' (accurate) or 'lightning' (fast)
        """
        self.model_type = model_type
        
        # Load appropriate model
        if model_type == 'thunder':
            self.model_url = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
            self.input_size = 256
        else:
            self.model_url = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
            self.input_size = 192
            
        print(f"Loading MoveNet {model_type} model...")
        self.movenet = hub.load(self.model_url)
        self.movenet_model = self.movenet.signatures['serving_default']
        print("MoveNet loaded successfully!")
        
        # Keypoint mapping (17 keypoints)
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Create keypoint index mapping
        self.keypoint_indices = {name: idx for idx, name in enumerate(self.keypoint_names)}
        
        # Keypoint connections for skeleton drawing
        self.connections = [
            ('nose', 'left_eye'), ('nose', 'right_eye'),
            ('left_eye', 'left_ear'), ('right_eye', 'right_ear'),
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
            ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'), ('right_knee', 'right_ankle')
        ]
        
        # Color scheme for visualization
        self.colors = {
            'keypoint': (255, 0, 0),    # Blue
            'connection': (0, 255, 0),  # Green
            'text': (255, 255, 255)     # White
        }
    
    def preprocess_frame(self, frame):
        """Preprocess frame for MoveNet inference"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize maintaining aspect ratio with padding
        frame_resized = tf.image.resize_with_pad(frame_rgb, self.input_size, self.input_size)
        
        # Convert to int32 as required by MoveNet
        frame_int32 = tf.cast(frame_resized, dtype=tf.int32)
        
        # Add batch dimension
        return tf.expand_dims(frame_int32, axis=0)
    
    def extract_keypoints(self, frame):
        """Extract keypoints from frame
        
        Returns:
            keypoints: Array of shape (17, 3) containing [y, x, confidence]
        """
        input_tensor = self.preprocess_frame(frame)
        outputs = self.movenet_model(input_tensor)
        keypoints = outputs['output_0'].numpy()[0, 0, :, :]
        return keypoints
    
    def get_keypoint_by_name(self, keypoints, name):
        """Get specific keypoint by name
        
        Args:
            keypoints: Keypoints array from extract_keypoints
            name: Name of keypoint (e.g., 'left_shoulder')
            
        Returns:
            tuple: (y, x, confidence) or None if invalid name
        """
        if name in self.keypoint_indices:
            idx = self.keypoint_indices[name]
            return keypoints[idx]
        return None
    
    def calculate_angle(self, p1, p2, p3):
        """Calculate angle between three keypoints
        
        Args:
            p1, p2, p3: Keypoint tuples (y, x, confidence)
            
        Returns:
            angle in degrees
        """
        # Convert to numpy arrays (swap x, y coordinates)
        a = np.array([p1[1], p1[0]])  # x, y
        b = np.array([p2[1], p2[0]]) 
        c = np.array([p3[1], p3[0]])
        
        # Calculate vectors
        ba = a - b
        bc = c - b
        
        # Calculate angle
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)
    
    def calculate_distance(self, p1, p2):
        """Calculate distance between two keypoints"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def draw_keypoints_only(self, frame, keypoints, confidence_threshold=0.3):
        """Draw only keypoints without connections"""
        height, width = frame.shape[:2]
        annotated_frame = frame.copy()
        
        for i, (y, x, confidence) in enumerate(keypoints):
            if confidence > confidence_threshold:
                x_px, y_px = int(x * width), int(y * height)
                cv2.circle(annotated_frame, (x_px, y_px), 5, self.colors['keypoint'], -1)
                cv2.putText(annotated_frame, str(i), (x_px + 5, y_px - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        return annotated_frame
    
    def draw_skeleton(self, frame, keypoints, confidence_threshold=0.3):
        """Draw complete skeleton with keypoints and connections"""
        height, width = frame.shape[:2]
        annotated_frame = frame.copy()
        
        # Draw connections first
        for connection in self.connections:
            kpt1_name, kpt2_name = connection
            kpt1_idx = self.keypoint_indices[kpt1_name]
            kpt2_idx = self.keypoint_indices[kpt2_name]
            
            y1, x1, c1 = keypoints[kpt1_idx]
            y2, x2, c2 = keypoints[kpt2_idx]
            
            if c1 > confidence_threshold and c2 > confidence_threshold:
                x1_px, y1_px = int(x1 * width), int(y1 * height)
                x2_px, y2_px = int(x2 * width), int(y2 * height)
                cv2.line(annotated_frame, (x1_px, y1_px), (x2_px, y2_px), 
                        self.colors['connection'], 2)
        
        # Draw keypoints
        for i, (y, x, confidence) in enumerate(keypoints):
            if confidence > confidence_threshold:
                x_px, y_px = int(x * width), int(y * height)
                cv2.circle(annotated_frame, (x_px, y_px), 5, self.colors['keypoint'], -1)
                cv2.putText(annotated_frame, self.keypoint_names[i], (x_px + 5, y_px - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
        
        return annotated_frame
    
    def analyze_frame_quality(self, keypoints, confidence_threshold=0.3):
        """Analyze the quality of pose detection in current frame"""
        visible_keypoints = np.sum(keypoints[:, 2] > confidence_threshold)
        avg_confidence = np.mean(keypoints[:, 2])
        
        # Check if key body parts are visible
        key_parts = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        key_visible = 0
        for part in key_parts:
            if keypoints[self.keypoint_indices[part], 2] > confidence_threshold:
                key_visible += 1
        
        quality_score = (visible_keypoints / 17) * 0.5 + avg_confidence * 0.3 + (key_visible / 4) * 0.2
        
        return {
            'quality_score': quality_score,
            'visible_keypoints': visible_keypoints,
            'avg_confidence': avg_confidence,
            'key_parts_visible': key_visible
        }
    
    def process_video_file(self, video_path, output_path=None, show_preview=True):
        """Process entire video file and extract keypoints sequence
        
        Returns:
            List of keypoints arrays for each frame
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
        
        # Setup output writer if requested
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        keypoints_sequence = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract keypoints
            keypoints = self.extract_keypoints(frame)
            keypoints_sequence.append(keypoints)
            
            # Annotate frame
            annotated_frame = self.draw_skeleton(frame, keypoints)
            
            # Add frame info
            cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show preview
            if show_preview:
                cv2.imshow('Processing Video', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Write output frame
            if writer:
                writer.write(annotated_frame)
            
            frame_count += 1
            
            # Progress update
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")
        
        cap.release()
        if writer:
            writer.release()
        if show_preview:
            cv2.destroyAllWindows()
        
        print(f"Processing complete! Extracted keypoints from {len(keypoints_sequence)} frames")
        return keypoints_sequence
    
    def real_time_analysis(self):
        """Real-time pose analysis with webcam"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Real-time pose analysis started")
        print("Press 'q' to quit, 's' to save screenshot, 'a' to toggle analysis")
        
        show_analysis = True
        fps_counter = deque(maxlen=30)
        
        while cap.isOpened():
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract keypoints
            keypoints = self.extract_keypoints(frame)
            
            # Draw skeleton
            annotated_frame = self.draw_skeleton(frame, keypoints)
            
            # Add analysis if enabled
            if show_analysis:
                quality = self.analyze_frame_quality(keypoints)
                
                # Display quality metrics
                cv2.putText(annotated_frame, f"Quality: {quality['quality_score']:.2f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Visible: {quality['visible_keypoints']}/17", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Avg Conf: {quality['avg_confidence']:.2f}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Calculate and display FPS
            end_time = time.time()
            fps = 1.0 / (end_time - start_time)
            fps_counter.append(fps)
            avg_fps = np.mean(fps_counter)
            
            cv2.putText(annotated_frame, f"FPS: {avg_fps:.1f}", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            cv2.imshow('MoveNet Real-time Analysis', annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_path = f"pose_screenshot_{int(time.time())}.jpg"
                cv2.imwrite(screenshot_path, annotated_frame)
                print(f"Screenshot saved: {screenshot_path}")
            elif key == ord('a'):
                show_analysis = not show_analysis
                print(f"Analysis display: {'ON' if show_analysis else 'OFF'}")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def export_to_tflite(self, output_path="movenet_model.tflite"):
        """Export model to TensorFlow Lite for mobile deployment"""
        try:
            # Create a concrete function
            concrete_func = self.movenet_model.get_concrete_function(
                tf.TensorSpec(shape=[1, self.input_size, self.input_size, 3], 
                             dtype=tf.int32)
            )
            
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            
            # Save model
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            print(f"TFLite model saved to: {output_path}")
            print(f"Model size: {len(tflite_model) / 1024 / 1024:.2f} MB")
            
            return output_path
            
        except Exception as e:
            print(f"TFLite export failed: {e}")
            return None
    
    def save_keypoints_to_json(self, keypoints_sequence, output_path):
        """Save keypoints sequence to JSON file"""
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        json_data = {
            'model_type': self.model_type,
            'keypoint_names': self.keypoint_names,
            'total_frames': len(keypoints_sequence),
            'keypoints_sequence': [keypoints.tolist() for keypoints in keypoints_sequence]
        }
        
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Keypoints saved to: {output_path}")

# ==============================================================================
# EXERCISE-SPECIFIC ANALYZERS USING MOVENET
# ==============================================================================

class PushupAnalyzer:
    """Specialized pushup analysis using MoveNet pose estimation"""
    
    def __init__(self):
        self.pose_estimator = MoveNetPoseEstimator('thunder')
        self.rep_count = 0
        self.is_down = False
        self.angle_threshold_down = 90
        self.angle_threshold_up = 160
        self.angle_history = deque(maxlen=10)
    
    def analyze_pushup_form(self, keypoints):
        """Analyze pushup form from keypoints"""
        # Get relevant keypoints
        left_shoulder = self.pose_estimator.get_keypoint_by_name(keypoints, 'left_shoulder')
        left_elbow = self.pose_estimator.get_keypoint_by_name(keypoints, 'left_elbow')
        left_wrist = self.pose_estimator.get_keypoint_by_name(keypoints, 'left_wrist')
        left_hip = self.pose_estimator.get_keypoint_by_name(keypoints, 'left_hip')
        left_ankle = self.pose_estimator.get_keypoint_by_name(keypoints, 'left_ankle')
        
        analysis = {}
        
        # Check if keypoints are confident enough
        if all(kp[2] > 0.3 for kp in [left_shoulder, left_elbow, left_wrist]):
            # Elbow angle
            elbow_angle = self.pose_estimator.calculate_angle(left_shoulder, left_elbow, left_wrist)
            analysis['elbow_angle'] = elbow_angle
            self.angle_history.append(elbow_angle)
            
            # Body alignment (should be straight line)
            if left_hip[2] > 0.3 and left_ankle[2] > 0.3:
                body_angle = self.pose_estimator.calculate_angle(left_shoulder, left_hip, left_ankle)
                analysis['body_alignment'] = body_angle
                
                # Check for sagging (body angle should be close to 180°)
                if body_angle < 160:
                    analysis['form_issue'] = 'hip_sag'
                elif body_angle > 200:
                    analysis['form_issue'] = 'hip_pike'
            
            # Rep counting logic
            if elbow_angle < self.angle_threshold_down and not self.is_down:
                self.is_down = True
                analysis['phase'] = 'down'
            elif elbow_angle > self.angle_threshold_up and self.is_down:
                self.is_down = False
                self.rep_count += 1
                analysis['phase'] = 'up'
                analysis['rep_completed'] = True
            else:
                analysis['phase'] = 'transition'
            
            # Form feedback
            if elbow_angle < 60:
                analysis['feedback'] = 'Too low - injury risk!'
            elif elbow_angle > 90 and self.is_down:
                analysis['feedback'] = 'Go lower'
            elif not self.is_down and elbow_angle < 160:
                analysis['feedback'] = 'Extend arms fully'
            else:
                analysis['feedback'] = 'Good form!'
            
            analysis['rep_count'] = self.rep_count
            analysis['angle_consistency'] = np.std(self.angle_history) if len(self.angle_history) > 1 else 0
        
        return analysis
    
    def real_time_pushup_counter(self):
        """Real-time pushup counting with form feedback"""
        cap = cv2.VideoCapture(0)
        print("Pushup Counter Started!")
        print("Position yourself for pushups. Press 'r' to reset, 'q' to quit")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            keypoints = self.pose_estimator.extract_keypoints(frame)
            analysis = self.analyze_pushup_form(keypoints)
            
            # Draw skeleton
            annotated_frame = self.pose_estimator.draw_skeleton(frame, keypoints)
            
            # Display analysis results
            y_offset = 30
            for key, value in analysis.items():
                if key in ['elbow_angle', 'rep_count', 'feedback']:
                    if key == 'elbow_angle':
                        text = f"Elbow Angle: {value:.1f}°"
                        color = (0, 255, 0)
                    elif key == 'rep_count':
                        text = f"Pushups: {value}"
                        color = (0, 0, 255)
                    elif key == 'feedback':
                        text = f"{value}"
                        color = (0, 255, 0) if 'Good' in value else (0, 165, 255)
                    
                    cv2.putText(annotated_frame, text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    y_offset += 40
            
            cv2.imshow('Pushup Counter', annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.rep_count = 0
                self.is_down = False
                self.angle_history.clear()
                print("Counter reset!")
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"Final pushup count: {self.rep_count}")

# ==============================================================================
# DEMO AND TESTING FUNCTIONS
# ==============================================================================

def test_pose_estimation():
    """Test basic pose estimation functionality"""
    print("Testing MoveNet Pose Estimation...")
    
    # Test with both models
    for model_type in ['lightning', 'thunder']:
        print(f"\nTesting {model_type} model:")
        pose_estimator = MoveNetPoseEstimator(model_type)
        
        # Test with webcam for a few seconds
        cap = cv2.VideoCapture(0)
        frame_times = []
        
        for i in range(30):  # Test 30 frames
            ret, frame = cap.read()
            if ret:
                start_time = time.time()
                keypoints = pose_estimator.extract_keypoints(frame)
                end_time = time.time()
                frame_times.append(end_time - start_time)
                
                if i == 0:  # Show first frame analysis
                    quality = pose_estimator.analyze_frame_quality(keypoints)
                    print(f"  Sample quality score: {quality['quality_score']:.3f}")
                    print(f"  Visible keypoints: {quality['visible_keypoints']}/17")
        
        cap.release()
        
        avg_time = np.mean(frame_times)
        print(f"  Average inference time: {avg_time*1000:.1f}ms")
        print(f"  Theoretical FPS: {1/avg_time:.1f}")

def demo_menu():
    """Interactive demo menu"""
    while True:
        print("\n" + "="*50)
        print("MOVENET POSE ESTIMATION DEMO")
        print("="*50)
        print("1. Real-time pose analysis (webcam)")
        print("2. Pushup counter (real-time)")
        print("3. Process video file")
        print("4. Test model performance")
        print("5. Export model to TFLite")
        print("6. Exit")
        
        choice = input("\nSelect option (1-6): ")
        
        if choice == '1':
            pose_estimator = MoveNetPoseEstimator('thunder')
            pose_estimator.real_time_analysis()
        
        elif choice == '2':
            pushup_analyzer = PushupAnalyzer()
            pushup_analyzer.real_time_pushup_counter()
        
        elif choice == '3':
            video_path = input("Enter video file path: ")
            output_path = input("Enter output path (optional, press Enter to skip): ")
            if not output_path:
                output_path = None
            
            try:
                pose_estimator = MoveNetPoseEstimator('thunder')
                keypoints_seq = pose_estimator.process_video_file(video_path, output_path)
                
                # Save keypoints
                json_path = video_path.replace('.mp4', '_keypoints.json')
                pose_estimator.save_keypoints_to_json(keypoints_seq, json_path)
                
            except Exception as e:
                print(f"Error processing video: {e}")
        
        elif choice == '4':
            test_pose_estimation()
        
        elif choice == '5':
            pose_estimator = MoveNetPoseEstimator('thunder')
            output_path = input("Enter output path for TFLite model: ")
            pose_estimator.export_to_tflite(output_path)
        
        elif choice == '6':
            break
        
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    # Check dependencies
    try:
        import tensorflow as tf
        import cv2
        print("✓ All dependencies available")
        print(f"TensorFlow version: {tf.__version__}")
        print(f"OpenCV version: {cv2.__version__}")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install: pip install tensorflow tensorflow-hub opencv-python")
        exit(1)
    
    # Run demo
    demo_menu()