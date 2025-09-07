# ==============================================================================
# INTEGRATED SPORTS ANALYSIS PIPELINE
# File: integrated_sports_pipeline.py
# ==============================================================================

import cv2
import numpy as np
import time
import json
import os
from pathlib import Path
from collections import deque, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Core ML Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import pandas as pd

# Try to import deep learning libraries
try:
    import tensorflow as tf
    import tensorflow_hub as hub
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available - MoveNet disabled")

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Ultralytics not available - YOLOv8 disabled")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe not available - alternative pose detection only")


class IntegratedSportsAnalysisPipeline:
    """Complete sports analysis pipeline integrating pose estimation, detection, tracking, and form analysis"""
    
    def __init__(self, config=None):
        """Initialize the integrated pipeline
        
        Args:
            config: Configuration dictionary for pipeline components
        """
        self.config = config or self._get_default_config()
        
        # Initialize components
        self.pose_estimator = None
        self.athlete_tracker = None
        self.form_classifier = None
        
        # Analysis data storage
        self.analysis_results = defaultdict(list)
        self.multi_person_data = {}
        self.session_stats = {
            'total_frames_processed': 0,
            'athletes_detected': 0,
            'pose_detections': 0,
            'form_assessments': 0,
            'processing_times': []
        }
        
        # Initialize available components
        self._initialize_components()
        
        print("Integrated Sports Analysis Pipeline initialized")
        print(f"Available components: {self._get_available_components()}")

    def _get_default_config(self):
        """Get default configuration for all pipeline components"""
        return {
            'pose_estimation': {
                'model_type': 'thunder',  # 'thunder' or 'lightning'
                'confidence_threshold': 0.3,
                'use_movenet': TENSORFLOW_AVAILABLE,
                'use_mediapipe': MEDIAPIPE_AVAILABLE
            },
            'detection_tracking': {
                'model_size': 'n',  # YOLOv8 model size
                'confidence_threshold': 0.5,
                'iou_threshold': 0.5,
                'tracking_enabled': True
            },
            'form_analysis': {
                'exercises': ['pushup', 'squat', 'situp', 'plank'],
                'quality_threshold': 0.6,
                'feedback_enabled': True
            },
            'pipeline': {
                'max_people_tracked': 5,
                'pose_buffer_size': 90,  # 3 seconds at 30fps
                'enable_multi_person': True,
                'save_intermediate_results': True
            }
        }

    def _initialize_components(self):
        """Initialize all available pipeline components"""
        print("Initializing pipeline components...")
        
        # Initialize pose estimation
        if self.config['pose_estimation']['use_movenet'] and TENSORFLOW_AVAILABLE:
            try:
                self.pose_estimator = MoveNetPoseEstimator(
                    model_type=self.config['pose_estimation']['model_type']
                )
                print("✓ MoveNet pose estimator initialized")
            except Exception as e:
                print(f"✗ MoveNet initialization failed: {e}")
        
        elif self.config['pose_estimation']['use_mediapipe'] and MEDIAPIPE_AVAILABLE:
            try:
                self.pose_estimator = MediaPipePoseEstimator()
                print("✓ MediaPipe pose estimator initialized")
            except Exception as e:
                print(f"✗ MediaPipe initialization failed: {e}")
        
        # Initialize detection and tracking
        if ULTRALYTICS_AVAILABLE:
            try:
                self.athlete_tracker = AthleteDetectionTracker(
                    model_size=self.config['detection_tracking']['model_size'],
                    confidence_threshold=self.config['detection_tracking']['confidence_threshold'],
                    iou_threshold=self.config['detection_tracking']['iou_threshold']
                )
                print("✓ YOLOv8 athlete tracker initialized")
            except Exception as e:
                print(f"✗ YOLOv8 initialization failed: {e}")
        
        # Initialize form classifier
        try:
            self.form_classifier = ExerciseFormClassifier()
            print("✓ Exercise form classifier initialized")
        except Exception as e:
            print(f"✗ Form classifier initialization failed: {e}")

    def _get_available_components(self):
        """Return list of available components"""
        components = []
        if self.pose_estimator:
            components.append("Pose Estimation")
        if self.athlete_tracker:
            components.append("Detection/Tracking")
        if self.form_classifier:
            components.append("Form Analysis")
        return components

    def process_single_person_frame(self, frame, exercise_type='pushup', person_id=0):
        """Process a single frame for single-person exercise analysis
        
        Args:
            frame: Input video frame
            exercise_type: Type of exercise being performed
            person_id: ID for tracking this person
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        start_time = time.time()
        
        results = {
            'frame_analysis': {},
            'pose_data': None,
            'form_assessment': None,
            'person_id': person_id,
            'exercise_type': exercise_type,
            'timestamp': time.time()
        }
        
        try:
            # Step 1: Pose Estimation
            if self.pose_estimator:
                pose_data = self._extract_pose_data(frame)
                results['pose_data'] = pose_data
                
                if pose_data and len(pose_data) > 0:
                    # Step 2: Form Analysis
                    if self.form_classifier:
                        form_results = self._analyze_exercise_form(
                            pose_data, exercise_type, person_id
                        )
                        results['form_assessment'] = form_results
                    
                    # Step 3: Frame Quality Analysis
                    results['frame_analysis'] = self._analyze_frame_quality(pose_data)
            
            # Record processing time
            processing_time = time.time() - start_time
            results['processing_time'] = processing_time
            self.session_stats['processing_times'].append(processing_time)
            self.session_stats['total_frames_processed'] += 1
            
            return results
            
        except Exception as e:
            results['error'] = str(e)
            return results

    def process_multi_person_frame(self, frame):
        """Process a single frame for multi-person analysis
        
        Args:
            frame: Input video frame
            
        Returns:
            Dictionary with analysis results for all detected people
        """
        start_time = time.time()
        
        results = {
            'detections': [],
            'tracked_people': {},
            'crowd_analysis': None,
            'timestamp': time.time()
        }
        
        try:
            # Step 1: Detect and track people
            if self.athlete_tracker:
                detections = self.athlete_tracker.detect_athletes(frame)
                tracks = self.athlete_tracker.simple_centroid_tracking(
                    detections, self.session_stats['total_frames_processed']
                )
                
                results['detections'] = detections
                results['crowd_analysis'] = self.athlete_tracker.analyze_crowd_density(detections)
                self.session_stats['athletes_detected'] += len(detections)
                
                # Step 2: Analyze each tracked person
                for track in tracks[:self.config['pipeline']['max_people_tracked']]:
                    person_id = track['track_id']
                    
                    # Extract person from frame
                    bbox = track['bbox']
                    x1, y1, x2, y2 = bbox
                    person_crop = frame[y1:y2, x1:x2]
                    
                    if person_crop.size > 0:
                        # Analyze individual person
                        person_results = self.process_single_person_frame(
                            person_crop, 'unknown', person_id
                        )
                        results['tracked_people'][person_id] = person_results
            
            # Record processing time
            processing_time = time.time() - start_time
            results['processing_time'] = processing_time
            self.session_stats['processing_times'].append(processing_time)
            self.session_stats['total_frames_processed'] += 1
            
            return results
            
        except Exception as e:
            results['error'] = str(e)
            return results

    def _extract_pose_data(self, frame):
        """Extract pose keypoints from frame using available pose estimator"""
        if not self.pose_estimator:
            return None
        
        try:
            if hasattr(self.pose_estimator, 'extract_keypoints'):
                # MoveNet or custom pose estimator
                keypoints = self.pose_estimator.extract_keypoints(frame)
                self.session_stats['pose_detections'] += 1
                return keypoints
            else:
                # Fallback to generic extraction
                return self._generic_pose_extraction(frame)
                
        except Exception as e:
            print(f"Pose extraction error: {e}")
            return None

    def _analyze_exercise_form(self, pose_data, exercise_type, person_id):
        """Analyze exercise form using the form classifier"""
        if not self.form_classifier or pose_data is None:
            return None
        
        try:
            # Convert pose data to feature format expected by classifier
            if person_id not in self.multi_person_data:
                self.multi_person_data[person_id] = {
                    'pose_buffer': deque(maxlen=self.config['pipeline']['pose_buffer_size']),
                    'exercise_type': exercise_type
                }
            
            # Add current pose data to buffer
            self.multi_person_data[person_id]['pose_buffer'].append(pose_data)
            pose_sequence = list(self.multi_person_data[person_id]['pose_buffer'])
            
            # Extract features for form analysis
            if exercise_type in self.form_classifier.feature_extractors:
                features = self.form_classifier.feature_extractors[exercise_type](pose_sequence)
            else:
                features = self.form_classifier._extract_generic_features(pose_sequence)
            
            # Predict form quality if we have enough data
            if len(features) > 0 and len(pose_sequence) >= 10:
                form_results = self.form_classifier.predict_form_quality(features, exercise_type)
                self.session_stats['form_assessments'] += 1
                return form_results
            
            return {'status': 'insufficient_data', 'buffer_size': len(pose_sequence)}
            
        except Exception as e:
            return {'error': str(e)}

    def _analyze_frame_quality(self, pose_data):
        """Analyze the quality of pose detection in the current frame"""
        if pose_data is None:
            return {'quality_score': 0, 'status': 'no_pose_data'}
        
        try:
            # Calculate confidence-based metrics
            if hasattr(pose_data, 'shape') and len(pose_data.shape) == 2:
                # Assuming pose_data is shape (num_keypoints, 3) with confidence in 3rd column
                confidences = pose_data[:, 2] if pose_data.shape[1] > 2 else np.ones(pose_data.shape[0])
                visible_keypoints = np.sum(confidences > self.config['pose_estimation']['confidence_threshold'])
                avg_confidence = np.mean(confidences)
                
                return {
                    'quality_score': avg_confidence,
                    'visible_keypoints': int(visible_keypoints),
                    'total_keypoints': len(confidences),
                    'visibility_ratio': visible_keypoints / len(confidences)
                }
            else:
                return {'quality_score': 1.0, 'status': 'alternative_format'}
                
        except Exception as e:
            return {'error': str(e), 'quality_score': 0}

    def analyze_video_file(self, video_path, exercise_type='pushup', output_dir=None, 
                          multi_person=False):
        """Analyze complete video file
        
        Args:
            video_path: Path to video file
            exercise_type: Type of exercise (for single person analysis)
            output_dir: Directory to save results
            multi_person: Whether to enable multi-person analysis
            
        Returns:
            Complete analysis results
        """
        print(f"Analyzing video: {video_path}")
        print(f"Multi-person mode: {multi_person}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video: {width}x{height}, {fps}fps, {total_frames} frames")
        
        # Setup output
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
            
            # Setup output video writer
            output_video_path = output_dir / f"analyzed_{Path(video_path).name}"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        else:
            video_writer = None
        
        # Process each frame
        all_results = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame based on mode
            if multi_person:
                frame_results = self.process_multi_person_frame(frame)
            else:
                frame_results = self.process_single_person_frame(frame, exercise_type)
            
            frame_results['frame_number'] = frame_count
            all_results.append(frame_results)
            
            # Create annotated frame
            annotated_frame = self._create_annotated_frame(frame, frame_results, multi_person)
            
            # Write output frame
            if video_writer:
                video_writer.write(annotated_frame)
            
            # Show progress
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
            
            frame_count += 1
            
            # Optional: Show preview (comment out for faster processing)
            # cv2.imshow('Analysis Preview', annotated_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        
        # Generate comprehensive report
        analysis_summary = self._generate_analysis_summary(all_results, multi_person)
        
        # Save results
        if output_dir:
            self._save_analysis_results(all_results, analysis_summary, output_dir, video_path)
        
        print(f"Analysis complete! Processed {frame_count} frames")
        return {
            'frame_results': all_results,
            'summary': analysis_summary,
            'session_stats': self.session_stats
        }

    def _create_annotated_frame(self, frame, results, multi_person=False):
        """Create annotated frame with analysis results"""
        annotated_frame = frame.copy()
        
        try:
            if multi_person:
                # Multi-person annotations
                if self.athlete_tracker and 'detections' in results:
                    # Draw detection boxes
                    detections = results.get('detections', [])
                    if detections:
                        for detection in detections:
                            bbox = detection['bbox']
                            confidence = detection['confidence']
                            x1, y1, x2, y2 = bbox
                            
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(annotated_frame, f"{confidence:.2f}", 
                                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Crowd analysis info
                crowd_info = results.get('crowd_analysis', {})
                if crowd_info:
                    cv2.putText(annotated_frame, f"People: {crowd_info.get('num_people', 0)}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(annotated_frame, f"Density: {crowd_info.get('distribution', 'unknown')}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            else:
                # Single person annotations
                if self.pose_estimator and results.get('pose_data') is not None:
                    # Draw pose skeleton
                    pose_data = results['pose_data']
                    annotated_frame = self._draw_pose_skeleton(annotated_frame, pose_data)
                
                # Form assessment info
                form_assessment = results.get('form_assessment')
                if form_assessment and 'quality' in form_assessment:
                    quality = form_assessment['quality']
                    confidence = form_assessment.get('confidence', 0)
                    score = form_assessment.get('score', 0)
                    
                    # Quality text color based on quality
                    color = (0, 255, 0) if quality == 'good' else (0, 0, 255)
                    
                    cv2.putText(annotated_frame, f"Form: {quality.upper()}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(annotated_frame, f"Score: {score:.1f}/10", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(annotated_frame, f"Confidence: {confidence:.2%}", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Frame quality info
                frame_analysis = results.get('frame_analysis', {})
                if 'visible_keypoints' in frame_analysis:
                    visible = frame_analysis['visible_keypoints']
                    total = frame_analysis.get('total_keypoints', 17)
                    cv2.putText(annotated_frame, f"Pose: {visible}/{total}", 
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # General info
            processing_time = results.get('processing_time', 0)
            cv2.putText(annotated_frame, f"Process: {processing_time*1000:.1f}ms", 
                       (10, annotated_frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        except Exception as e:
            cv2.putText(annotated_frame, f"Annotation Error: {str(e)[:50]}", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return annotated_frame

    def _draw_pose_skeleton(self, frame, pose_data):
        """Draw pose skeleton on frame"""
        if pose_data is None or not hasattr(self.pose_estimator, 'draw_skeleton'):
            return frame
        
        try:
            return self.pose_estimator.draw_skeleton(frame, pose_data)
        except:
            # Fallback to simple keypoint drawing
            return self._draw_simple_keypoints(frame, pose_data)

    def _draw_simple_keypoints(self, frame, pose_data):
        """Simple keypoint drawing fallback"""
        if pose_data is None or len(pose_data.shape) != 2:
            return frame
        
        height, width = frame.shape[:2]
        for i, keypoint in enumerate(pose_data):
            if len(keypoint) >= 3 and keypoint[2] > 0.3:  # confidence threshold
                x, y = int(keypoint[1] * width), int(keypoint[0] * height)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        
        return frame

    def _generate_analysis_summary(self, all_results, multi_person=False):
        """Generate comprehensive analysis summary"""
        summary = {
            'total_frames': len(all_results),
            'processing_stats': {},
            'performance_metrics': {},
            'analysis_quality': {}
        }
        
        # Processing statistics
        processing_times = [r.get('processing_time', 0) for r in all_results if 'processing_time' in r]
        if processing_times:
            summary['processing_stats'] = {
                'avg_processing_time': np.mean(processing_times),
                'max_processing_time': np.max(processing_times),
                'min_processing_time': np.min(processing_times),
                'theoretical_fps': 1 / np.mean(processing_times) if np.mean(processing_times) > 0 else 0
            }
        
        if multi_person:
            # Multi-person analysis summary
            total_detections = sum(len(r.get('detections', [])) for r in all_results)
            summary['detection_stats'] = {
                'total_people_detected': total_detections,
                'avg_people_per_frame': total_detections / len(all_results) if all_results else 0
            }
            
            # Crowd density analysis
            density_data = [r.get('crowd_analysis', {}) for r in all_results if 'crowd_analysis' in r]
            if density_data:
                distributions = [d.get('distribution', 'unknown') for d in density_data]
                summary['crowd_analysis'] = {
                    'distribution_modes': pd.Series(distributions).value_counts().to_dict()
                }
        
        else:
            # Single person analysis summary
            form_assessments = [r.get('form_assessment') for r in all_results 
                              if r.get('form_assessment') and 'quality' in r.get('form_assessment', {})]
            
            if form_assessments:
                qualities = [f['quality'] for f in form_assessments]
                scores = [f.get('score', 0) for f in form_assessments]
                
                summary['form_analysis'] = {
                    'total_assessments': len(form_assessments),
                    'quality_distribution': pd.Series(qualities).value_counts().to_dict(),
                    'average_score': np.mean(scores),
                    'score_std': np.std(scores)
                }
        
        # Analysis quality metrics
        frame_qualities = [r.get('frame_analysis', {}).get('quality_score', 0) for r in all_results]
        if frame_qualities:
            summary['analysis_quality'] = {
                'avg_pose_quality': np.mean(frame_qualities),
                'pose_detection_rate': sum(1 for q in frame_qualities if q > 0.3) / len(frame_qualities)
            }
        
        return summary

    def _save_analysis_results(self, results, summary, output_dir, video_path):
        """Save analysis results to files"""
        output_dir = Path(output_dir)
        base_name = Path(video_path).stem
        
        # Save detailed results
        results_path = output_dir / f"{base_name}_detailed_results.json"
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = self._convert_numpy_for_json(results)
            json.dump(json_results, f, indent=2, default=str)
        
        # Save summary
        summary_path = output_dir / f"{base_name}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save session statistics
        stats_path = output_dir / f"{base_name}_session_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(self.session_stats, f, indent=2, default=str)
        
        # Generate visualization
        self._create_analysis_visualization(results, summary, output_dir, base_name)
        
        print(f"Results saved to: {output_dir}")

    def _convert_numpy_for_json(self, obj):
        """Convert numpy arrays to lists for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_for_json(v) for v in obj]
        else:
            return obj

    def _create_analysis_visualization(self, results, summary, output_dir, base_name):
        """Create visualization plots for analysis results"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Sports Analysis Results: {base_name}', fontsize=16)
            
            # Processing time over frames
            processing_times = [r.get('processing_time', 0) * 1000 for r in results]  # Convert to ms
            axes[0, 0].plot(processing_times)
            axes[0, 0].set_title('Processing Time per Frame')
            axes[0, 0].set_xlabel('Frame Number')
            axes[0, 0].set_ylabel('Processing Time (ms)')
            
            # Quality scores over time
            quality_scores = [r.get('frame_analysis', {}).get('quality_score', 0) for r in results]
            axes[0, 1].plot(quality_scores, color='green')
            axes[0, 1].set_title('Pose Detection Quality')
            axes[0, 1].set_xlabel('Frame Number')
            axes[0, 1].set_ylabel('Quality Score')
            axes[0, 1].set_ylim([0, 1])
            
            # Form quality distribution (if available)
            form_qualities = [r.get('form_assessment', {}).get('quality', 'unknown') 
                            for r in results if r.get('form_assessment')]
            if form_qualities:
                quality_counts = pd.Series(form_qualities).value_counts()
                axes[1, 0].bar(quality_counts.index, quality_counts.values, 
                              color=['green' if q == 'good' else 'red' for q in quality_counts.index])
                axes[1, 0].set_title('Form Quality Distribution')
                axes[1, 0].set_ylabel('Count')
            else:
                axes[1, 0].text(0.5, 0.5, 'No Form Data Available', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Form Quality Distribution')
            
            # Performance metrics
            if 'processing_stats' in summary:
                stats = summary['processing_stats']
                metrics = ['avg_processing_time', 'theoretical_fps']
                values = [stats.get(m, 0) for m in metrics]
                labels = ['Avg Process Time (s)', 'Theoretical FPS']
                
                axes[1, 1].bar(labels, values)
                axes[1, 1].set_title('Performance Metrics')
                axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plot_path = output_dir / f"{base_name}_analysis_plots.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Visualization saved: {plot_path}")
            
        except Exception as e:
            print(f"Visualization creation failed: {e}")

    def real_time_analysis(self, exercise_type='pushup', multi_person=False, 
                          show_stats=True):
        """Real-time analysis using webcam"""
        print(f"Starting real-time analysis...")
        print(f"Exercise type: {exercise_type}")
        print(f"Multi-person mode: {multi_person}")
        print("Press 'q' to quit, 's' to save screenshot, 'r' to reset stats")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Real-time statistics
        fps_counter = deque(maxlen=30)
        frame_count = 0
        
        while cap.isOpened():
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            if multi_person:
                results = self.process_multi_person_frame(frame)
            else:
                results = self.process_single_person_frame(frame, exercise_type)
            
            # Create annotated frame
            annotated_frame = self._create_annotated_frame(frame, results, multi_person)
            
            # Add real-time stats
            end_time = time.time()
            fps = 1.0 / (end_time - start_time)
            fps_counter.append(fps)
            avg_fps = np.mean(fps_counter)
            
            if show_stats:
                cv2.putText(annotated_frame, f"FPS: {avg_fps:.1f}", 
                           (annotated_frame.shape[1] - 120, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(annotated_frame, f"Frames: {frame_count}", 
                           (annotated_frame.shape[1] - 120, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Display frame
            cv2.imshow('Integrated Sports Analysis', annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_path = f"sports_analysis_screenshot_{int(time.time())}.jpg"
                cv2.imwrite(screenshot_path, annotated_frame)
                print(f"Screenshot saved: {screenshot_path}")
            elif key == ord('r'):
                self.session_stats = {
                    'total_frames_processed': 0,
                    'athletes_detected': 0,
                    'pose_detections': 0,
                    'form_assessments': 0,
                    'processing_times': []
                }
                self.multi_person_data.clear()
                print("Statistics reset!")
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        print(f"\nReal-time Analysis Complete:")
        print(f"Frames processed: {frame_count}")
        print(f"Average FPS: {np.mean(fps_counter):.1f}")
        print(f"Total pose detections: {self.session_stats['pose_detections']}")
        print(f"Total form assessments: {self.session_stats['form_assessments']}")


# ==============================================================================
# SUPPORTING CLASSES (Simplified versions for integration)
# ==============================================================================

class MoveNetPoseEstimator:
    """Simplified MoveNet pose estimator for integration"""
    
    def __init__(self, model_type='thunder'):
        self.model_type = model_type
        if TENSORFLOW_AVAILABLE:
            self.model_url = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
            self.input_size = 256 if model_type == 'thunder' else 192
            self.movenet = hub.load(self.model_url)
            self.movenet_model = self.movenet.signatures['serving_default']
            print(f"MoveNet {model_type} loaded")
        else:
            raise RuntimeError("TensorFlow not available")
    
    def extract_keypoints(self, frame):
        """Extract keypoints from frame"""
        if not TENSORFLOW_AVAILABLE:
            return None
        
        # Preprocess frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = tf.image.resize_with_pad(frame_rgb, self.input_size, self.input_size)
        frame_int32 = tf.cast(frame_resized, dtype=tf.int32)
        input_tensor = tf.expand_dims(frame_int32, axis=0)
        
        # Run inference
        outputs = self.movenet_model(input_tensor)
        keypoints = outputs['output_0'].numpy()[0, 0, :, :]
        return keypoints
    
    def draw_skeleton(self, frame, keypoints, confidence_threshold=0.3):
        """Draw pose skeleton on frame"""
        height, width = frame.shape[:2]
        annotated_frame = frame.copy()
        
        # Define connections
        connections = [
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        
        # Draw connections
        for connection in connections:
            kpt1_idx, kpt2_idx = connection
            if (kpt1_idx < len(keypoints) and kpt2_idx < len(keypoints) and 
                keypoints[kpt1_idx][2] > confidence_threshold and 
                keypoints[kpt2_idx][2] > confidence_threshold):
                
                y1, x1, _ = keypoints[kpt1_idx]
                y2, x2, _ = keypoints[kpt2_idx]
                x1_px, y1_px = int(x1 * width), int(y1 * height)
                x2_px, y2_px = int(x2 * width), int(y2 * height)
                cv2.line(annotated_frame, (x1_px, y1_px), (x2_px, y2_px), (0, 255, 0), 2)
        
        # Draw keypoints
        for i, (y, x, confidence) in enumerate(keypoints):
            if confidence > confidence_threshold:
                x_px, y_px = int(x * width), int(y * height)
                cv2.circle(annotated_frame, (x_px, y_px), 4, (255, 0, 0), -1)
        
        return annotated_frame


class MediaPipePoseEstimator:
    """Simplified MediaPipe pose estimator for integration"""
    
    def __init__(self):
        if MEDIAPIPE_AVAILABLE:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5
            )
            self.mp_draw = mp.solutions.drawing_utils
            print("MediaPipe pose estimator loaded")
        else:
            raise RuntimeError("MediaPipe not available")
    
    def extract_keypoints(self, frame):
        """Extract keypoints using MediaPipe"""
        if not MEDIAPIPE_AVAILABLE:
            return None
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            keypoints = []
            for landmark in results.pose_landmarks.landmark:
                keypoints.append([landmark.y, landmark.x, landmark.visibility])
            return np.array(keypoints)
        return None
    
    def draw_skeleton(self, frame, keypoints):
        """Draw MediaPipe skeleton"""
        # For MediaPipe format, we need to convert back to MediaPipe format
        # This is a simplified version
        height, width = frame.shape[:2]
        annotated_frame = frame.copy()
        
        for i, keypoint in enumerate(keypoints):
            if len(keypoint) >= 3 and keypoint[2] > 0.3:
                x, y = int(keypoint[1] * width), int(keypoint[0] * height)
                cv2.circle(annotated_frame, (x, y), 3, (0, 255, 0), -1)
        
        return annotated_frame


class AthleteDetectionTracker:
    """Simplified YOLOv8 detection and tracking for integration"""
    
    def __init__(self, model_size='n', confidence_threshold=0.5, iou_threshold=0.5):
        self.model_size = model_size
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        if ULTRALYTICS_AVAILABLE:
            self.model = YOLO(f'yolov8{model_size}.pt')
            self.track_history = defaultdict(list)
            self.next_id = 0
            print(f"YOLOv8{model_size} loaded")
        else:
            raise RuntimeError("Ultralytics not available")
    
    def detect_athletes(self, frame):
        """Detect people in frame"""
        if not ULTRALYTICS_AVAILABLE:
            return []
        
        results = self.model(frame, classes=[0], conf=self.confidence_threshold, 
                           iou=self.iou_threshold, verbose=False)
        
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for i, (box, conf) in enumerate(zip(boxes, confidences)):
                x1, y1, x2, y2 = box.astype(int)
                detections.append({
                    'id': i,
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(conf),
                    'centroid': [(x1 + x2) / 2, (y1 + y2) / 2],
                    'area': (x2 - x1) * (y2 - y1)
                })
        
        return detections
    
    def simple_centroid_tracking(self, detections, frame_id):
        """Simple tracking implementation"""
        tracked_objects = []
        for i, detection in enumerate(detections):
            detection['track_id'] = i  # Simplified tracking
            tracked_objects.append(detection)
        return tracked_objects
    
    def analyze_crowd_density(self, detections):
        """Analyze crowd density"""
        num_people = len(detections)
        if num_people <= 2:
            distribution = 'sparse'
        elif num_people <= 5:
            distribution = 'moderate'
        else:
            distribution = 'crowded'
        
        return {
            'num_people': num_people,
            'distribution': distribution
        }


class ExerciseFormClassifier:
    """Simplified form classifier for integration"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_extractors = {
            'pushup': self._extract_pushup_features,
            'squat': self._extract_squat_features,
            'situp': self._extract_situp_features,
            'plank': self._extract_plank_features,
        }
        print("Exercise form classifier loaded")
    
    def _extract_pushup_features(self, keypoints_sequence):
        """Extract pushup features"""
        if not keypoints_sequence:
            return np.array([])
        
        # Simplified feature extraction
        features = []
        for keypoints in keypoints_sequence[-10:]:  # Use last 10 frames
            if len(keypoints) >= 17:
                # Mock elbow angle calculation
                elbow_angle = 90 + np.random.normal(0, 10)  # Simplified
                features.append(elbow_angle)
        
        if features:
            return np.array([np.mean(features), np.std(features), 
                           len(features), max(features) - min(features)])
        return np.array([])
    
    def _extract_squat_features(self, keypoints_sequence):
        """Extract squat features"""
        return self._extract_pushup_features(keypoints_sequence)  # Simplified
    
    def _extract_situp_features(self, keypoints_sequence):
        """Extract situp features"""
        return self._extract_pushup_features(keypoints_sequence)  # Simplified
    
    def _extract_plank_features(self, keypoints_sequence):
        """Extract plank features"""
        return self._extract_pushup_features(keypoints_sequence)  # Simplified
    
    def _extract_generic_features(self, keypoints_sequence):
        """Generic feature extraction"""
        if not keypoints_sequence:
            return np.array([])
        return np.array([len(keypoints_sequence), np.random.random(), 
                        np.random.random(), np.random.random()])
    
    def predict_form_quality(self, features, exercise_type):
        """Predict form quality (simplified)"""
        if len(features) == 0:
            return {'quality': 'unknown', 'confidence': 0.0, 'score': 0.0}
        
        # Mock prediction based on feature values
        quality_score = np.mean(features) / 100.0  # Normalize
        quality = 'good' if quality_score > 0.7 else 'poor'
        confidence = min(0.95, quality_score + 0.1)
        score = quality_score * 10
        
        return {
            'quality': quality,
            'confidence': confidence,
            'score': score,
            'exercise': exercise_type
        }


# ==============================================================================
# DEMO AND MAIN FUNCTIONS
# ==============================================================================

def create_demo_pipeline():
    """Create a demo pipeline with synthetic data for testing"""
    config = {
        'pose_estimation': {
            'model_type': 'thunder',
            'confidence_threshold': 0.3,
            'use_movenet': TENSORFLOW_AVAILABLE,
            'use_mediapipe': MEDIAPIPE_AVAILABLE and not TENSORFLOW_AVAILABLE
        },
        'detection_tracking': {
            'model_size': 'n',
            'confidence_threshold': 0.5,
            'iou_threshold': 0.5,
            'tracking_enabled': True
        },
        'form_analysis': {
            'exercises': ['pushup', 'squat', 'situp', 'plank'],
            'quality_threshold': 0.6,
            'feedback_enabled': True
        },
        'pipeline': {
            'max_people_tracked': 3,
            'pose_buffer_size': 60,
            'enable_multi_person': True,
            'save_intermediate_results': True
        }
    }
    
    pipeline = IntegratedSportsAnalysisPipeline(config)
    return pipeline


def demo_menu():
    """Interactive demo menu for the integrated pipeline"""
    while True:
        print("\n" + "="*60)
        print("INTEGRATED SPORTS ANALYSIS PIPELINE")
        print("="*60)
        print("1. Real-time single person analysis")
        print("2. Real-time multi-person analysis")
        print("3. Analyze video file (single person)")
        print("4. Analyze video file (multi-person)")
        print("5. Test pipeline components")
        print("6. Create demo pipeline")
        print("7. Exit")
        
        choice = input("\nSelect option (1-7): ")
        
        if choice == '1':
            pipeline = create_demo_pipeline()
            exercise = input("Enter exercise type [pushup/squat/situp/plank]: ").strip() or 'pushup'
            pipeline.real_time_analysis(exercise_type=exercise, multi_person=False)
        
        elif choice == '2':
            pipeline = create_demo_pipeline()
            pipeline.real_time_analysis(multi_person=True)
        
        elif choice == '3':
            video_path = input("Enter video file path: ")
            exercise = input("Enter exercise type [pushup/squat/situp/plank]: ").strip() or 'pushup'
            output_dir = input("Enter output directory (optional): ").strip() or None
            
            try:
                pipeline = create_demo_pipeline()
                results = pipeline.analyze_video_file(
                    video_path, exercise_type=exercise, 
                    output_dir=output_dir, multi_person=False
                )
                print(f"Analysis complete! Processed {results['summary']['total_frames']} frames")
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '4':
            video_path = input("Enter video file path: ")
            output_dir = input("Enter output directory (optional): ").strip() or None
            
            try:
                pipeline = create_demo_pipeline()
                results = pipeline.analyze_video_file(
                    video_path, output_dir=output_dir, multi_person=True
                )
                print(f"Analysis complete! Processed {results['summary']['total_frames']} frames")
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '5':
            print(f"\nComponent Status:")
            print(f"TensorFlow (MoveNet): {'✓' if TENSORFLOW_AVAILABLE else '✗'}")
            print(f"Ultralytics (YOLOv8): {'✓' if ULTRALYTICS_AVAILABLE else '✗'}")
            print(f"MediaPipe: {'✓' if MEDIAPIPE_AVAILABLE else '✗'}")
            
            if TENSORFLOW_AVAILABLE or ULTRALYTICS_AVAILABLE or MEDIAPIPE_AVAILABLE:
                pipeline = create_demo_pipeline()
                print(f"Available pipeline components: {pipeline._get_available_components()}")
            else:
                print("No deep learning components available. Install dependencies.")
        
        elif choice == '6':
            print("Creating demo pipeline...")
            pipeline = create_demo_pipeline()
            print("Demo pipeline created successfully!")
            print(f"Available components: {pipeline._get_available_components()}")
            
            # Run a quick test
            print("\nRunning component test...")
            test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            results = pipeline.process_single_person_frame(test_frame, 'pushup')
            print(f"Test results: {list(results.keys())}")
        
        elif choice == '7':
            break
        
        else:
            print("Invalid choice!")


if __name__ == "__main__":
    print("Integrated Sports Analysis Pipeline")
    print("Checking dependencies...")
    
    print(f"TensorFlow: {'✓' if TENSORFLOW_AVAILABLE else '✗ pip install tensorflow tensorflow-hub'}")
    print(f"Ultralytics: {'✓' if ULTRALYTICS_AVAILABLE else '✗ pip install ultralytics'}")
    print(f"MediaPipe: {'✓' if MEDIAPIPE_AVAILABLE else '✗ pip install mediapipe'}")
    print(f"OpenCV: {'✓' if 'cv2' in locals() or 'cv2' in globals() else '✗ pip install opencv-python'}")
    
    if not any([TENSORFLOW_AVAILABLE, ULTRALYTICS_AVAILABLE, MEDIAPIPE_AVAILABLE]):
        print("\nNo deep learning libraries available!")
        print("Install at least one of: tensorflow, ultralytics, mediapipe")
        print("Example: pip install tensorflow ultralytics mediapipe opencv-python")
    else:
        demo_menu()