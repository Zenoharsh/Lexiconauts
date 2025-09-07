# ==============================================================================
# MODEL 2: YOLOV8 OBJECT DETECTION & TRACKING
# File: yolo_detection_tracker.py
# ==============================================================================

import cv21
import numpy as np
from ultralytics import YOLO
import time
from collections import defaultdict, deque
import json
import os

class AthleteDetectionTracker:
    """YOLOv8-based athlete detection and tracking system"""
    
    def __init__(self, model_size='n', confidence_threshold=0.5, iou_threshold=0.5):
        """Initialize YOLOv8 model and tracking parameters
        
        Args:
            model_size: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (extra large)
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
        """
        self.model_size = model_size
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Load YOLOv8 model
        print(f"Loading YOLOv8{model_size} model...")
        self.model = YOLO(f'yolov8{model_size}.pt')
        print("YOLOv8 loaded successfully!")
        
        # Tracking variables
        self.track_history = defaultdict(list)
        self.track_colors = {}
        self.next_id = 0
        self.max_disappeared = 30  # frames before removing a track
        self.distance_threshold = 50  # pixels for track matching
        
        # Statistics
        self.detection_stats = {
            'total_detections': 0,
            'total_frames_processed': 0,
            'average_detections_per_frame': 0
        }
    
    def detect_athletes(self, frame, return_crops=False):
        """Detect people (athletes) in frame
        
        Args:
            frame: Input image frame
            return_crops: If True, return cropped athlete images
            
        Returns:
            List of detection dictionaries
        """
        # Run YOLO detection (class 0 = person)
        results = self.model(frame, classes=[0], conf=self.confidence_threshold, 
                           iou=self.iou_threshold, verbose=False)
        
        detections = []
        crops = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for i, (box, conf) in enumerate(zip(boxes, confidences)):
                x1, y1, x2, y2 = box.astype(int)
                
                detection = {
                    'id': i,
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(conf),
                    'centroid': [(x1 + x2) / 2, (y1 + y2) / 2],
                    'area': (x2 - x1) * (y2 - y1),
                    'class': 'person'
                }
                detections.append(detection)
                
                # Extract crop if requested
                if return_crops:
                    crop = frame[y1:y2, x1:x2]
                    crops.append(crop)
                
                self.detection_stats['total_detections'] += 1
        
        self.detection_stats['total_frames_processed'] += 1
        self.detection_stats['average_detections_per_frame'] = (
            self.detection_stats['total_detections'] / 
            self.detection_stats['total_frames_processed']
        )
        
        if return_crops:
            return detections, crops
        return detections
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        x1, y1, x2, y2 = box1
        x1_p, y1_p, x2_p, y2_p = box2
        
        # Calculate intersection area
        xi1 = max(x1, x1_p)
        yi1 = max(y1, y1_p)
        xi2 = min(x2, x2_p)
        yi2 = min(y2, y2_p)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        
        # Calculate union area
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_p - x1_p) * (y2_p - y1_p)
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0
    
    def simple_centroid_tracking(self, detections, frame_id):
        """Simple centroid-based tracking algorithm"""
        current_centroids = {}
        
        # Extract centroids from current detections
        for i, detection in enumerate(detections):
            current_centroids[i] = {
                'centroid': detection['centroid'],
                'bbox': detection['bbox'],
                'confidence': detection['confidence']
            }
        
        # If no existing tracks, initialize all as new
        if len(self.track_history) == 0:
            for i, centroid_data in current_centroids.items():
                track_id = self.next_id
                self.next_id += 1
                self.track_history[track_id] = [centroid_data]
                self.track_colors[track_id] = self._generate_color(track_id)
                centroid_data['track_id'] = track_id
            return list(current_centroids.values())
        
        # Match current detections with existing tracks
        matched_tracks = []
        used_detection_indices = set()
        used_track_ids = set()
        
        # Calculate distances between current detections and existing tracks
        for track_id, track_history in self.track_history.items():
            if len(track_history) == 0:
                continue
                
            last_centroid = track_history[-1]['centroid']
            best_match_idx = None
            min_distance = float('inf')
            
            for i, centroid_data in current_centroids.items():
                if i in used_detection_indices:
                    continue
                    
                current_centroid = centroid_data['centroid']
                distance = np.sqrt((current_centroid[0] - last_centroid[0])**2 + 
                                 (current_centroid[1] - last_centroid[1])**2)
                
                if distance < self.distance_threshold and distance < min_distance:
                    min_distance = distance
                    best_match_idx = i
            
            # If match found, update track
            if best_match_idx is not None:
                matched_data = current_centroids[best_match_idx].copy()
                matched_data['track_id'] = track_id
                matched_data['distance_moved'] = min_distance
                
                self.track_history[track_id].append(matched_data)
                
                # Keep only recent history (last 30 frames)
                if len(self.track_history[track_id]) > 30:
                    self.track_history[track_id] = self.track_history[track_id][-30:]
                
                matched_tracks.append(matched_data)
                used_detection_indices.add(best_match_idx)
                used_track_ids.add(track_id)
        
        # Create new tracks for unmatched detections
        for i, centroid_data in current_centroids.items():
            if i not in used_detection_indices:
                track_id = self.next_id
                self.next_id += 1
                centroid_data['track_id'] = track_id
                self.track_history[track_id] = [centroid_data]
                self.track_colors[track_id] = self._generate_color(track_id)
                matched_tracks.append(centroid_data)
        
        # Remove old tracks that haven't been updated
        tracks_to_remove = []
        for track_id in self.track_history.keys():
            if track_id not in used_track_ids:
                if len(self.track_history[track_id]) > 0:
                    # Check how long since last update
                    # For now, we'll keep all tracks (you can implement timeout logic here)
                    pass
        
        return matched_tracks
    
    def advanced_kalman_tracking(self, detections, frame_id):
        """More advanced Kalman filter-based tracking (placeholder)"""
        # This is a placeholder for more sophisticated tracking
        # Would implement Kalman filters for motion prediction
        return self.simple_centroid_tracking(detections, frame_id)
    
    def _generate_color(self, track_id):
        """Generate unique color for each track"""
        np.random.seed(track_id)
        return (
            int(np.random.randint(0, 255)),
            int(np.random.randint(0, 255)),
            int(np.random.randint(0, 255))
        )
    
    def draw_detections(self, frame, detections, show_trails=True):
        """Draw detection boxes and tracking information"""
        annotated_frame = frame.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            track_id = detection.get('track_id', -1)
            confidence = detection['confidence']
            
            x1, y1, x2, y2 = bbox
            
            # Get color for this track
            color = self.track_colors.get(track_id, (0, 255, 0))
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID and confidence
            label = f"ID:{track_id} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw centroid
            centroid = detection['centroid']
            cv2.circle(annotated_frame, (int(centroid[0]), int(centroid[1])), 
                      4, color, -1)
            
            # Draw trail if requested
            if show_trails and track_id in self.track_history:
                trail = self.track_history[track_id]
                if len(trail) > 1:
                    points = np.array([(int(t['centroid'][0]), int(t['centroid'][1])) 
                                     for t in trail[-10:]], dtype=np.int32)
                    cv2.polylines(annotated_frame, [points], False, color, 2)
        
        return annotated_frame
    
    def get_track_statistics(self, track_id):
        """Get statistics for a specific track"""
        if track_id not in self.track_history:
            return None
        
        track_data = self.track_history[track_id]
        if len(track_data) < 2:
            return None
        
        # Calculate movement statistics
        distances = []
        for i in range(1, len(track_data)):
            prev_centroid = track_data[i-1]['centroid']
            curr_centroid = track_data[i]['centroid']
            distance = np.sqrt((curr_centroid[0] - prev_centroid[0])**2 + 
                             (curr_centroid[1] - prev_centroid[1])**2)
            distances.append(distance)
        
        # Calculate bounding box sizes
        areas = [t['bbox'][2] * t['bbox'][3] for t in track_data]
        confidences = [t['confidence'] for t in track_data]
        
        return {
            'track_id': track_id,
            'total_frames': len(track_data),
            'avg_movement_per_frame': np.mean(distances) if distances else 0,
            'total_distance_moved': sum(distances) if distances else 0,
            'avg_confidence': np.mean(confidences),
            'avg_area': np.mean(areas),
            'first_seen_frame': 0,  # Would need frame tracking for this
            'last_seen_frame': len(track_data) - 1
        }
    
    def process_video_file(self, video_path, output_path=None, show_preview=True):
        """Process entire video file with detection and tracking"""
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
        
        all_tracks_data = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect athletes
            detections = self.detect_athletes(frame)
            
            # Track athletes
            tracks = self.simple_centroid_tracking(detections, frame_count)
            
            # Store tracking data
            frame_data = {
                'frame_id': frame_count,
                'detections': len(detections),
                'tracks': []
            }
            
            for track in tracks:
                frame_data['tracks'].append({
                    'track_id': track['track_id'],
                    'bbox': track['bbox'],
                    'confidence': track['confidence'],
                    'centroid': track['centroid']
                })
            
            all_tracks_data.append(frame_data)
            
            # Draw annotations
            annotated_frame = self.draw_detections(frame, tracks)
            
            # Add frame info
            cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Detections: {len(detections)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Active Tracks: {len(tracks)}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show preview
            if show_preview:
                cv2.imshow('Processing Video - Detection & Tracking', annotated_frame)
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
        
        print(f"Processing complete! Tracked {len(self.track_history)} unique objects")
        
        # Save tracking data
        if output_path:
            json_path = output_path.replace('.mp4', '_tracking_data.json')
            self.save_tracking_data(all_tracks_data, json_path)
        
        return all_tracks_data
    
    def real_time_tracking(self):
        """Real-time detection and tracking with webcam"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Real-time athlete tracking started")
        print("Press 'q' to quit, 'r' to reset tracks, 't' to toggle trails")
        
        show_trails = True
        frame_count = 0
        fps_counter = deque(maxlen=30)
        
        while cap.isOpened():
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect and track
            detections = self.detect_athletes(frame)
            tracks = self.simple_centroid_tracking(detections, frame_count)
            
            # Draw results
            annotated_frame = self.draw_detections(frame, tracks, show_trails)
            
            # Add statistics
            cv2.putText(annotated_frame, f"Detections: {len(detections)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Active Tracks: {len(tracks)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Total Tracks: {len(self.track_history)}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Calculate and display FPS
            end_time = time.time()
            fps = 1.0 / (end_time - start_time)
            fps_counter.append(fps)
            avg_fps = np.mean(fps_counter)
            
            cv2.putText(annotated_frame, f"FPS: {avg_fps:.1f}", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            cv2.imshow('Real-time Athlete Tracking', annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.track_history.clear()
                self.track_colors.clear()
                self.next_id = 0
                print("Tracking reset!")
            elif key == ord('t'):
                show_trails = not show_trails
                print(f"Trails: {'ON' if show_trails else 'OFF'}")
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        print(f"\nFinal Statistics:")
        print(f"Total unique tracks: {len(self.track_history)}")
        for track_id in self.track_history.keys():
            stats = self.get_track_statistics(track_id)
            if stats:
                print(f"Track {track_id}: {stats['total_frames']} frames, "
                      f"avg movement: {stats['avg_movement_per_frame']:.1f}px")
    
    def benchmark_model_performance(self, num_frames=100):
        """Benchmark detection performance"""
        print(f"Benchmarking YOLOv8{self.model_size} performance...")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam for benchmarking")
            return
        
        detection_times = []
        tracking_times = []
        
        for frame_count in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Benchmark detection
            start_time = time.time()
            detections = self.detect_athletes(frame)
            detection_time = time.time() - start_time
            detection_times.append(detection_time)
            
            # Benchmark tracking
            start_time = time.time()
            tracks = self.simple_centroid_tracking(detections, frame_count)
            tracking_time = time.time() - start_time
            tracking_times.append(tracking_time)
            
            if frame_count % 20 == 0:
                print(f"Benchmarked {frame_count}/{num_frames} frames...")
        
        cap.release()
        
        # Calculate statistics
        avg_detection_time = np.mean(detection_times)
        avg_tracking_time = np.mean(tracking_times)
        total_avg_time = avg_detection_time + avg_tracking_time
        
        print(f"\nPerformance Results:")
        print(f"Model: YOLOv8{self.model_size}")
        print(f"Average detection time: {avg_detection_time*1000:.1f}ms")
        print(f"Average tracking time: {avg_tracking_time*1000:.1f}ms")
        print(f"Total average time: {total_avg_time*1000:.1f}ms")
        print(f"Theoretical FPS: {1/total_avg_time:.1f}")
        print(f"Detection FPS: {1/avg_detection_time:.1f}")
        
        return {
            'model_size': self.model_size,
            'avg_detection_time': avg_detection_time,
            'avg_tracking_time': avg_tracking_time,
            'theoretical_fps': 1/total_avg_time,
            'detection_fps': 1/avg_detection_time
        }
    
    def save_tracking_data(self, tracking_data, output_path):
        """Save tracking results to JSON file"""
        save_data = {
            'model_info': {
                'model_size': self.model_size,
                'confidence_threshold': self.confidence_threshold,
                'iou_threshold': self.iou_threshold
            },
            'statistics': self.detection_stats,
            'tracking_data': tracking_data,
            'track_summaries': {}
        }
        
        # Add track summaries
        for track_id in self.track_history.keys():
            stats = self.get_track_statistics(track_id)
            if stats:
                save_data['track_summaries'][track_id] = stats
        
        with open(output_path, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        print(f"Tracking data saved to: {output_path}")
    
    def load_tracking_data(self, input_path):
        """Load tracking results from JSON file"""
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        self.model_size = data['model_info']['model_size']
        self.confidence_threshold = data['model_info']['confidence_threshold']
        self.iou_threshold = data['model_info']['iou_threshold']
        self.detection_stats = data['statistics']
        
        print(f"Tracking data loaded from: {input_path}")
        return data
    
    def export_to_mobile(self, export_formats=['tflite', 'coreml']):
        """Export YOLOv8 model for mobile deployment"""
        try:
            for format_name in export_formats:
                if format_name == 'tflite':
                    # Export to TensorFlow Lite
                    export_path = self.model.export(format='tflite', imgsz=320)
                    print(f"TFLite model exported: {export_path}")
                
                elif format_name == 'coreml':
                    # Export to CoreML (for iOS)
                    export_path = self.model.export(format='coreml', imgsz=320)
                    print(f"CoreML model exported: {export_path}")
                
                elif format_name == 'onnx':
                    # Export to ONNX (universal format)
                    export_path = self.model.export(format='onnx', imgsz=320)
                    print(f"ONNX model exported: {export_path}")
            
            print("Mobile export completed successfully!")
            
        except Exception as e:
            print(f"Mobile export failed: {e}")
    
    def analyze_crowd_density(self, detections):
        """Analyze crowd density and distribution"""
        if not detections:
            return {'density': 0, 'distribution': 'empty'}
        
        num_people = len(detections)
        
        # Calculate average distance between people
        if num_people > 1:
            centroids = [det['centroid'] for det in detections]
            distances = []
            for i in range(len(centroids)):
                for j in range(i+1, len(centroids)):
                    dist = np.sqrt((centroids[i][0] - centroids[j][0])**2 + 
                                 (centroids[i][1] - centroids[j][1])**2)
                    distances.append(dist)
            avg_distance = np.mean(distances)
        else:
            avg_distance = 0
        
        # Determine distribution pattern
        if num_people <= 2:
            distribution = 'sparse'
        elif avg_distance < 100:
            distribution = 'crowded'
        elif avg_distance < 200:
            distribution = 'moderate'
        else:
            distribution = 'sparse'
        
        return {
            'num_people': num_people,
            'avg_distance': avg_distance,
            'distribution': distribution,
            'density_score': num_people / (avg_distance + 1)
        }

# ==============================================================================
# DEMO AND TESTING FUNCTIONS
# ==============================================================================

def compare_model_sizes():
    """Compare performance of different YOLOv8 model sizes"""
    model_sizes = ['n', 's', 'm']  # Start with smaller models
    results = {}
    
    print("Comparing YOLOv8 model sizes...")
    
    for size in model_sizes:
        print(f"\nTesting YOLOv8{size}...")
        try:
            tracker = AthleteDetectionTracker(model_size=size)
            performance = tracker.benchmark_model_performance(num_frames=50)
            results[size] = performance
            
            # Clean up to save memory
            del tracker
            
        except Exception as e:
            print(f"Error testing YOLOv8{size}: {e}")
            results[size] = None
    
    # Display comparison
    print("\n" + "="*60)
    print("YOLOV8 MODEL COMPARISON")
    print("="*60)
    print(f"{'Model':<8} {'Det Time':<12} {'Track Time':<12} {'Total FPS':<10}")
    print("-"*60)
    
    for size, result in results.items():
        if result:
            print(f"YOLOv8{size:<3} {result['avg_detection_time']*1000:>8.1f}ms "
                  f"{result['avg_tracking_time']*1000:>8.1f}ms "
                  f"{result['theoretical_fps']:>8.1f}")
    
    return results

def demo_menu():
    """Interactive demo menu for detection and tracking"""
    while True:
        print("\n" + "="*50)
        print("YOLOV8 DETECTION & TRACKING DEMO")
        print("="*50)
        print("1. Real-time tracking (webcam)")
        print("2. Process video file")
        print("3. Benchmark model performance")
        print("4. Compare model sizes")
        print("5. Export models for mobile")
        print("6. Crowd density analysis (live)")
        print("7. Exit")
        
        choice = input("\nSelect option (1-7): ")
        
        if choice == '1':
            model_size = input("Enter model size (n/s/m/l/x) [default: n]: ").strip() or 'n'
            tracker = AthleteDetectionTracker(model_size=model_size)
            tracker.real_time_tracking()
        
        elif choice == '2':
            video_path = input("Enter video file path: ")
            output_path = input("Enter output path (optional): ").strip()
            if not output_path:
                output_path = None
            
            model_size = input("Enter model size (n/s/m/l/x) [default: n]: ").strip() or 'n'
            
            try:
                tracker = AthleteDetectionTracker(model_size=model_size)
                tracking_data = tracker.process_video_file(video_path, output_path)
                print(f"Processed {len(tracking_data)} frames")
                
            except Exception as e:
                print(f"Error processing video: {e}")
        
        elif choice == '3':
            model_size = input("Enter model size (n/s/m/l/x) [default: n]: ").strip() or 'n'
            tracker = AthleteDetectionTracker(model_size=model_size)
            tracker.benchmark_model_performance()
        
        elif choice == '4':
            compare_model_sizes()
        
        elif choice == '5':
            model_size = input("Enter model size (n/s/m/l/x) [default: n]: ").strip() or 'n'
            formats = input("Enter formats (tflite,coreml,onnx) [default: tflite]: ").strip()
            if not formats:
                formats = ['tflite']
            else:
                formats = [f.strip() for f in formats.split(',')]
            
            tracker = AthleteDetectionTracker(model_size=model_size)
            tracker.export_to_mobile(formats)
        
        elif choice == '6':
            tracker = AthleteDetectionTracker(model_size='n')
            
            cap = cv2.VideoCapture(0)
            print("Live crowd density analysis - Press 'q' to quit")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                detections = tracker.detect_athletes(frame)
                crowd_info = tracker.analyze_crowd_density(detections)
                
                annotated_frame = tracker.draw_detections(frame, 
                    tracker.simple_centroid_tracking(detections, 0))
                
                # Display crowd info
                cv2.putText(annotated_frame, f"People: {crowd_info['num_people']}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(annotated_frame, f"Density: {crowd_info['distribution']}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                cv2.imshow('Crowd Density Analysis', annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
        
        elif choice == '7':
            break
        
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    # Check dependencies
    try:
        from ultralytics import YOLO
        import cv2
        print("✓ All dependencies available")
        print(f"OpenCV version: {cv2.__version__}")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install: pip install ultralytics opencv-python")
        exit(1)
    
    # Run demo
    demo_menu()