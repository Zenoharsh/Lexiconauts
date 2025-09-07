"""
Dataset Organization and Preprocessing
File: src/preprocessing/dataset_organizer.py
"""

import os
import csv
import json
import shutil
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import cv2

class DatasetOrganizer:
    def __init__(self, raw_data_path="./datasets/raw_data", 
                 processed_path="./datasets/processed_data"):
        self.raw_path = Path(raw_data_path)
        self.processed_path = Path(processed_path)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
        # Create organized structure
        self.exercise_types = ['pushup', 'squat', 'situp', 'plank', 'deadlift', 'pullup', 'lunge']
        self.quality_types = ['good', 'poor', 'unknown']
        
        for exercise in self.exercise_types:
            for quality in self.quality_types:
                (self.processed_path / exercise / quality).mkdir(parents=True, exist_ok=True)
    
    def organize_kaggle_fitness_data(self):
        """Organize Kaggle fitness dataset"""
        print("Organizing Kaggle fitness data...")
        
        kaggle_path = self.raw_path / "kaggle_fitness"
        if not kaggle_path.exists():
            print("Kaggle fitness dataset not found")
            return []
        
        organized_data = []
        
        # Look for video files and annotations
        video_files = list(kaggle_path.rglob("*.mp4")) + list(kaggle_path.rglob("*.avi"))
        
        print(f"Found {len(video_files)} video files")
        
        for video_file in tqdm(video_files, desc="Organizing Kaggle videos"):
            # Try to infer exercise type and quality from filename/path
            filename = video_file.name.lower()
            path_parts = str(video_file).lower().split(os.sep)
            
            # Detect exercise type
            exercise_type = 'unknown'
            for exercise in self.exercise_types:
                if exercise in filename or any(exercise in part for part in path_parts):
                    exercise_type = exercise
                    break
            
            # Detect quality
            quality = 'unknown'
            if any(word in filename for word in ['correct', 'good', 'proper']):
                quality = 'good'
            elif any(word in filename for word in ['incorrect', 'bad', 'wrong', 'poor']):
                quality = 'poor'
            
            # Copy to organized structure
            if exercise_type != 'unknown':
                dest_dir = self.processed_path / exercise_type / quality
                dest_file = dest_dir / f"{video_file.stem}_{len(organized_data):04d}.mp4"
                
                try:
                    shutil.copy2(video_file, dest_file)
                    
                    organized_data.append({
                        'original_path': str(video_file),
                        'organized_path': str(dest_file),
                        'exercise_type': exercise_type,
                        'quality_label': quality,
                        'source': 'kaggle_fitness',
                        'filename': dest_file.name
                    })
                except Exception as e:
                    print(f"Error copying {video_file}: {e}")
        
        return organized_data
    
    def organize_ui_prmd_data(self):
        """Organize UI-PRMD dataset"""
        print("Organizing UI-PRMD data...")
        
        ui_prmd_path = self.raw_path / "ui_prmd"
        if not ui_prmd_path.exists():
            print("UI-PRMD dataset not found")
            return []
        
        organized_data = []
        
        # UI-PRMD has specific structure
        exercise_mapping = {
            'squats': 'squat',
            'deadlifts': 'deadlift',
            'pushups': 'pushup',
            'pullups': 'pullup'
        }
        
        for ui_exercise, standard_exercise in exercise_mapping.items():
            exercise_dir = ui_prmd_path / ui_exercise
            if not exercise_dir.exists():
                continue
            
            video_files = list(exercise_dir.glob("*.mp4"))
            
            for video_file in tqdm(video_files, desc=f"Organizing {ui_exercise}"):
                filename = video_file.name.lower()
                
                # Determine quality from filename
                quality = 'good' if 'correct' in filename else 'poor'
                
                # Copy to organized structure
                dest_dir = self.processed_path / standard_exercise / quality
                dest_file = dest_dir / f"ui_prmd_{video_file.stem}_{len(organized_data):04d}.mp4"
                
                try:
                    shutil.copy2(video_file, dest_file)
                    
                    organized_data.append({
                        'original_path': str(video_file),
                        'organized_path': str(dest_file),
                        'exercise_type': standard_exercise,
                        'quality_label': quality,
                        'source': 'ui_prmd',
                        'filename': dest_file.name
                    })
                except Exception as e:
                    print(f"Error copying {video_file}: {e}")
        
        return organized_data
    
    def create_dataset_manifest(self, organized_data):
        """Create manifest file with all dataset information"""
        print("Creating dataset manifest...")
        
        # Create DataFrame
        df = pd.DataFrame(organized_data)
        
        # Add video metadata
        print("Extracting video metadata...")
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing videos"):
            video_path = row['organized_path']
            
            try:
                cap = cv2.VideoCapture(video_path)
                
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    duration = frame_count / fps if fps > 0 else 0
                    
                    df.loc[idx, 'fps'] = fps
                    df.loc[idx, 'frame_count'] = frame_count
                    df.loc[idx, 'width'] = width
                    df.loc[idx, 'height'] = height
                    df.loc[idx, 'duration_seconds'] = duration
                
                cap.release()
                
            except Exception as e:
                print(f"Error analyzing {video_path}: {e}")
                df.loc[idx, 'fps'] = 0
                df.loc[idx, 'frame_count'] = 0
                df.loc[idx, 'width'] = 0
                df.loc[idx, 'height'] = 0
                df.loc[idx, 'duration_seconds'] = 0
        
        # Save manifest
        manifest_path = self.processed_path / "dataset_manifest.csv"
        df.to_csv(manifest_path, index=False)
        
        # Create summary
        summary = {
            'total_videos': len(df),
            'exercises': df['exercise_type'].value_counts().to_dict(),
            'quality_distribution': df['quality_label'].value_counts().to_dict(),
            'sources': df['source'].value_counts().to_dict(),
            'total_duration_hours': df['duration_seconds'].sum() / 3600
        }
        
        summary_path = self.processed_path / "dataset_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Dataset manifest saved: {manifest_path}")
        print(f"Dataset summary saved: {summary_path}")
        
        # Print summary
        print("\n" + "="*50)
        print("DATASET ORGANIZATION SUMMARY")
        print("="*50)
        print(f"Total videos: {summary['total_videos']}")
        print(f"Total duration: {summary['total_duration_hours']:.1f} hours")
        print("\nExercise distribution:")
        for exercise, count in summary['exercises'].items():
            print(f"  {exercise}: {count}")
        print("\nQuality distribution:")
        for quality, count in summary['quality_distribution'].items():
            print(f"  {quality}: {count}")
        
        return df, summary
    
    def organize_all_data(self):
        """Organize all available datasets"""
        print("Starting complete data organization...")
        
        all_organized_data = []
        
        # Organize each dataset
        kaggle_data = self.organize_kaggle_fitness_data()
        all_organized_data.extend(kaggle_data)
        
        ui_prmd_data = self.organize_ui_prmd_data()
        all_organized_data.extend(ui_prmd_data)
        
        # Create manifest
        if all_organized_data:
            df, summary = self.create_dataset_manifest(all_organized_data)
            return df, summary
        else:
            print("No data was organized!")
            return None, None

if __name__ == "__main__":
    organizer = DatasetOrganizer()
    df, summary = organizer.organize_all_data()