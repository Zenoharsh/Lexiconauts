"""
Dataset Downloader for Sports Analysis
File: src/preprocessing/dataset_downloader.py
"""

import os
import zipfile
import requests
import kaggle
from pathlib import Path
from tqdm import tqdm
import subprocess
import shutil

class DatasetDownloader:
    def __init__(self, base_path="./datasets/raw_data"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
    def download_kaggle_fitness_dataset(self):
        """Download main fitness pose dataset from Kaggle"""
        print("Downloading Kaggle Fitness Dataset...")
        
        dataset_path = self.base_path / "kaggle_fitness"
        dataset_path.mkdir(exist_ok=True)
        
        try:
            # Download workout fitness dataset
            kaggle.api.dataset_download_files(
                'hasyimabdillah/workoutfitness-computer-vision',
                path=str(dataset_path),
                unzip=True
            )
            print(f"✓ Kaggle fitness dataset downloaded to: {dataset_path}")
            
            # Also try alternative fitness datasets
            alt_datasets = [
                'bharathsharma/gym-workout-detection-dataset',
                'shashwatwork/pose-classification-dataset'
            ]
            
            for alt_dataset in alt_datasets:
                try:
                    alt_path = self.base_path / alt_dataset.split('/')[-1]
                    alt_path.mkdir(exist_ok=True)
                    kaggle.api.dataset_download_files(
                        alt_dataset,
                        path=str(alt_path),
                        unzip=True
                    )
                    print(f"✓ Downloaded alternative dataset: {alt_dataset}")
                except Exception as e:
                    print(f"✗ Could not download {alt_dataset}: {e}")
            
            return True
            
        except Exception as e:
            print(f"✗ Error downloading Kaggle dataset: {e}")
            return False
    
    def download_ui_prmd_dataset(self):
        """Download UI-PRMD exercise dataset from GitHub"""
        print("Downloading UI-PRMD Dataset...")
        
        dataset_path = self.base_path / "ui_prmd"
        
        if dataset_path.exists():
            print(f"UI-PRMD dataset already exists at: {dataset_path}")
            return True
        
        try:
            # Clone the repository
            subprocess.run([
                'git', 'clone', 
                'https://github.com/abdullahmahmood/UI-PRMD.git',
                str(dataset_path)
            ], check=True)
            
            print(f"✓ UI-PRMD dataset downloaded to: {dataset_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Git clone failed: {e}")
            
            # Fallback: try direct download
            try:
                url = "https://github.com/abdullahmahmood/UI-PRMD/archive/main.zip"
                self._download_and_extract(url, dataset_path, "UI-PRMD-main")
                return True
            except Exception as e2:
                print(f"✗ Fallback download failed: {e2}")
                return False
    
    def download_penn_action_dataset(self):
        """Download Penn Action dataset"""
        print("Downloading Penn Action Dataset...")
        
        dataset_path = self.base_path / "penn_action"
        dataset_path.mkdir(exist_ok=True)
        
        try:
            url = "http://dreamdragon.github.io/PennAction/Datasets/Penn_Action.tar.gz"
            tar_file = dataset_path / "Penn_Action.tar.gz"
            
            # Download
            print("Downloading Penn Action (this may take a while)...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(tar_file, 'wb') as f, tqdm(
                desc="Penn Action",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
            
            # Extract
            print("Extracting Penn Action dataset...")
            subprocess.run(['tar', '-xzf', str(tar_file), '-C', str(dataset_path)], check=True)
            
            # Clean up
            tar_file.unlink()
            
            print(f"✓ Penn Action dataset downloaded to: {dataset_path}")
            return True
            
        except Exception as e:
            print(f"✗ Error downloading Penn Action: {e}")
            return False
    
    def download_sample_videos(self):
        """Download sample videos for testing"""
        print("Downloading sample test videos...")
        
        sample_path = Path("./data_samples/videos")
        sample_path.mkdir(parents=True, exist_ok=True)
        
        # Sample video URLs (you can add more)
        sample_urls = {
            "pushup_demo.mp4": "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4",
            "squat_demo.mp4": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"
        }
        
        for filename, url in sample_urls.items():
            try:
                file_path = sample_path / filename
                if not file_path.exists():
                    response = requests.get(url, stream=True)
                    response.raise_for_status()
                    
                    with open(file_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    print(f"✓ Downloaded sample: {filename}")
            except Exception as e:
                print(f"✗ Could not download {filename}: {e}")
    
    def _download_and_extract(self, url, extract_path, folder_name):
        """Helper function to download and extract zip files"""
        zip_file = extract_path.parent / "temp_download.zip"
        
        # Download
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(zip_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Extract
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_path.parent)
        
        # Move to correct location
        extracted_folder = extract_path.parent / folder_name
        if extracted_folder.exists():
            shutil.move(str(extracted_folder), str(extract_path))
        
        # Clean up
        zip_file.unlink()
    
    def download_all_datasets(self):
        """Download all available datasets"""
        print("Starting complete dataset download...")
        
        results = {}
        
        results['kaggle_fitness'] = self.download_kaggle_fitness_dataset()
        results['ui_prmd'] = self.download_ui_prmd_dataset()
        results['penn_action'] = self.download_penn_action_dataset()
        self.download_sample_videos()
        
        # Summary
        print("\n" + "="*50)
        print("DATASET DOWNLOAD SUMMARY")
        print("="*50)
        
        for dataset, success in results.items():
            status = "✓ SUCCESS" if success else "✗ FAILED"
            print(f"{dataset:<20}: {status}")
        
        successful_downloads = sum(results.values())
        print(f"\nTotal successful downloads: {successful_downloads}/{len(results)}")
        
        return results

if __name__ == "__main__":
    downloader = DatasetDownloader()
    downloader.download_all_datasets()