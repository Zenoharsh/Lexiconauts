"""
Kaggle API Setup Helper
File: src/utils/kaggle_setup.py
"""

import os
import json
from pathlib import Path

def setup_kaggle_credentials():
    """Setup Kaggle API credentials"""
    print("Setting up Kaggle API...")
    
    # Check if credentials exist
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    if not kaggle_json.exists():
        print("Kaggle credentials not found!")
        print("Please follow these steps:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token'")
        print("3. Download kaggle.json")
        print("4. Place it in:", kaggle_dir)
        
        # Create directory if it doesn't exist
        kaggle_dir.mkdir(exist_ok=True)
        
        # Create template
        template = {
            "username": "your_kaggle_username",
            "key": "your_kaggle_key"
        }
        
        with open(kaggle_json, 'w') as f:
            json.dump(template, f, indent=2)
        
        print(f"Template created at: {kaggle_json}")
        print("Please edit this file with your actual credentials")
        return False
    
    # Set permissions (Linux/Mac)
    try:
        os.chmod(kaggle_json, 0o600)
    except:
        pass
    
    print("Kaggle API credentials found!")
    return True

if __name__ == "__main__":
    setup_kaggle_credentials()