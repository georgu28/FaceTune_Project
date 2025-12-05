"""
Data loader for FER2013 dataset.
Loads images from folder structure: data/train/ and data/test/ with emotion-labeled subfolders.
Converts images to 48x48x1 numpy arrays and normalizes them.
One-hot encodes emotion labels based on folder names.
"""

import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
import os
from pathlib import Path


def load_fer2013_from_folders(train_dir='data/train', test_dir='data/test', target_size=(48, 48)):
    """
    Load FER2013 dataset from folder structure.
    
    Args:
        train_dir: Path to training images directory
        test_dir: Path to test images directory
        target_size: Target size for images (width, height)
        
    Returns:
        X_train, X_test, y_train, y_test: Training and test data splits
        emotion_labels: Dictionary mapping emotion indices to names
    """
    # Define emotion labels mapping (FER2013 uses 0-6)
    emotion_labels = {
        0: 'Angry',
        1: 'Disgust',
        2: 'Fear',
        3: 'Happy',
        4: 'Sad',
        5: 'Surprise',
        6: 'Neutral'
    }
    
    # Create reverse mapping from folder name to emotion index (case-insensitive)
    folder_to_emotion = {
        'angry': 0,
        'disgust': 1,
        'fear': 2,
        'happy': 3,
        'sad': 4,
        'surprise': 5,
        'neutral': 6
    }
    
    def load_images_from_folder(folder_path, emotion_idx, target_size):
        """Load all images from a folder and assign emotion label."""
        images = []
        labels = []
        
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist, skipping...")
            return images, labels
        
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"  Loading {len(image_files)} images from {os.path.basename(folder_path)}...")
        
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            try:
                # Read image in grayscale
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    print(f"Warning: Could not read {img_path}, skipping...")
                    continue
                
                # Resize to target size if necessary
                if img.shape[:2] != target_size[::-1]:  # cv2 uses (width, height), but shape is (height, width)
                    img = cv2.resize(img, target_size)
                
                # Normalize pixel values to be between 0 and 1
                img = img.astype(np.float32) / 255.0
                
                # Reshape to (height, width, 1) for grayscale
                img = img.reshape(target_size[1], target_size[0], 1)
                
                images.append(img)
                labels.append(emotion_idx)
                
            except Exception as e:
                print(f"Error loading {img_path}: {e}, skipping...")
                continue
        
        return images, labels
    
    # Load training images
    print("="*60)
    print("Loading training images...")
    print("="*60)
    X_train = []
    y_train = []
    
    train_path = Path(train_dir)
    if not train_path.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    
    for emotion_name, emotion_idx in folder_to_emotion.items():
        emotion_folder = train_path / emotion_name
        images, labels = load_images_from_folder(emotion_folder, emotion_idx, target_size)
        X_train.extend(images)
        y_train.extend(labels)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Load test images
    print("\n" + "="*60)
    print("Loading test images...")
    print("="*60)
    X_test = []
    y_test = []
    
    test_path = Path(test_dir)
    if not test_path.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    for emotion_name, emotion_idx in folder_to_emotion.items():
        emotion_folder = test_path / emotion_name
        images, labels = load_images_from_folder(emotion_folder, emotion_idx, target_size)
        X_test.extend(images)
        y_test.extend(labels)
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    # One-hot encode the emotion labels (7 classes)
    print("\nOne-hot encoding emotion labels...")
    y_train_categorical = to_categorical(y_train, num_classes=7)
    y_test_categorical = to_categorical(y_test, num_classes=7)
    
    # Print statistics
    print(f"\n" + "="*60)
    print("Dataset loaded successfully!")
    print("="*60)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Image shape: {X_train.shape[1:]}")
    print(f"Number of classes: {len(emotion_labels)}")
    
    print(f"\nEmotion distribution in training set:")
    for emotion_idx, emotion_name in emotion_labels.items():
        count = np.sum(y_train == emotion_idx)
        print(f"  {emotion_name}: {count}")
    
    print(f"\nEmotion distribution in test set:")
    for emotion_idx, emotion_name in emotion_labels.items():
        count = np.sum(y_test == emotion_idx)
        print(f"  {emotion_name}: {count}")
    
    return X_train, X_test, y_train_categorical, y_test_categorical, emotion_labels


# Keep the old function name for backward compatibility
def load_fer2013(csv_path=None, train_dir='data/train', test_dir='data/test'):
    """
    Load FER2013 dataset. 
    Supports both CSV format and folder structure.
    
    Args:
        csv_path: Path to the FER2013 CSV file (optional, if None uses folder structure)
        train_dir: Path to training images directory (used if csv_path is None)
        test_dir: Path to test images directory (used if csv_path is None)
        
    Returns:
        X_train, X_test, y_train, y_test: Training and test data splits
        emotion_labels: Dictionary mapping emotion indices to names
    """
    # If csv_path is not provided or doesn't exist, use folder structure
    if csv_path is None or not os.path.exists(csv_path):
        if csv_path is not None:
            print(f"CSV file not found at {csv_path}. Using folder structure instead.")
        return load_fer2013_from_folders(train_dir=train_dir, test_dir=test_dir)
    
    # Otherwise, use CSV loading (keeping old implementation for backward compatibility)
    import pandas as pd
    
    print(f"Loading FER2013 dataset from CSV: {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Define emotion labels mapping (FER2013 uses 0-6)
    emotion_labels = {
        0: 'Angry',
        1: 'Disgust',
        2: 'Fear',
        3: 'Happy',
        4: 'Sad',
        5: 'Surprise',
        6: 'Neutral'
    }
    
    # Convert pixels column (string of space-separated integers) to numpy arrays
    print("Converting pixels to numpy arrays...")
    X = []
    for pixels in df['pixels']:
        # Convert string of space-separated integers to 48x48 array
        pixel_array = np.array([int(pixel) for pixel in pixels.split()], dtype=np.float32)
        pixel_array = pixel_array.reshape(48, 48, 1)  # Reshape to 48x48x1 (grayscale)
        X.append(pixel_array)
    
    X = np.array(X)
    
    # Normalize pixel values to be between 0 and 1
    print("Normalizing pixel values...")
    X = X / 255.0
    
    # Get emotion labels
    y = df['emotion'].values
    
    # One-hot encode the emotion labels (7 classes)
    print("One-hot encoding emotion labels...")
    y_categorical = to_categorical(y, num_classes=7)
    
    # Split data based on 'Usage' column
    # 'Training' -> train set
    # 'PrivateTest' or 'PublicTest' -> test set
    train_mask = df['Usage'] == 'Training'
    test_mask = (df['Usage'] == 'PrivateTest') | (df['Usage'] == 'PublicTest')
    
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y_categorical[train_mask]
    y_test = y_categorical[test_mask]
    
    print(f"\nDataset loaded successfully!")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Image shape: {X_train.shape[1:]}")
    print(f"Number of classes: {len(emotion_labels)}")
    print(f"\nEmotion distribution in training set:")
    for emotion_idx, emotion_name in emotion_labels.items():
        count = np.sum(y[train_mask] == emotion_idx)
        print(f"  {emotion_name}: {count}")
    
    return X_train, X_test, y_train, y_test, emotion_labels


if __name__ == "__main__":
    # Test the data loader
    try:
        X_train, X_test, y_train, y_test, emotion_labels = load_fer2013()
        print(f"\n" + "="*60)
        print("Data Summary:")
        print("="*60)
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_test shape: {y_test.shape}")
        print(f"\nImage shape: {X_train.shape[1:]}")
        print(f"Pixel value range: [{X_train.min():.2f}, {X_train.max():.2f}]")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()