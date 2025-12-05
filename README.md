# Emotion Recognition using FER2013 Dataset

A deep learning project for facial expression recognition using Convolutional Neural Networks (CNN). This project classifies emotions from 48x48 grayscale images into 7 categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

## Project Overview

This project implements a complete pipeline for emotion recognition:
- **Data Loading**: Handles FER2013 CSV format with pixel normalization
- **Model Architecture**: VGG-style CNN optimized for 48x48 grayscale images
- **Training**: Data augmentation and advanced callbacks to prevent overfitting
- **Evaluation**: Confusion matrix visualization and performance metrics
- **Real-time Inference**: Webcam-based emotion detection

## Dataset

The project uses the **FER2013** (Facial Expression Recognition 2013) dataset:
- **Input**: 48x48 pixel grayscale images
- **Classes**: 7 emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
- **Format**: CSV file with columns: `emotion`, `pixels`, `Usage`
- **Download**: Available on [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)

## Project Structure

```
FaceTune_Project/
├── data/
│   ├── fer2013.csv          # FER2013 dataset (not included, download separately)
│   ├── train/               # Alternative: organized image folders
│   └── test/
├── src/
│   ├── data.py              # Data loading and preprocessing
│   ├── models.py            # CNN model architecture
│   ├── train.py             # Training script with augmentation
│   ├── evaluate.py          # Evaluation and visualization
│   ├── infer.py             # Real-time webcam inference with music
│   └── music_player.py      # Music playback based on emotions
├── music/                   # Emotion-specific music folders
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── sad/
│   ├── surprise/
│   └── neutral/
├── notebooks/
│   └── explore.ipynb        # Data exploration notebook
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd FaceTune_Project
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `tensorflow>=2.13.0`
- `opencv-python>=4.8.0`
- `pandas>=2.0.0`
- `numpy>=1.24.0`
- `matplotlib>=3.7.0`
- `scikit-learn>=1.3.0`
- `seaborn>=0.12.0`
- `pygame>=2.5.0` (for music playback)

### 4. Download FER2013 Dataset

Download the FER2013 dataset from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) and place the CSV file in the `data/` directory:

```bash
data/fer2013.csv
```

## Usage

### 1. Data Loading

Test the data loader:

```bash
python src/data.py
```

This will load and preprocess the FER2013 dataset, showing data statistics.

### 2. Model Architecture

View the model architecture:

```bash
python src/models.py
```

This will create the model and print a summary showing all layers and parameters.

### 3. Training

Train the emotion recognition model:

```bash
python src/train.py
```

Training configuration:
- **Epochs**: 100 (with early stopping)
- **Batch Size**: 64
- **Data Augmentation**: Rotation, zoom, flip, shift
- **Callbacks**:
  - `ReduceLROnPlateau`: Reduces learning rate when validation loss plateaus
  - `EarlyStopping`: Stops training if no improvement for 15 epochs
  - `ModelCheckpoint`: Saves best model to `emotion_model.h5`
  - `CSVLogger`: Logs training metrics to `training_log.csv`

The best model will be saved as `emotion_model.h5` and training history as `training_history.json`.

### 4. Evaluation

Evaluate the trained model and generate visualizations:

```bash
python src/evaluate.py
```

This will:
- Evaluate model on test set
- Generate confusion matrix heatmap (`confusion_matrix.png`)
- Plot training history (accuracy and loss vs epochs)
- Show classification report
- Analyze confusion patterns between emotions

### 5. Real-time Inference with Music Playback

Run emotion detection using webcam with automatic music playback:

```bash
python src/infer.py
```

**Controls**:
- `q` - Quit application
- `p` - Pause/Resume music
- `+` - Increase volume
- `-` - Decrease volume
- `n` - Next track
- `s` - Stop music

**Features**:
- Real-time face detection using Haar Cascade classifier
- Emotion prediction with confidence scores
- Color-coded bounding boxes for different emotions
- **Automatic music playback** based on detected emotions
- Smooth music transitions when emotions change
- Music status displayed on video feed

**Requirements**:
- Webcam connected to your computer
- Trained model file (`emotion_model.h5`)
- Music files in `music/` directory (optional, see below)

**Music Setup**:
1. Create emotion-specific folders in the `music/` directory:
   - `music/angry/` - Aggressive, intense music
   - `music/disgust/` - Dark, ambient music
   - `music/fear/` - Tense, suspenseful music
   - `music/happy/` - Upbeat, cheerful music
   - `music/sad/` - Melancholic, emotional music
   - `music/surprise/` - Energetic, exciting music
   - `music/neutral/` - Calm, ambient music

2. Add music files (MP3, WAV, OGG, M4A, FLAC) to the appropriate folders
3. The application will automatically play music matching the detected emotion
4. See `music/README.md` for more details

## Model Architecture

The CNN model follows a VGG-style architecture:

1. **Four Convolutional Blocks**:
   - Each block contains 2 Conv2D layers with Batch Normalization and ELU activation
   - MaxPooling2D for dimensionality reduction
   - Dropout (0.25) to prevent overfitting
   - Feature maps: 32 → 64 → 128 → 256

2. **Fully Connected Layers**:
   - Flatten layer
   - Two Dense layers (512 units each, ELU activation, Dropout 0.5)
   - Output layer: 7 units with softmax activation

**Total Parameters**: ~3-4 million trainable parameters

## Training Features

### Data Augmentation

To prevent overfitting on the relatively small FER2013 dataset:

- **Rotation Range**: ±10 degrees
- **Zoom Range**: ±10%
- **Horizontal Flip**: Randomly flip images
- **Width/Height Shift**: ±10% shift range

### Callbacks

- **ReduceLROnPlateau**: Monitors validation loss, reduces learning rate by factor of 0.1 when plateaus (patience=5)
- **EarlyStopping**: Stops training early if validation loss doesn't improve (patience=15), restores best weights
- **ModelCheckpoint**: Saves the best model based on validation loss

## Results

After training, you should expect:

- **Test Accuracy**: Typically 60-65% (FER2013 is a challenging dataset)
- **Common Confusions**: 
  - Fear ↔ Surprise (similar facial expressions)
  - Sad ↔ Neutral
  - Disgust ↔ Angry

The confusion matrix visualization helps identify which emotions are frequently confused.

## Project Phases

This project was implemented following these phases:

1. **Phase 1: Setup & Data Loading**
   - Project structure setup
   - Requirements.txt with dependencies
   - CSV data loader with normalization and one-hot encoding

2. **Phase 2: Model Architecture**
   - VGG-style CNN for 48x48 grayscale images
   - Batch normalization and ELU activation
   - Dropout layers for regularization

3. **Phase 3: Training & Augmentation**
   - ImageDataGenerator for data augmentation
   - Adam optimizer with learning rate scheduling
   - Callbacks for training optimization

4. **Phase 4: Evaluation**
   - Confusion matrix heatmap
   - Training history visualization
   - Classification metrics

5. **Phase 5: Real-time Inference**
   - OpenCV webcam integration
   - Haar Cascade face detection
   - Live emotion prediction with confidence scores

## Troubleshooting

### Common Issues

1. **Model file not found**
   - Make sure you've trained the model first using `train.py`
   - Check that `emotion_model.h5` exists in the project root

2. **FER2013 CSV not found**
   - Download the dataset from Kaggle
   - Place `fer2013.csv` in the `data/` directory

3. **Webcam not working**
   - Check camera permissions
   - Try different camera indices: `python src/infer.py --camera_index 1`
   - Ensure OpenCV is properly installed

4. **Out of memory errors**
   - Reduce batch size in `train.py` (default: 64)
   - Reduce image resolution (modify preprocessing)

## Music Playback Features

The application includes intelligent music playback that responds to detected emotions:

- **Automatic Emotion-to-Music Mapping**: Each emotion category has its own music folder
- **Smooth Transitions**: 3-second delay prevents rapid music switching
- **Confidence Threshold**: Only switches music when confidence is above threshold (default: 50%)
- **Random Playlist**: Tracks are shuffled for variety
- **User Controls**: Pause, volume control, skip tracks, stop music
- **Visual Feedback**: Current track displayed on video feed

## Future Improvements

- [x] Music playback based on emotions
- [ ] Fine-tuning with transfer learning
- [ ] Support for multiple face detection in single frame
- [ ] Emotion timeline tracking
- [ ] Web interface for emotion detection
- [ ] Model ensemble for improved accuracy
- [ ] Support for video file processing
- [ ] Spotify/Apple Music integration
- [ ] Custom emotion-to-music mapping configuration

## License

This project is for educational purposes. Please refer to the FER2013 dataset license for data usage terms.

## Acknowledgments

- FER2013 dataset creators
- TensorFlow/Keras team
- OpenCV community

## Contact

For questions or issues, please open an issue on the repository.

