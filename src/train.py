"""
Training script for emotion recognition model.
Implements data augmentation, callbacks, and training loop.
"""

import os
import json
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    CSVLogger
)

from data import load_fer2013
from models import create_emotion_model


def train_model(csv_path=None,
                train_dir='data/train',
                test_dir='data/test',
                model_save_path='emotion_model.h5',
                history_save_path='training_history.json',
                epochs=100,
                batch_size=64):
    """
    Train the emotion recognition model.
    
    Args:
        csv_path: Path to FER2013 CSV file (optional, if None uses folder structure)
        train_dir: Path to training images directory (used if csv_path is None)
        test_dir: Path to test images directory (used if csv_path is None)
        model_save_path: Path to save the best model
        history_save_path: Path to save training history
        epochs: Number of training epochs
        batch_size: Batch size for training
    """
    print("="*60)
    print("EMOTION RECOGNITION MODEL TRAINING")
    print("="*60)
    
    # Load data
    X_train, X_test, y_train, y_test, emotion_labels = load_fer2013(
        csv_path=csv_path,
        train_dir=train_dir,
        test_dir=test_dir
    )
    
    # Create model
    print("\nCreating model...")
    model = create_emotion_model(input_shape=(48, 48, 1), num_classes=7)
    
    # Compile model
    print("\nCompiling model...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Data augmentation
    print("\nSetting up data augmentation...")
    train_datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        fill_mode='nearest'
    )
    
    # For validation/test data, only rescale (no augmentation)
    test_datagen = ImageDataGenerator()
    
    # Create data generators
    train_generator = train_datagen.flow(
        X_train, y_train,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Callbacks
    print("\nSetting up callbacks...")
    callbacks = [
        # Reduce learning rate when validation loss plateaus
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=5,
            verbose=1,
            min_lr=1e-7
        ),
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            verbose=1,
            restore_best_weights=True
        ),
        # Save best model
        ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1,
            mode='min'
        ),
        # Log training history to CSV
        CSVLogger('training_log.csv', append=False)
    ]
    
    print(f"\nCallbacks configured:")
    print(f"  - ReduceLROnPlateau: monitor='val_loss', patience=5")
    print(f"  - EarlyStopping: monitor='val_loss', patience=15")
    print(f"  - ModelCheckpoint: saving best model to '{model_save_path}'")
    print(f"  - CSVLogger: logging to 'training_log.csv'")
    
    # Training
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_test.shape[0]}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print("="*60 + "\n")
    
    history = model.fit(
        train_generator,
        steps_per_epoch=len(X_train) // batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Save training history
    print(f"\nSaving training history to {history_save_path}...")
    history_dict = {
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']]
    }
    
    with open(history_save_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print(f"Best model saved to: {model_save_path}")
    print(f"Training history saved to: {history_save_path}")
    
    # Evaluate final model
    print("\nEvaluating final model on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    # Train the model
    # Uses folder structure by default (data/train and data/test)
    train_model(
        csv_path=None,  # Set to 'data/fer2013.csv' if using CSV format
        train_dir='data/train',
        test_dir='data/test',
        epochs=100,
        batch_size=64
    )
