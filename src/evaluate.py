"""
Evaluation script for emotion recognition model.
Generates confusion matrix heatmap and training history plots.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow import keras
import os

from data import load_fer2013


def evaluate_model(model_path='emotion_model.h5',
                   csv_path=None,
                   train_dir='data/train',
                   test_dir='data/test',
                   history_path='training_history.json',
                   save_plots=True):
    """
    Evaluate the trained model and generate visualizations.
    
    Args:
        model_path: Path to the trained model file
        csv_path: Path to FER2013 CSV file (optional, if None uses folder structure)
        train_dir: Path to training images directory (used if csv_path is None)
        test_dir: Path to test images directory (used if csv_path is None)
        history_path: Path to training history JSON file
        save_plots: Whether to save plots to files
    """
    print("="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"\nLoading model from {model_path}...")
    model = keras.models.load_model(model_path)
    
    # Load data
    X_train, X_test, y_train, y_test, emotion_labels = load_fer2013(
        csv_path=csv_path,
        train_dir=train_dir,
        test_dir=test_dir
    )
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Predictions
    print("\nGenerating predictions...")
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Confusion Matrix
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    # Plot confusion matrix heatmap
    plt.figure(figsize=(10, 8))
    emotion_names = [emotion_labels[i] for i in range(len(emotion_labels))]
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=emotion_names,
        yticklabels=emotion_names,
        cbar_kws={'label': 'Count'}
    )
    
    plt.title('Confusion Matrix - Emotion Recognition Model', fontsize=16, pad=20)
    plt.xlabel('Predicted Emotion', fontsize=12)
    plt.ylabel('True Emotion', fontsize=12)
    plt.tight_layout()
    
    if save_plots:
        confusion_matrix_path = 'confusion_matrix.png'
        plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {confusion_matrix_path}")
    
    plt.show()
    
    # Classification Report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(
        y_true_classes,
        y_pred_classes,
        target_names=emotion_names,
        digits=4
    ))
    
    # Plot Training History
    if os.path.exists(history_path):
        print(f"\nLoading training history from {history_path}...")
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        # Plot Accuracy vs Epochs
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        axes[0].plot(history['accuracy'], label='Training Accuracy', marker='o')
        axes[0].plot(history['val_accuracy'], label='Validation Accuracy', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Model Accuracy vs Epochs')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss plot
        axes[1].plot(history['loss'], label='Training Loss', marker='o')
        axes[1].plot(history['val_loss'], label='Validation Loss', marker='s')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Model Loss vs Epochs')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            history_plot_path = 'training_history.png'
            plt.savefig(history_plot_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {history_plot_path}")
        
        plt.show()
    else:
        print(f"\nTraining history file not found at {history_path}")
        print("Skipping training history plot.")
    
    # Analyze confusion patterns
    print("\n" + "="*60)
    print("CONFUSION ANALYSIS")
    print("="*60)
    
    # Find most confused emotion pairs
    confusion_pairs = []
    for i in range(len(emotion_labels)):
        for j in range(len(emotion_labels)):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append((
                    emotion_labels[i],
                    emotion_labels[j],
                    cm[i, j]
                ))
    
    confusion_pairs.sort(key=lambda x: x[2], reverse=True)
    
    print("\nTop 10 most confused emotion pairs:")
    for true_emotion, pred_emotion, count in confusion_pairs[:10]:
        print(f"  {true_emotion} → {pred_emotion}: {count} misclassifications")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    # Uses folder structure by default (data/train and data/test)
    evaluate_model(
        model_path='emotion_model.h5',
        csv_path=None,  # Set to 'data/fer2013.csv' if using CSV format
        train_dir='data/train',
        test_dir='data/test',
        history_path='training_history.json'
    )
