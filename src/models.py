"""
CNN Model for FER2013 Emotion Recognition.
VGG-style architecture for 48x48 grayscale images.
"""

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model


def create_emotion_model(input_shape=(48, 48, 1), num_classes=7):
    """
    Create a VGG-style CNN for emotion recognition.
    
    Architecture:
    - Four Convolutional blocks, each with:
        * 2 Conv2D layers (with batch normalization and ELU activation)
        * MaxPooling2D
        * Dropout (0.25)
    - Flatten layer
    - Two Dense layers (512 units, ELU activation, Dropout 0.5)
    - Output Dense layer with 7 units and softmax activation
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of emotion classes (7 for FER2013)
        
    Returns:
        model: Compiled Keras model
    """
    inputs = layers.Input(shape=input_shape)
    
    # First Convolutional Block
    x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    x = layers.Conv2D(32, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Second Convolutional Block
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Third Convolutional Block
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Fourth Convolutional Block
    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Flatten
    x = layers.Flatten()(x)
    
    # First Dense layer
    x = layers.Dense(512, activation='elu')(x)
    x = layers.Dropout(0.5)(x)
    
    # Second Dense layer
    x = layers.Dense(512, activation='elu')(x)
    x = layers.Dropout(0.5)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='emotion_recognition_model')
    
    return model


if __name__ == "__main__":
    # Create and print model summary
    print("Creating emotion recognition model...")
    model = create_emotion_model(input_shape=(48, 48, 1), num_classes=7)
    print("\n" + "="*50)
    print("Model Summary:")
    print("="*50)
    model.summary()
    
    # Count parameters
    trainable_params = sum([keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = sum([keras.backend.count_params(w) for w in model.non_trainable_weights])
    print(f"\nTotal trainable parameters: {trainable_params:,}")
    print(f"Total non-trainable parameters: {non_trainable_params:,}")
