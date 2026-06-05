"""
model.py - Defines the neural network model
This is like the "brain" structure that all clients will use
"""

import tensorflow as tf # type: ignore
from tensorflow import keras # type: ignore

def create_model():
    """
    Creates a simple neural network for MNIST digit classification
    MNIST: Handwritten digits (0-9) as 28x28 pixel images
    """
    
    model = keras.Sequential([
        # Input layer: Flatten 28x28 image into 784 numbers
        keras.layers.Flatten(input_shape=(28, 28)),
        
        # Hidden layer: 128 neurons with ReLU activation
        keras.layers.Dense(128, activation='relu'),
        
        # Dropout: Prevents overfitting (randomly turns off 10% of neurons)
        keras.layers.Dropout(0.1),
        
        # Output layer: 10 neurons (one for each digit 0-9)
        keras.layers.Dense(10, activation='softmax')
    ])
    
    # Compile the model (prepare it for training)
    model.compile(
        optimizer='adam',           # Optimization algorithm
        loss='sparse_categorical_crossentropy',  # Loss function for classification
        metrics=['accuracy']        # Track accuracy during training
    )
    
    return model

# Simple test to verify model works
if __name__ == "__main__":
    model = create_model()
    model.summary()  # Prints the model architecture
    print("✓ Model created successfully!")