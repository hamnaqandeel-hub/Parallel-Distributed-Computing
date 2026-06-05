"""
centralized.py - Traditional (centralized) training
This trains on ALL data at once (not federated)
We'll compare this with federated learning
"""

import tensorflow as tf  # type: ignore
from tensorflow import keras # type: ignore
import numpy as np
import matplotlib.pyplot as plt # type: ignore
import time
from model import create_model

def train_centralized():
    """
    Train a model on all MNIST data (no privacy, all data in one place)
    """
    print("=" * 50)
    print("🏛️ CENTRALIZED TRAINING (Traditional Approach)")
    print("=" * 50)
    
    # Load full MNIST dataset
    print("\n📊 Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize pixel values
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    
    # Create model
    print("\n🏗️ Creating model...")
    model = create_model()
    model.summary()
    
    # Train the model
    print("\n🏋️ Training model...")
    start_time = time.time()
    
    history = model.fit(
        x_train, y_train,
        epochs=5,                    # Train for 20 epochs
        batch_size=32,
        validation_data=(x_test, y_test),
        verbose=1                     # Show progress bars
    )
    
    training_time = time.time() - start_time
    print(f"\n✅ Training completed in {training_time:.2f} seconds")
    
    # Evaluate final model
    print("\n📊 Evaluating final model...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Save training history
    return {
        'history': history,
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'training_time': training_time,
        'epochs': history.history['accuracy'],
        'val_epochs': history.history['val_accuracy']
    }


def plot_centralized_results(results):
    """
    Plot the training progress for centralized approach
    """
    history = results['history']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Centralized Training - Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss', marker='o')
    ax2.plot(history.history['val_loss'], label='Validation Loss', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Centralized Training - Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('centralized_training_plot.png', dpi=150)
    plt.show()
    print("✓ Centralized training plot saved as 'centralized_training_plot.png'")
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 CENTRALIZED TRAINING SUMMARY")
    print("=" * 50)
    print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Training Time: {results['training_time']:.2f} seconds")


if __name__ == "__main__":
    # Run centralized training
    results = train_centralized()
    plot_centralized_results(results)