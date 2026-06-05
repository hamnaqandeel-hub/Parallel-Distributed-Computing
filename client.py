"""
client.py - Each client trains the model on their local data
Each client represents one "student" with their own data partition
"""

import numpy as np
import tensorflow as tf # type: ignore
from tensorflow import keras # type: ignore
import flwr as fl # type: ignore

# Import our model creation function
from model import create_model

class FlowerClient(fl.client.NumPyClient):
    """
    This class defines how each client:
    1. Gets their data
    2. Trains the model
    3. Sends updates back to server
    """
    
    def __init__(self, client_id, x_train, y_train, x_val, y_val):
        """
        Initialize the client with their specific data partition
        
        Args:
            client_id: Which client number (0, 1, 2, 3, 4)
            x_train, y_train: Training images and labels for this client
            x_val, y_val: Validation data for testing
        """
        self.client_id = client_id
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.model = create_model()
        
        print(f"✓ Client {client_id} initialized with {len(x_train)} training samples")
    
    def get_parameters(self, config):
        """
        Return the current model weights (parameters)
        Server asks for these to aggregate them
        """
        return self.model.get_weights()
    
    def fit(self, parameters, config):
        """
        Train the model on local data
        
        Args:
            parameters: Current global model weights from server
            config: Training configuration (epochs, batch size, etc.)
        
        Returns:
            Updated model weights, number of samples, training metrics
        """
        # Set the model weights to the ones received from server
        self.model.set_weights(parameters)
        
        # Training settings
        epochs = config.get("local_epochs", 5)      # Default: 5 epochs
        batch_size = config.get("batch_size", 32)   # Default: 32 samples per batch
        
        # Train the model
        history = self.model.fit(
            self.x_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.x_val, self.y_val),
            verbose=0  # 0 = silent, 1 = progress bar, 2 = one line per epoch
        )
        
        # Get the last epoch's metrics
        loss = history.history['loss'][-1]
        accuracy = history.history['accuracy'][-1]
        val_loss = history.history['val_loss'][-1]
        val_accuracy = history.history['val_accuracy'][-1]
        
        print(f"  Client {self.client_id} - Loss: {loss:.4f}, Acc: {accuracy:.4f}, Val Acc: {val_accuracy:.4f}")
        
        # Return updated weights, number of samples, and metrics
        return self.model.get_weights(), len(self.x_train), {
            "loss": loss,
            "accuracy": accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        }
    
    def evaluate(self, parameters, config):
        """
        Evaluate the model on local validation data
        Server uses this to check global model performance
        """
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_val, self.y_val, verbose=0)
        print(f"  Client {self.client_id} Evaluation - Loss: {loss:.4f}, Acc: {accuracy:.4f}")
        return loss, len(self.x_val), {"accuracy": accuracy}


def load_mnist_partitions(num_clients=5):
    """
    Load MNIST dataset and split it among clients
    Each client gets different (non-overlapping) data - simulates real FL
    
    Args:
        num_clients: Number of clients to split data into
    
    Returns:
        List of (x_train, y_train, x_val, y_val) for each client
    """
    print("\n📊 Loading MNIST dataset...")
    
    # Load MNIST dataset
    (x_train_all, y_train_all), (x_test_all, y_test_all) = keras.datasets.mnist.load_data()
    
    # Normalize pixel values to range [0, 1] (from 0-255)
    x_train_all = x_train_all.astype('float32') / 255.0
    x_test_all = x_test_all.astype('float32') / 255.0
    
    print(f"Total training samples: {len(x_train_all)}")
    print(f"Total test samples: {len(x_test_all)}")
    
    # Split training data among clients
    # Each client gets an equal portion of the training data
    samples_per_client = len(x_train_all) // num_clients
    
    client_datasets = []
    
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = (i + 1) * samples_per_client if i < num_clients - 1 else len(x_train_all)
        
        # Get this client's training data
        x_client_train = x_train_all[start_idx:end_idx]
        y_client_train = y_train_all[start_idx:end_idx]
        
        # Use a portion of test data as validation for this client
        val_samples = len(x_test_all) // num_clients
        val_start = i * val_samples
        val_end = (i + 1) * val_samples if i < num_clients - 1 else len(x_test_all)
        
        x_client_val = x_test_all[val_start:val_end]
        y_client_val = y_test_all[val_start:val_end]
        
        client_datasets.append((x_client_train, y_client_train, x_client_val, y_client_val))
        
        print(f"  Client {i}: {len(x_client_train)} training, {len(x_client_val)} validation samples")
    
    return client_datasets


def main():
    """
    Main function to start a Flower client
    """
    print("=" * 50)
    print("🌟 FEDERATED LEARNING CLIENT 🌟")
    print("=" * 50)
    
    # Get client ID from command line argument
    import sys
    if len(sys.argv) != 2:
        print("Usage: python client.py <client_id>")
        print("Example: python client.py 0")
        sys.exit(1)
    
    client_id = int(sys.argv[1])
    
    # Load data partitions
    all_clients_data = load_mnist_partitions(num_clients=5)
    
    # Get this client's data
    x_train, y_train, x_val, y_val = all_clients_data[client_id]
    
    # Create and start the Flower client
    client = FlowerClient(client_id, x_train, y_train, x_val, y_val)
    
    # Start client (connect to server on localhost:8080)
    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=client
    )
    
    print(f"\n✓ Client {client_id} finished!")


if __name__ == "__main__":
    main()