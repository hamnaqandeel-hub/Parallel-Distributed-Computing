"""
server.py - Central server that coordinates federated learning
The server:
1. Sends initial model to all clients
2. Collects updated weights from clients
3. Averages them together (FedAvg or FedProx)
4. Repeats for multiple rounds
"""

import flwr as fl # type: ignore
import numpy as np
import matplotlib.pyplot as plt # type: ignore
from model import create_model

# Global list to store accuracy after each round
round_accuracies = []


class SaveMetricsStrategy(fl.server.strategy.FedAvg):
    """
    Custom strategy that saves accuracy after each round for plotting
    """
    
    def aggregate_evaluate(self, server_round, results, failures):
        """
        Called after each round of evaluation
        """
        # Call parent method to get standard metrics
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)
        
        # Calculate average accuracy from all clients
        if metrics_aggregated is not None:
            accuracies = [r.metrics["accuracy"] for _, r in results if "accuracy" in r.metrics]
            if accuracies:
                avg_accuracy = np.mean(accuracies)
                round_accuracies.append(avg_accuracy)
                print(f"\n📊 Round {server_round} - Average Accuracy: {avg_accuracy:.4f}")
        
        return loss_aggregated, metrics_aggregated


def create_fedavg_strategy(num_rounds=10):
    """
    Create FedAvg (Federated Averaging) strategy
    This is the standard approach: average all client weights
    """
    # Create initial model to get its parameters
    initial_model = create_model()
    initial_parameters = fl.common.ndarrays_to_parameters(initial_model.get_weights())
    
    strategy = SaveMetricsStrategy(
        # Fraction of clients to use each round
        fraction_fit=1.0,        # Use all available clients
        fraction_evaluate=1.0,    # Evaluate on all clients
        
        # Minimum number of clients
        min_fit_clients=2,        # Need at least 2 clients to train
        min_evaluate_clients=2,   # Need at least 2 clients to evaluate
        min_available_clients=2,  # Need at least 2 clients available
        
        # Use initial model parameters
        initial_parameters=initial_parameters,
        
        # Client training configuration
        on_fit_config_fn=lambda round: {
            "local_epochs": 5,     # Each client trains for 5 epochs per round
            "batch_size": 32,      # Batch size for training
        },
        
        # Server evaluation configuration
        on_evaluate_config_fn=lambda round: {
            "batch_size": 32,      # Batch size for evaluation
        }
    )
    
    return strategy


def create_fedprox_strategy(num_rounds=10, proximal_mu=0.1):
    """
    Create FedProx strategy (extension of FedAvg with proximal term)
    FedProx adds a penalty to prevent clients from drifting too far
    
    Args:
        proximal_mu: Proximal term strength (higher = less drift)
    """
    from flwr.server.strategy import FedProx # type: ignore
    
    # Create initial model
    initial_model = create_model()
    initial_parameters = fl.common.ndarrays_to_parameters(initial_model.get_weights())
    
    strategy = FedProx(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        initial_parameters=initial_parameters,
        proximal_mu=proximal_mu,   # Proximal term strength
        on_fit_config_fn=lambda round: {
            "local_epochs": 5,
            "batch_size": 32,
        },
        on_evaluate_config_fn=lambda round: {
            "batch_size": 32,
        }
    )
    
    return strategy


def run_federated_learning(strategy_name="fedavg", num_rounds=10):
    """
    Run federated learning with specified strategy
    
    Args:
        strategy_name: "fedavg" or "fedprox"
        num_rounds: Number of communication rounds
    """
    print("=" * 60)
    print(f"🚀 STARTING FEDERATED LEARNING WITH {strategy_name.upper()} STRATEGY")
    print("=" * 60)
    print(f"Total rounds: {num_rounds}")
    print("Waiting for clients to connect...")
    print("(Open 5 terminal windows and run 'python client.py 0' through 'python client.py 4')\n")
    
    # Clear previous accuracies
    global round_accuracies
    round_accuracies = []
    
    # Choose strategy
    if strategy_name.lower() == "fedavg":
        strategy = create_fedavg_strategy(num_rounds)
    elif strategy_name.lower() == "fedprox":
        strategy = create_fedprox_strategy(num_rounds, proximal_mu=0.1)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    # Start Flower server
    history = fl.server.start_server(
        server_address="localhost:8080",  # Address to listen on
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy
    )
    
    print(f"\n✅ Federated learning with {strategy_name} completed!")
    
    return round_accuracies, history


if __name__ == "__main__":
    # For testing the server file
    print("This file is meant to be imported, not run directly.")
    print("Use centralized.py or run as module")