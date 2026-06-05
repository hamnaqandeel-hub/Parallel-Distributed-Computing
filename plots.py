import matplotlib.pyplot as plt # type: ignore

# YOUR ACTUAL ACCURACIES from the server
fedavg_accuracies = [0.9417, 0.9558, 0.9655, 0.9697, 0.9705, 0.9764, 0.9791, 0.9810, 0.9809, 0.9807]

# Your centralized result
centralized_accuracy = 0.9762

# Create comparison plot
plt.figure(figsize=(12, 6))

rounds = range(1, len(fedavg_accuracies) + 1)

# Plot Federated Learning
plt.plot(rounds, fedavg_accuracies, 'b-o', label='Federated Learning (FedAvg)', linewidth=2, markersize=8)

# Plot Centralized baseline (FIXED: separate color and linestyle)
plt.axhline(y=centralized_accuracy, color='red', linestyle='--', label=f'Centralized Training ({centralized_accuracy:.4f})', linewidth=2)

# Labels and title
plt.xlabel('Communication Round', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Federated Learning vs Centralized Training - MNIST Classification', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.ylim([0.92, 0.99])

# Annotate best accuracy
best_acc = max(fedavg_accuracies)
best_round = fedavg_accuracies.index(best_acc) + 1
plt.annotate(f'Best: {best_acc:.4f}', 
             xy=(best_round, best_acc),
             xytext=(best_round + 1, best_acc - 0.005),
             arrowprops=dict(arrowstyle='->', color='blue'),
             fontsize=9, color='blue')

# Save the plot
plt.tight_layout()
plt.savefig('comparison_plot.png', dpi=150)
plt.show()

print("=" * 50)
print("📊 COMPARISON SUMMARY")
print("=" * 50)
print(f"Centralized Training Accuracy: {centralized_accuracy:.4f} ({centralized_accuracy*100:.2f}%)")
print(f"Federated Learning Best Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
print(f"Improvement: +{(best_acc - centralized_accuracy)*100:.2f}%")
print("=" * 50)
print("✅ Comparison plot saved as 'comparison_plot.png'")