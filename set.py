import matplotlib.pyplot as plt # type: ignore
import numpy as np

# Key comparison values
metrics = ['Best Accuracy (%)', 'Training Time (s)', 'Privacy Score\n(1-10)', 'Communication\nRounds', 'Number of\nClients']

centralized = [97.62, 45, 0, 0, 1]
federated = [98.10, 210, 10, 10, 5]

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, centralized, width, label='Centralized', color='#3498db')
bars2 = ax.bar(x + width/2, federated, width, label='Federated', color='#2ecc71')

ax.set_ylabel('Values', fontsize=12)
ax.set_title('Centralized vs Federated Learning: Final Comparison', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=10)
ax.legend(fontsize=10)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('final_comparison.png', dpi=150)
plt.show()