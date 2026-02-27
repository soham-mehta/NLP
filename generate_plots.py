import matplotlib.pyplot as plt
import numpy as np

def smooth_curve(points, factor=0.95):
    """Turns the noisy batch 'candles' into a smooth continuous line."""
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

try:
    # Load your saved data
    data = {
        "One-Hot (Part 1)": np.load('losses_to_one_hot.npy'),
        "BoW (Part 1)": np.load('losses_to_bow.npy'),
        "Word2Vec (Part 1)": np.load('losses_to_word2vec.npy'),
        "BERT (Part 2)": np.load('losses_bert.npy')
    }

    # Create 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Determine shared limits for Fair Comparison (Requirement 2)
    # Find the max steps across all models for the X-axis
    max_steps = max([len(d) for d in data.values()])
    # Set a fixed Y-limit (Cross-Entropy Loss rarely exceeds 1.5-2.0 after a few steps)
    y_limit = 1.5 

    for i, (name, losses) in enumerate(data.items()):
        ax = axes[i]
        
        # Plot the smooth continuous line
        ax.plot(smooth_curve(losses), label='Smoothed Loss', color='blue', linewidth=2)
        
        # Formatting for "Identical Scaling" requirement
        ax.set_title(name)
        ax.set_xlim(0, max_steps) # Identical x-axis scaling
        ax.set_ylim(0, y_limit)   # Identical y-axis scaling
        ax.set_xlabel("Gradient Update Steps")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("task5_side_by_side.png")
    print("Side-by-side plot saved as 'task5_side_by_side.png'")
    plt.show()

except FileNotFoundError as e:
    print(f"Error: Could not find the .npy files. Run your training scripts first. {e}")