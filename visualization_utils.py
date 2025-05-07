# visualization_utils.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_performance_vs_complexity(sentence_lengths, syntactic_depths, accuracies):
    """
    Plot system performance against sentence complexity.
    
    Args:
        sentence_lengths: List of sentence lengths
        syntactic_depths: List of syntactic parse tree depths
        accuracies: List of corresponding accuracies
    """
    plt.figure(figsize=(12, 8))
    
    # Calculate complexity score (combination of length and depth)
    complexity_scores = np.array(sentence_lengths) * 0.2 + np.array(syntactic_depths) * 0.8
    
    # Create scatter plot
    sc = plt.scatter(complexity_scores, accuracies, c=syntactic_depths, 
                    cmap='viridis', alpha=0.7, s=100)
    
    # Add trendline
    z = np.polyfit(complexity_scores, accuracies, 1)
    p = np.poly1d(z)
    plt.plot(complexity_scores, p(complexity_scores), "r--", alpha=0.8, 
             label=f"Trend: y={z[0]:.4f}x+{z[1]:.4f}")
    
    plt.colorbar(sc, label='Syntactic Depth')
    plt.xlabel('Sentence Complexity Score')
    plt.ylabel('Accuracy')
    plt.title('Performance vs. Sentence Complexity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('evaluation/complexity_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_error_classification(error_types, error_counts):
    """
    Create a horizontal bar chart of error types.
    
    Args:
        error_types: List of error type names
        error_counts: List of corresponding error counts
    """
    plt.figure(figsize=(12, 8))
    
    # Sort by frequency
    sorted_indices = np.argsort(error_counts)
    error_types = [error_types[i] for i in sorted_indices]
    error_counts = [error_counts[i] for i in sorted_indices]
    
    # Create horizontal bar chart
    bars = plt.barh(error_types, error_counts, color='maroon', alpha=0.8)
    
    # Add counts as text
    for i, (type_name, count) in enumerate(zip(error_types, error_counts)):
        plt.text(count + 0.5, i, str(count), va='center')
    
    plt.xlabel('Count')
    plt.ylabel('Error Type')
    plt.title('Classification of Error Types')
    plt.tight_layout()
    plt.savefig('evaluation/error_classification.png', dpi=300, bbox_inches='tight')
    plt.close()