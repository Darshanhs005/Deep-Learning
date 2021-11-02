import matplotlib.pyplot as plt
import numpy as np

def plot_label_distribution(labels, split, class_names):
    # Given a flat list of integers `labels`, counts how many of each
    counts = [ sum(labels == c) for c in range(len(class_names)) ]

    # Plot a histogram of the distribution
    plt.title(f'{split} distribution')
    plt.bar(class_names, counts)
    plt.xlabel('Class')
    plt.ylabel('Num examples')
    plt.show()
