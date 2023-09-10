import matplotlib.pyplot as plt
import numpy as np


def compare_similarity(
        natural_instruction_bertscore: list,
        unnatural_instruction_bertscore: list,
) -> None:
    """
    Compare and visualize the similarity distributions between natural and unnatural instructions using BERTScore.

    Args:
        natural_instruction_bertscore (list): List of BERTScore values for natural instructions.
        unnatural_instruction_bertscore (list): List of BERTScore values for unnatural instructions.
    """

    # Set the number of bins and the range of values
    num_bins = 100
    value_range = (0, 1)

    # Create histogram data for natural and unnatural bertscore
    natural_hist, bin_edges = np.histogram(natural_instruction_bertscore, bins=num_bins, range=value_range)
    unnatural_hist, _ = np.histogram(unnatural_instruction_bertscore, bins=num_bins, range=value_range)

    # Calculate bin width
    bin_width = bin_edges[1] - bin_edges[0]

    # Create x-axis values at the center of each bin
    x = bin_edges[:-1] + bin_width / 2

    # Create subplots
    fig, ax = plt.subplots()

    # Plot natural bertscore histogram
    ax.bar(x, natural_hist, width=bin_width, alpha=0.5, label='Super-Natural')

    # Plot unnatural bertscore histogram
    ax.bar(x, unnatural_hist, width=bin_width, alpha=0.5, label='Unnatural')

    # Set labels and title
    ax.set_xlabel('Similarity')
    ax.set_ylabel('Count')
    ax.set_title('BERTScore Distribution')

    # Add legend
    ax.legend()

    # Show the plot
    plt.show()