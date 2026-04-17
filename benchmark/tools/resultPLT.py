import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def calculate_accuracy_and_recall(clean, dirty, cleaned, attributes):
    """
    Compute repair accuracy and recall over a set of attributes.

    :param clean: clean DataFrame
    :param dirty: dirty DataFrame
    :param cleaned: cleaned DataFrame
    :param attributes: attributes to score over
    :return: repair accuracy and recall
    """
    total_true_positives = 0
    total_false_positives = 0
    total_true_negatives = 0

    for attribute in attributes:
        clean_values = clean[attribute]
        dirty_values = dirty[attribute]
        cleaned_values = cleaned[attribute]

        # Align indices.
        common_indices = clean_values.index.intersection(cleaned_values.index).intersection(dirty_values.index)
        clean_values = clean_values.loc[common_indices]
        dirty_values = dirty_values.loc[common_indices]
        cleaned_values = cleaned_values.loc[common_indices]

        # Correctly repaired cells.
        true_positives = ((cleaned_values == clean_values) & (dirty_values != cleaned_values)).sum()
        # Incorrectly repaired cells.
        false_positives = ((cleaned_values != clean_values) & (dirty_values != cleaned_values)).sum()
        # Cells that should have been repaired.
        true_negatives = (dirty_values != clean_values).sum()

        total_true_positives += true_positives
        total_false_positives += false_positives
        total_true_negatives += true_negatives

    # Overall repair accuracy.
    accuracy = total_true_positives / (total_true_positives + total_false_positives)
    # Overall repair recall.
    recall = total_true_positives / total_true_negatives

    return accuracy, recall


def plot_metrics(names, clean_dfs, dirty_dfs, cleaned_dfs, attributes_list):
    """
    Plot accuracy, recall, and F1 for a set of cleaning runs. Supports both "multiple cleaners on one dataset" and "one cleaner on multiple datasets" layouts.

    :param names: list of names (either cleaner names or dataset names)
    :param clean_dfs: list of clean DataFrames
    :param dirty_dfs: list of dirty DataFrames
    :param cleaned_dfs: list of cleaned DataFrames (one list per entry)
    :param attributes_list: per-dataset attribute lists
    """
    accuracies = []
    recalls = []
    f1_scores = []
    labels = []

    for name, clean, dirty, cleaned_list, attributes in zip(names, clean_dfs, dirty_dfs, cleaned_dfs, attributes_list):
        for i, cleaned in enumerate(cleaned_list):
            accuracy, recall = calculate_accuracy_and_recall(clean, dirty, cleaned, attributes)
            accuracies.append(accuracy)
            recalls.append(recall)
            f1_scores.append(2 * (accuracy * recall) / (accuracy + recall))
            labels.append(f"{name} Cleaned {i + 1}")

    x = np.arange(len(labels))
    width = 0.35

    # Plot accuracy.
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(x, accuracies, width, label='Accuracy')
    ax.set_xlabel('Cleaned Datasets')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy for Multiple Systems/Datasets')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    plt.savefig("accuracy_metrics.png")
    plt.show()

    # Plot recall.
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(x, recalls, width, label='Recall')
    ax.set_xlabel('Cleaned Datasets')
    ax.set_ylabel('Recall')
    ax.set_title('Recall for Multiple Systems/Datasets')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    plt.savefig("recall_metrics.png")
    plt.show()

    # Plot F1.
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(x, f1_scores, width, label='F1 Score')
    ax.set_xlabel('Cleaned Datasets')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score for Multiple Systems/Datasets')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    plt.savefig("f1_score_metrics.png")
    plt.show()


if __name__ == "__main__":
    # Example data loading.
    names = ["System1", "System2"]
    clean_paths = ["data/clean1.csv", "data/clean2.csv"]
    dirty_paths = ["data/dirty1.csv", "data/dirty2.csv"]
    cleaned_paths = [["data/cleaned1_1.csv", "data/cleaned1_2.csv"], ["data/cleaned2_1.csv", "data/cleaned2_2.csv"]]

    clean_dfs = [pd.read_csv(path) for path in clean_paths]
    dirty_dfs = [pd.read_csv(path) for path in dirty_paths]
    cleaned_dfs = [[pd.read_csv(path) for path in paths] for paths in cleaned_paths]

    attributes_list = [["attr1", "attr2", "attr3"], ["attr1", "attr2"]]

    plot_metrics(names, clean_dfs, dirty_dfs, cleaned_dfs, attributes_list)
