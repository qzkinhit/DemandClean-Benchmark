import random

import numpy as np
from matplotlib import pyplot as plt

from matplotlib.ticker import FuncFormatter, MaxNLocator



def single_system_performance_with_bar(data_sets, metrics, performance_data):
    """
    Draw bar charts of a single system across multiple datasets and metrics. "S" denotes cleaning speed in seconds.

    :param data_sets: list of dataset names
    :param metrics: list of metric names
    :param performance_data: dict of metric -> per-dataset values
    """
    num_datasets = len(data_sets)
    num_metrics = len(metrics)
    x = np.arange(num_datasets)  # dataset positions
    colors = plt.cm.viridis(np.linspace(0, 1, num_datasets))  # per-dataset colors
    fig, axes = plt.subplots(1, num_metrics, figsize=(20, 5))

    for i, metric in enumerate(metrics):
        ax = axes[i]
        metric_data = performance_data[metric]

        # Bar chart for this single-system metric.
        ax.bar(x, metric_data, color=colors[i], width=0.5, label=metric)

        # Subplot title and labels.
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(data_sets, rotation=45)
        # Set a y-axis range appropriate for each metric.
        if metric in ["F1", "EDR", "REDR"]:
            ax.set_ylim(0, 1)  # assumed range [0, 1]
        elif metric == "S":  # cleaning speed
            ax.set_ylabel("Speed (seconds per 100 records)")
            ax.set_ylim(0, max(metric_data) * 1.1)  # auto range for cleaning speed
        elif metric == "Hybrid Distance":
            ax.set_ylim(0, max(metric_data) * 1.1)  # auto range for Hybrid Distance

        ax.grid(axis='y', linestyle='--', alpha=0.7)

    fig.suptitle("Single System Performance Across Datasets for Various Metrics (Including Cleaning Speed)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()



def performance_difference_with_bar(data_sets, metrics, baseline_performance, alternative_performance):
    """
    Plot the percentage performance difference between a greedy alternative and a baseline system across datasets, colored per dataset.
    The y-axis is clamped to 50-150% to highlight differences.

    :param data_sets: list of dataset names
    :param metrics: list of metric names
    :param baseline_performance: dict of metric -> baseline values
    :param alternative_performance: dict of metric -> alternative values
    """
    num_datasets = len(data_sets)
    num_metrics = len(metrics)
    x = np.arange(num_datasets)  # dataset positions
    colors = plt.cm.viridis(np.linspace(0, 1, num_datasets))  # per-dataset colors

    fig, axes = plt.subplots(1, num_metrics, figsize=(20, 5), sharey=True)

    for i, metric in enumerate(metrics):
        ax = axes[i]
        baseline = baseline_performance[metric]
        alternative = alternative_performance[metric]

        # Compute the percentage difference.
        percentage_difference = [(alt / base) * 100 for alt, base in zip(alternative, baseline)]

        # Bar chart of the differences.
        for j in range(num_datasets):
            ax.bar(x[j], percentage_difference[j], color=colors[j], width=0.5, label=data_sets[j] if i == 0 else "")

        # Subplot title and labels.
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(data_sets, rotation=45)
        ax.set_ylim(50, 150)  # clamp y-axis to [50, 150]%
        ax.set_ylabel("Difference (%)")
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Shared legend.
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles[:num_datasets], labels[:num_datasets], loc="upper center", ncol=num_datasets,
               bbox_to_anchor=(0.5, 1.15))
    fig.suptitle("Performance Difference Across Datasets: Greedy vs. Baseline System")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def add_random_fluctuations(base_values, fluctuation_range=0.5):
    """
    Add random fluctuations to a list of base values.

    :param base_values: list of base values
    :param fluctuation_range: fluctuation range
    :return: list of values with added fluctuation
    """
    return [value + random.uniform(-fluctuation_range, fluctuation_range) * value for value in base_values]


def injected_error_rates(error_injection_rates, datasets, cell_error_rates, entry_error_rates):
    """
    Plot two charts: cell error rates and entry error rates for each dataset under multiple independent error-injection rates.

    :param error_injection_rates: list of injection rates (x-axis)
    :param datasets: list of dataset names
    :param cell_error_rates: 2D list of cell error rates per dataset per injection rate
    :param entry_error_rates: 2D list of entry error rates per dataset per injection rate
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 9))
    markers = ['o', 's', 'D', '^', 'v', 'P', '*']  # marker shapes
    colors = plt.cm.viridis(np.linspace(0, 1, len(datasets)))  # per-dataset colors

    # Chart 1: cell error rate.
    ax1 = axes[0]
    for i, dataset in enumerate(datasets):
        ax1.plot(error_injection_rates, cell_error_rates[i], marker=markers[i % len(markers)],
                 color=colors[i], label=dataset, linestyle='-', markersize=6)
    ax1.set_title("Cell Error Rate by Injection Rate")
    ax1.set_xlabel("Error Injection Rate (%)")
    ax1.set_ylabel("Cell Error Rate (%)")
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Chart 2: entry error rate.
    ax2 = axes[1]
    for i, dataset in enumerate(datasets):
        ax2.plot(error_injection_rates, entry_error_rates[i], marker=markers[i % len(markers)],
                 color=colors[i], label=dataset, linestyle='-', markersize=6)
    ax2.set_title("Entry Error Rate by Injection Rate")
    ax2.set_xlabel("Error Injection Rate (%)")
    ax2.set_ylabel("Entry Error Rate (%)")
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    # Shared legend.
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(datasets), bbox_to_anchor=(0.5, 1.15))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def baseline_performance_difference_with_line(data_sets, system_names, performance_metrics, baseline_index=0):
    """
    Plot line charts of system performance across five metrics in a single figure, with a shared legend on top.

    :param data_sets: list of dataset names
    :param system_names: list of system names
    :param performance_metrics: dict of metric -> 2D list of per-system values
    :param baseline_index: baseline system index (default: first system)
    """
    fig, axes = plt.subplots(1, 5, figsize=(12, 8))
    axes = axes.flatten()  # flatten axes for indexing
    markers = ['o', 's', 'D', '^', 'v', 'P', '*']  # per-system markers

    for i, (metric_name, performance_data) in enumerate(performance_metrics.items()):
        ax = axes[i]

        # Compute relative performance (%).
        baseline_performance = performance_data[baseline_index]
        relative_performance = [
            [perf / base * 100 for perf, base in zip(system_perf, baseline_performance)]
            for system_perf in performance_data
        ]

        # Plot each system's line chart.
        for j, (system_name, rel_perf) in enumerate(zip(system_names, relative_performance)):
            ax.plot(np.arange(len(data_sets)), rel_perf, marker=markers[j % len(markers)], label=system_name,
                    linestyle='-', markersize=6)

        # Subplot title and labels.
        ax.set_title(metric_name)
        ax.set_xticks(np.arange(len(data_sets)))
        ax.set_xticklabels(data_sets, rotation=45)
        ax.set_ylim(70, 105)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Shared legend below the main plot.
    legend_elements = [
        plt.Line2D([0], [0], marker=markers[i], color='black', label=system_name, markersize=8, linestyle='None')
        for i, system_name in enumerate(system_names)]
    fig.legend(handles=legend_elements, loc="upper center", ncol=len(system_names))

    # Title at the very bottom.
    fig.text(0.5, -0.1, "Performance Comparison Across Systems for Different Metrics (Relative to Baseline)",
             ha='center', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # accommodate title and legend
    plt.show()


def actual_performance_comparison_with_bars(data_sets, system_names, performance_metrics):
    """
    Bar chart comparing systems across datasets, with different hatch patterns per system under each dataset.

    Each metric adjusts its y-axis based on the data range; out-of-range values are rendered as "<" or ">".

    :param data_sets: list of dataset names
    :param system_names: list of system names
    :param performance_metrics: dict of metric -> 2D list of per-system values
    """
    num_metrics = len(performance_metrics)
    fig, axes = plt.subplots(1, min(num_metrics, 4), figsize=(13, 5), sharex=True)

    # If more than four metrics, wrap into extra rows.
    if num_metrics > 4:
        fig, axes = plt.subplots((num_metrics + 3) // 4, 4, figsize=(20, 5 * ((num_metrics + 3) // 4)), sharex=True)

    axes = axes.flatten()  # flatten axes for indexing
    hatch_patterns = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']  # per-system hatch patterns
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # default color cycle

    # Per-metric y-axis bounds.
    limits = {
        "Clean Time per 100 Records(s)": (None, 70),  # values > 70 are rendered as >70
        "EDR": (-0.6, None),  # lower-bound only; values < -0.5 are rendered as <-0.5
        "REDR": (-0.6, None)  # lower-bound only; values < -0.5 are rendered as <-0.5
    }
    expand_ratio = 0  # ratio for display expansion

    for i, (metric_name, performance_data) in enumerate(performance_metrics.items()):
        ax = axes[i]

        # Transpose so each column corresponds to a system.
        transposed_data = list(zip(*performance_data))
        num_datasets = len(data_sets)
        num_systems = len(system_names)

        bar_width = 0.15  # bar width
        x = np.arange(num_datasets)  # dataset x positions

        # Clip data to the configured bounds.
        adjusted_performance = []
        for perf_data in transposed_data:
            if metric_name in limits:
                lower_limit, upper_limit = limits[metric_name]
                adjusted_perf_data = [
                    max(value, lower_limit) if lower_limit is not None else value for value in perf_data
                ]
                adjusted_perf_data = [
                    min(value, upper_limit) if upper_limit is not None else value for value in adjusted_perf_data
                ]
            else:
                adjusted_perf_data = perf_data
            adjusted_performance.append(adjusted_perf_data)

        # Plot each system's bars.
        for j, (system_name, system_data) in enumerate(zip(system_names, adjusted_performance)):
            ax.bar(x + j * bar_width, system_data, width=bar_width, label=system_name,
                   color=colors[j % len(colors)], hatch=hatch_patterns[j % len(hatch_patterns)])

        # Subplot title and labels.
        ax.set_title(metric_name)
        ax.set_xticks(x + bar_width * (num_systems - 1) / 2)
        ax.set_xticklabels(data_sets, rotation=45)

        if metric_name in limits:
            lower_limit, upper_limit = limits[metric_name]
            # Fine-tune the lower bound so bars align with the baseline.
            adjusted_lower_limit = lower_limit if lower_limit is not None and lower_limit < 0 else lower_limit
            # expanded_upper_limit = upper_limit + 0.05 if upper_limit<0 is not None else None
            ax.set_ylim(adjusted_lower_limit, upper_limit)
            yticks = ax.get_yticks()
            # adjusted_lower_limit = lower_limit + 0.01 if lower_limit is not None and lower_limit < 0 else lower_limit
            # ax.set_ylim(adjusted_lower_limit, upper_limit)
            # Deduplicate labels while preserving order so custom_yticks matches yticks length.
            seen = set()
            custom_yticks = []
            for y in yticks:
                label = (
                    f"<{lower_limit}" if lower_limit is not None and y <= lower_limit else
                    f">{upper_limit}" if upper_limit is not None and y >= upper_limit else
                    (f"{int(y)}" if metric_name == "Clean Time per 100 Records(s)"  else f"{y:.2f}")
                )
                if label not in seen:
                    custom_yticks.append(label)
                    seen.add(label)
                else:
                    # Keep length aligned by padding duplicates with an empty string.
                    custom_yticks.append("")
                if metric_name == "EDR":
                    print(1111)

            ax.set_yticks(yticks)
            ax.set_yticklabels(custom_yticks)
        # Grid lines.
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Remove unused axes.
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Legend on top.
    fig.legend(system_names, loc="upper center", ncol=len(system_names), fontsize=19)
    fig.text(0.5, 0.15, "Datasets", ha='center', fontsize=12)
    fig.text(0.5, 0.08, "* All baseline systems except UniClean unable to handle full 50k Tax dataset in 24 hour."
                        "we evaluated using segmented 10k batch inputs.", ha='center', fontsize=12)

    plt.tight_layout(rect=[0, 0.1, 1, 0.85])  # accommodate title and legend
    return plt
def actual_performance_comparison_with_line(data_sets, system_names, performance_metrics, baseline_index=0):
    """
    Plot line charts of actual system performance across five metrics, with a shared legend on top.
    For the Speed metric, values above 70 are rendered as ">70"; for EDR/REDR, values below -2 are rendered as "<-2".

    :param data_sets: list of dataset names
    :param system_names: list of system names
    :param performance_metrics: dict of metric -> 2D list of per-system values
    :param baseline_index: baseline system index (default: first system)
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()  # flatten axes for indexing
    markers = ['o', 's', 'D', '^', 'v', 'P', '*']  # per-system markers
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # default color cycle

    # Per-metric bounds with expansion ratio for display.
    limits = {
        "Clean Time per 100 Records(s)": (0, 70),  # clamp Speed to [0, 70]
        "EDR": (-1.5, 1),                   # clamp EDR to [-2, 1]
        "REDR": (-1.5, 1)                   # clamp REDR to [-2, 1]
    }
    expand_ratio = 0.1  # display expansion ratio

    for i, (metric_name, performance_data) in enumerate(performance_metrics.items()):
        ax = axes[i]

        # Transpose so each column corresponds to a system.
        transposed_data = list(zip(*performance_data))
        adjusted_performance = []

        # Clip data to the configured bounds.
        for perf_data in transposed_data:
            if metric_name in limits:
                lower_limit, upper_limit = limits[metric_name]
                adjusted_perf_data = [
                    min(max(value, lower_limit), upper_limit) for value in perf_data
                ]
            else:
                adjusted_perf_data = perf_data
            adjusted_performance.append(adjusted_perf_data)

        # Plot each system's line chart.
        for j, (system_name, perf_data) in enumerate(zip(system_names, adjusted_performance)):
            ax.plot(np.arange(len(data_sets)), perf_data, marker=markers[j % len(markers)], label=system_name,
                    linestyle='-', markersize=15, color=colors[j % len(colors)])  # tweak markersize

        # Subplot title and labels.
        ax.set_title(metric_name)
        ax.set_xticks(np.arange(len(data_sets)))
        ax.set_xticklabels(data_sets, rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Set y-axis range and custom tick labels.
        if metric_name in limits:
            lower_limit, upper_limit = limits[metric_name]
            expanded_lower_limit = lower_limit - abs(lower_limit * expand_ratio)
            expanded_upper_limit = upper_limit + abs(upper_limit * expand_ratio)
            ax.set_ylim(expanded_lower_limit, expanded_upper_limit)

            yticks = ax.get_yticks()
            custom_yticks = [f"<{lower_limit}" if y <= lower_limit else f">{upper_limit}" if y >= upper_limit else y for y in yticks]
            ax.set_yticklabels(custom_yticks)

    # Shared legend with system colors.
    legend_elements = [
        plt.Line2D([0], [0], marker=markers[i], color=colors[i % len(colors)], label=system_name, markersize=8, linestyle='None')
        for i, system_name in enumerate(system_names)]
    fig.legend(handles=legend_elements, loc="upper center", ncol=len(system_names), fontsize=15)

    # Title at the very bottom.
    fig.text(0.5, -0.1, "Actual Performance Comparison Across Systems for Different Metrics",
             ha='center', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # accommodate title and legend
    return plt
if __name__ == "__main__":
    # # Redefined example data.
    # data_sets_example = ["Hospital", "Flights", "Beers", "Rayyan", "Tax", "Soccer", "Commercial"]
    # metrics_example = ["F1", "S", "EDR", "REDR", "Hybrid Distance"]
    # performance_data_example = {
    #     "F1": [0.86, 0.91, 0.89, 0.88, 0.90, 0.86, 0.87],
    #     "S": [1.5, 1.8, 2.0, 1.7, 1.9, 1.6, 1.4],  # cleaning speed (seconds per 100 records)
    #     "EDR": [0.88, 0.93, 0.91, 0.90, 0.92, 0.88, 0.89],
    #     "REDR": [0.89, 0.94, 0.92, 0.91, 0.93, 0.89, 0.90],
    #     "Hybrid Distance": [0.15, 0.13, 0.17, 0.14, 0.16, 0.15, 0.14]
    # }
    #
    # # Draw a bar chart with a distinct color per dataset.
    # single_system_performance_with_bar(data_sets_example, metrics_example, performance_data_example)
    # # Example data.
    # data_sets_example = ["Hospital", "Flights", "Beers", "Rayyan", "Tax", "Soccer"]
    # metrics_example = ["F1/F0", "S/S0", "EDR/EDR0", "REDR/REDR0", "HD/HD0"]
    # baseline_performance_example = {
    #     "F1/F0": [0.90, 0.92, 0.91, 0.93, 0.89, 0.91],
    #     "S/S0": [1.2, 1.4, 1.5, 1.3, 1.2, 1.4],
    #     "EDR/EDR0": [0.88, 0.90, 0.89, 0.91, 0.87, 0.89],
    #     "REDR/REDR0": [0.87, 0.89, 0.88, 0.90, 0.86, 0.88],
    #     "HD/HD0": [0.15, 0.14, 0.16, 0.15, 0.13, 0.14]
    # }
    # alternative_performance_example = {
    #     "F1/F0": [0.85, 0.88, 0.86, 0.89, 0.84, 0.87],
    #     "S/S0": [1.3, 1.6, 1.7, 1.4, 1.3, 1.5],
    #     "EDR/EDR0": [0.82, 0.85, 0.84, 0.87, 0.81, 0.83],
    #     "REDR/REDR0": [0.80, 0.83, 0.82, 0.85, 0.79, 0.81],
    #     "HD/HD0": [0.18, 0.16, 0.19, 0.17, 0.15, 0.16]
    # }
    #
    # # Draw a per-dataset-colored difference bar chart.
    # performance_difference_with_bar(data_sets_example, metrics_example, baseline_performance_example,
    #                                     alternative_performance_example)
    # # Example data.
    # error_injection_rates_example = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
    # datasets_example = ["Hospital", "Flights", "Beers", "Rayyan", "Tax", "Soccer"]
    #
    # # Base cell and entry error rates.
    # base_cell_error_rates = [
    #     [2, 4, 6, 8, 10, 12, 14, 16],
    #     [1.5, 3, 4.5, 6, 7.5, 9, 10.5, 12],
    #     [2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20],
    #     [1, 2, 3, 4, 5, 6, 7, 8],
    #     [2.2, 4.4, 6.6, 8.8, 11, 13.2, 15.4, 17.6],
    #     [1.8, 3.6, 5.4, 7.2, 9, 10.8, 12.6, 14.4]
    # ]
    # base_entry_error_rates = [
    #     [5, 10, 15, 20, 25, 30, 35, 40],
    #     [4, 8, 12, 16, 20, 24, 28, 32],
    #     [6, 12, 18, 24, 30, 36, 42, 48],
    #     [3, 6, 9, 12, 15, 18, 21, 24],
    #     [5.5, 11, 16.5, 22, 27.5, 33, 38.5, 44],
    #     [4.5, 9, 13.5, 18, 22.5, 27, 31.5, 36]
    # ]
    #
    # # Add fluctuation to each base error rate.
    # cell_error_rates_example = [add_random_fluctuations(base) for base in base_cell_error_rates]
    # entry_error_rates_example = [add_random_fluctuations(base) for base in base_entry_error_rates]
    #
    # injected_error_rates(error_injection_rates_example, datasets_example, cell_error_rates_example,
    #                           entry_error_rates_example)
    # # Simulated input data.
    # data_sets_example = ["Hospital", "Flights", "Beers", "Rayyan", "Tax", "Soccer", "Commercial"]
    # system_names_example = ["Uniclean", "Baran", "Holistic", "bigDansing", "Horizon", "Holoclean"]
    # performance_metrics_example = {
    #     "F1 Score": [
    #         [0.95, 0.97, 0.93, 0.92, 0.96, 0.94, 0.95],
    #         [0.76, 0.83, 0.84, 0.81, 0.80, 0.78, 0.80],
    #         [0.72, 0.76, 0.78, 0.74, 0.73, 0.71, 0.72],
    #         [0.82, 0.86, 0.89, 0.90, 0.84, 0.85, 0.86],
    #         [0.89, 0.92, 0.91, 0.93, 0.90, 0.91, 0.92],
    #         [0.74, 0.79, 0.81, 0.78, 0.76, 0.79, 0.77]
    #     ],
    #     "Speed(s/100 records)": [
    #         [4.3, 4.4, 4.1, 3.9, 4.2, 4.0, 4.3],
    #         [3.3, 3.1, 3.2, 2.8, 2.6, 2.4, 2.7],
    #         [2.1, 2.4, 2.5, 2.2, 2.0, 1.9, 2.1],
    #         [3.3, 3.5, 3.8, 3.7, 3.2, 3.3, 3.4],
    #         [3.7, 4.0, 3.9, 4.1, 3.8, 3.9, 4.0],
    #         [3.0, 2.8, 3.0, 2.6, 2.5, 2.6, 2.5]
    #     ],
    #     "EDR": [
    #         [0.92, 0.93, 0.90, 0.88, 0.91, 0.89, 0.92],
    #         [0.75, 0.82, 0.83, 0.79, 0.78, 0.76, 0.78],
    #         [0.72, 0.75, 0.76, 0.73, 0.71, 0.70, 0.72],
    #         [0.84, 0.87, 0.88, 0.86, 0.83, 0.84, 0.85],
    #         [0.88, 0.90, 0.90, 0.92, 0.89, 0.90, 0.91],
    #         [0.72, 0.79, 0.80, 0.77, 0.75, 0.77, 0.76]
    #     ],
    #     "REDR": [
    #         [0.94, 0.95, 0.91, 0.90, 0.93, 0.92, 0.94],
    #         [0.74, 0.82, 0.83, 0.80, 0.78, 0.77, 0.79],
    #         [0.73, 0.76, 0.78, 0.74, 0.72, 0.71, 0.73],
    #         [0.85, 0.88, 0.89, 0.88, 0.85, 0.86, 0.87],
    #         [0.89, 0.92, 0.90, 0.94, 0.90, 0.91, 0.92],
    #         [0.75, 0.80, 0.82, 0.79, 0.76, 0.78, 0.77]
    #     ]
    #     # ,
    #     # "Hybrid Distance": [
    #     #     [0.91, 0.92, 0.89, 0.87, 0.90, 0.88, 0.91],
    #     #     [0.72, 0.80, 0.82, 0.78, 0.77, 0.75, 0.76],
    #     #     [0.71, 0.74, 0.75, 0.71, 0.70, 0.69, 0.71],
    #     #     [0.83, 0.86, 0.87, 0.85, 0.82, 0.83, 0.84],
    #     #     [0.86, 0.91, 0.89, 0.90, 0.87, 0.88, 0.89],
    #     #     [0.70, 0.78, 0.79, 0.76, 0.74, 0.75, 0.74]
    #     # ]
    # }
    data_sets = ["Hospital", "Flights", "Beers", "Rayyan", "Tax50k"]
    system_names = ["Baran", "bigDansing", "Holistic", "Horizon", "Uniclean"]
    performance_metrics = {
        "F1 Score": [
            [0.6651, 0.6050, 0.6080, 0.5841, 0.8847],  # Hospital
            [0.6278, 0.3870, 0.4067, 0.4049, 0.6537],  # Flights
            [0.7907, 0.0940, 0.0939, 0.1051, 0.8373],  # Beers
            [0.2513, 0.0259, 0.0006, 0.0091, 0.9213],  # Rayyan
            [0.0634, 0.0912, 0.0876, 0.0018, 0.5011]  # Tax50k
        ],
        "Clean Time per 100 Records(s)": [
            [46.90, 23.30, 105.33, 0.32, 10.81],  # Hospital
            [19.35, 2694.66, 231.85, 0.23, 3.56],  # Flights
            [20.57, 1.27, 65.43, 2.59, 1.30],  # Beers
            [44.39, 83.35, 2017.27, 0.62, 5.24],  # Rayyan
            [21.97, 103.70, 574.09, 27.58, 25.18]  # Tax50k
        ],
        "EDR": [
            [0.5246, -0.0766, -0.0236, 0.0570, 0.7839],  # Hospital
            [0.4478, -0.1382, -0.1191, 0.1148, 0.5175],  # Flights
            [0.7540, -0.0104, -0.0113, 0.0027, 0.8329],  # Beers
            [0.1437, -0.5120, -2.0654, -0.5281, 0.9005],  # Rayyan
            [0.0160, -1.0693, -1.2427, -50.9573, -0.0306]  # Tax50k
        ],
        "REDR": [
            [0.4742, 0.0221, 0.0442, 0.0270, 0.7543],  # Hospital
            [0.0326, -0.0693, -0.0636, -0.1534, 0.1129],  # Flights
            [0.6867, 0.0, 0.0, 0.0, 0.7730],  # Beers
            [0.1033, -0.1699, -0.1956, -0.1918, 0.8827],  # Rayyan
            [0.0800, -0.0272, -0.0353, -25.9592, 0.4250]  # Tax50k
        ]
    }
    # Call plot_baseline_performance_combined to produce the combined plot.
    # plt=actual_performance_comparison_with_line(data_sets, system_names, performance_metrics)

    plt = actual_performance_comparison_with_bars(data_sets, system_names, performance_metrics)
    # Save the figure as SVG/PNG/EPS.
    plt.savefig("demoplt.png", format="png")
    plt.savefig("result1107.eps", format="eps")
