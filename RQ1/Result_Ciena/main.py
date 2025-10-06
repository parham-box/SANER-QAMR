import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.stats import mannwhitneyu,wilcoxon  # Added import for statistical testing

# Base directory path
base_dir = ""  # Change this to your actual path
subdirs = ["QAMR", "Baseline"]  # The two main subdirectories
metrics = ["arel_res_ciena.csv", "cp_res_ciena.csv", "crec_res_ciena.csv", 
           "crel_res_ciena.csv", "f_res_ciena.csv", "ac_res_ciena.csv"]

# Dictionary to store results
results = {"QAMR": {}, "Baseline": {}}
max_len = 0
c1 = "#5d57d9"
c2 = "#bd3376"

for subdir in subdirs:
    subdir_path = os.path.join(base_dir, subdir)
    metric_data = {metric: [] for metric in metrics}

    for run in range(1, 11):  # Assuming folders are named from 1 to 10
        run_path = os.path.join(subdir_path, str(run))
        
        if not os.path.exists(run_path):
            print(f"Directory {run_path} does not exist. Skipping.")
            continue
        
        for metric in metrics:
            file_path = os.path.join(run_path, metric)
            
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                max_len = len(df)
                if "score" in df.columns:
                    avg_score = df["score"].mean()
                    metric_data[metric].append(avg_score)
                else:
                    continue
            else:
                continue

    results[subdir] = metric_data

# Print results
for subdir, metrics_data in results.items():
    print(f"\nResults for {subdir}:")
    for metric, values in metrics_data.items():
        print(f"{metric}: {values}")

# Define global font properties
plt.rcParams.update({
    "font.family": "Times New Roman",
    "xtick.labelsize": 20,
})

def define_box_properties(plot_name, color_code):
    for element in ['whiskers', 'caps']:
        plt.setp(plot_name[element], color=color_code, linewidth=2, zorder=1)  # Set zorder to a lower value
        for line in plot_name[element]:
            line.set_path_effects([])  # Ensure no shadow effect
            # Apply consistent path effects (if needed)
            line.set_path_effects([
                pe.Stroke(linewidth=3.8, foreground="black"),  # Subtle border enhancement
                pe.Normal()
            ])
    
    for element in ['boxes']:
        for box in plot_name[element]:
            box.set_path_effects([])  # No additional path effects for clean borders

metric_names = [
    "Contextual\nRelevancy",
    "Contextual\nRecall",
    "Contextual\nPrecision",
    "Answer\nFaithfulness",
    "Answer\nRelevancy",
    "Answer\nCorrectness",
]
metric_files = [
    "crel_res_ciena.csv",
    "crec_res_ciena.csv",
    "cp_res_ciena.csv",
    "f_res_ciena.csv",
    "arel_res_ciena.csv",
    "ac_res_ciena.csv",
]

# Prepare data for plotting
data = []
means = []
for metric_file in metric_files:
    our_values = results["QAMR"].get(metric_file, [])
    basic_values = results["Baseline"].get(metric_file, [])

    data.append((our_values, basic_values))
    means.append((np.mean(our_values) if our_values else None, 
                  np.mean(basic_values) if basic_values else None))
# Write means and total rows to a CSV file
means_data = []
for mea, metric_name, data_pair, metric_file in zip(means, metric_names, data, metric_files):
    our_rows = sum(
        len(pd.read_csv(os.path.join(base_dir, "QAMR", str(run), metric_file)))
        for run in range(1, 11)
        if os.path.exists(os.path.join(base_dir, "QAMR", str(run), metric_file))
    )
    basic_rows = sum(
        len(pd.read_csv(os.path.join(base_dir, "Baseline", str(run), metric_file)))
        for run in range(1, 11)
        if os.path.exists(os.path.join(base_dir, "Baseline", str(run), metric_file))
    )
    
    # Append results to data
    means_data.append({
        "Metric": metric_name,
        "Mean Our": mea[0],
        "Rows Our": our_rows,
        "Mean Basic": mea[1],
        "Rows Basic": basic_rows
    })

# Plot all metrics in one figure
fig, ax = plt.subplots(figsize=(14, 10))
positions = np.arange(len(metric_names)) * 5
width = 0.7
ax.set_ylim(-2, positions[-1] + 2.5)  # Increase the upper limit slightly

separator_positions = [(positions[i] + positions[i + 1]) / 2 for i in range(len(positions) - 1)]
for pos in separator_positions:
    ax.axhline(y=pos, color="black", linestyle="-.", linewidth=0.8, alpha=1)
spacing = 1.5  # Add a spacing factor between "Our" and "Basic" plots

for i, ((our_values, basic_values), (mean_our, mean_basic), metric_name) in enumerate(zip(data, means, metric_names)):
    meanprops = dict(marker='^', markerfacecolor='#02bf21', markeredgecolor='none', markersize=12)  # Enlarged triangle
    medianprops = dict(color='orange', linewidth=2.5)  # Thicker median line

    bp_our = ax.boxplot(
        our_values,
        positions=[positions[i] + width + 0.5],
        widths=width,
        showmeans=True,
        vert=False,
        sym='',
        patch_artist=True,
        meanprops=meanprops,  # Apply the mean marker properties
        medianprops=medianprops,  # Apply the median line properties
        boxprops=dict(facecolor=c1),
    )
    define_box_properties(bp_our, c1)
    if mean_our is not None:
        ax.annotate(f"{mean_our:.2f}", xy=(mean_our - 0.005, positions[i] + width + 0.45+ 0.5), 
                    fontsize=20, color="#000", fontweight='bold')

    bp_basic = ax.boxplot(
        basic_values,
        positions=[positions[i] - width - 0.5],
        widths=width,
        showmeans=True,
        vert=False,
        sym='',
        patch_artist=True,
        meanprops=meanprops,  # Apply the mean marker properties
        medianprops=medianprops,  # Apply the median line properties
        boxprops=dict(facecolor=c2),
    )
    define_box_properties(bp_basic, c2)
    if mean_basic is not None:
        ax.annotate(f"{mean_basic:.2f}", xy=(mean_basic - 0.005, positions[i] - width + 0.45- 0.5), 
                    fontsize=20, color="#000", fontweight='bold')

ax.set_yticks(positions)
ax.set_yticklabels(metric_names, fontsize=22)
ax.set_xlabel("Percentage", fontsize=22)
ax.legend(
    [bp_our["boxes"][0], bp_basic["boxes"][0]],
    ['QAMR', 'Baseline'],
    loc="lower left",
    fontsize=20,
)

plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("Comparison_Metrics_Ciena.pdf", bbox_inches="tight")
plt.show()

# ----------------------- Statistical Testing Section -----------------------
# Function to compute A12
def compute_a12(our, basic):
    count = 0
    ties = 0
    for o in our:
        for b in basic:
            if o > b:
                count += 1
            elif o == b:
                ties += 1
    total = len(our) * len(basic)
    a12 = (count + 0.5 * ties) / total
    return a12

# Prepare a list to store statistical test results
stat_results_list = []

# Iterate through each metric and perform statistical tests
for metric_file, metric_name in zip(metric_files, metric_names):
    our_data = results["QAMR"].get(metric_file, [])
    basic_data = results["Baseline"].get(metric_file, [])
    print(len(our_data))
    print(len(basic_data))
    
    if not our_data or not basic_data:
        print(f"Insufficient data for metric {metric_name}. Skipping statistical test.")
        continue
    
    try:
        stat, p_value = wilcoxon(our_data, basic_data)
    except ValueError as e:
        print(f"Error performing Wilcoxon test for {metric_name}: {e}")
        p_value = np.nan
    
    # Compute A12
    a12 = compute_a12(our_data, basic_data)
    
    # Append results to the list
    stat_results_list.append({
        "Metric": metric_name,
        "p-value": p_value,
        "A12": a12
    })

# Convert the list of results into a DataFrame
stat_results = pd.DataFrame(stat_results_list)

# Display the statistical test results
print("\nStatistical Test Results:")
print(stat_results)

# Optionally, save the results to a CSV file
stat_results.to_csv("statistical_test_results.csv", index=False)
