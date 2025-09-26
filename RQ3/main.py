import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.stats import wilcoxon  # For statistical testing

# Base directory path
base_dir = ""  # Set this to your actual path

# List of all groups to process
subdirs = ["Base", "QueryRewriting", "DualChunking", "ContextReduction", "ContextSelection","QAMR" ][::-1]

# Metrics (filenames) to process
metrics = [
    "arel_res_public.csv", "cp_res_public.csv", "crec_res_public.csv", 
    "crel_res_public.csv", "f_res_public.csv", "ac_res_public.csv"
]

# Define colors for each group
colors = {
    "Base": "#e6f545", 
    "QueryRewriting": "#9ff5be", 
    "DualChunking": "#ab8302",
    "ContextReduction": "#ed765c", 
    "ContextSelection": "#e09be0", 
    "QAMR": "#5d57d9", 
}

from matplotlib import rcParams

# Dictionary to store results per group, per metric
results = {group: {metric: [] for metric in metrics} for group in subdirs}
max_len = 0


for subdir in subdirs:
    subdir_path = os.path.join(base_dir, subdir)
    if not os.path.exists(subdir_path):
        print(f"Directory {subdir_path} does not exist. Skipping group {subdir}.")
        continue

    # List all directories that start with "run"
    run_folders = [d for d in os.listdir(subdir_path)
                   if os.path.isdir(os.path.join(subdir_path, d)) and d.startswith("run")]
    
    for run_folder in run_folders:
        run_path = os.path.join(subdir_path, run_folder)
        
        for metric in metrics:
            file_path = os.path.join(run_path, metric)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                max_len = len(df) if len(df) > max_len else max_len
                if "score" in df.columns:
                    avg_score = df["score"].mean()
                    results[subdir][metric].append(avg_score)
                else:
                    print(f"'score' column not found in {file_path}.")
            else:
                print(f"File {file_path} not found.")
                    
# Print out the results for verification
for subdir, metrics_data in results.items():
    print(f"\nResults for {subdir}:")
    for metric, values in metrics_data.items():
        print(f"{metric}: {values}")

# Set global font properties
plt.rcParams.update({
    "font.family": "Times New Roman",
    "xtick.labelsize": 20,
})

def define_box_properties(plot_name, color_code):
    for element in ['whiskers', 'caps']:
        plt.setp(plot_name[element], color=color_code, linewidth=2, zorder=1)
        for line in plot_name[element]:
            line.set_path_effects([
                pe.Stroke(linewidth=3.8, foreground="black"),
                pe.Normal()
            ])
    for element in ['boxes']:
        for box in plot_name[element]:
            box.set_path_effects([])

metric_names = [
    "Contextual\nRelevancy",
    "Contextual\nRecall",
    "Contextual\nPrecision",
    "Answer\nFaithfulness",
    "Answer\nRelevancy",
    "Answer\nCorrectness",
]
metric_files = [
    "crel_res_public.csv",
    "crec_res_public.csv",
    "cp_res_public.csv",
    "f_res_public.csv",
    "arel_res_public.csv",
    "ac_res_public.csv",
]

# Prepare data for plotting and compute means per group for each metric
data = []   # List of dictionaries: each dictionary holds group -> list of values for one metric
means = []  # List of dictionaries: each dictionary holds group -> mean value for one metric
for metric_file in metric_files:
    metric_data = {}
    metric_means = {}
    for group in subdirs:
        values = results[group].get(metric_file, [])
        metric_data[group] = values
        metric_means[group] = np.mean(values) if values else None
    data.append(metric_data)
    means.append(metric_means)

# Write means and total rows to a CSV file (one row per metric, per group)
means_data = []
for metric_name, metric_means, metric_file in zip(metric_names, means, metric_files):
    for group in subdirs:
        subdir_path = os.path.join(base_dir, group)
        if not os.path.exists(subdir_path):
            print(f"Directory {subdir_path} does not exist. Skipping group {group}.")
            continue
        # Count total rows across run folders for each group and metric
        run_folders = [d for d in os.listdir(subdir_path)
                if os.path.isdir(os.path.join(subdir_path, d)) and d.startswith("run")]

        total_rows = sum(
            len(pd.read_csv(os.path.join(base_dir, group, "run"+str(run), metric_file)))
            for run in range(1, len(run_folders) + 1)
            if os.path.exists(os.path.join(base_dir, group, "run"+str(run), metric_file))
        )
        means_data.append({
            "Metric": metric_name,
            "Group": group,
            "Mean": metric_means[group],
            "Rows": total_rows
        })

# Plotting boxplots for all metrics and groups
fig, ax = plt.subplots(figsize=(14, 20))
num_metrics = len(metric_names)
positions = np.arange(num_metrics) * 11
width = 0.7
ax.set_ylim(-5, positions[-1] + 5.3)
separator_positions = [(positions[i] + positions[i+1]) / 2 for i in range(len(positions)-1)]
for pos in separator_positions:
    ax.axhline(y=pos, color="black", linestyle="-.", linewidth=0.8, alpha=1)
spacing = 1.6  # Adjust spacing between groups within each metric

legend_handles = {}
for i, (metric_data, metric_means, metric_name) in enumerate(zip(data, means, metric_names)):
    n_groups = len(subdirs)
    for j, group in enumerate(subdirs):
        offset = (j - (n_groups - 1) / 2) * spacing
        pos = positions[i] + offset
        values = metric_data.get(group, [])
        mean_val = metric_means.get(group)
        meanprops = dict(marker='^', markerfacecolor='#02bf21', markeredgecolor='none', markersize=12)
        medianprops = dict(color='orange', linewidth=2.5)
        bp = ax.boxplot(
            values,
            positions=[pos],
            widths=width,
            showmeans=True,
            vert=False,
            sym='',
            patch_artist=True,
            meanprops=meanprops,
            medianprops=medianprops,
            boxprops=dict(facecolor=colors[group]),
        )
        define_box_properties(bp, colors[group])
        if mean_val is not None:
            ax.annotate(f"{mean_val:.2f}", xy=(mean_val - 0.007, pos + 0.43),
                        fontsize=20, color="#000", fontweight='bold')
        if group not in legend_handles:
            key = ''
            if group == 'QueryRewriting':
                key = 'Base + Query Rewriting'
            elif group == 'ContextReduction':
                key = 'Base + Context Reduction'
            elif group == 'ContextSelection':
                key = 'Base + Context Selection'
            elif group == 'DualChunking':
                key = 'Base + Dual Chunking'
            elif group == 'Base':
                key = 'Base'
            elif group == "QAMR":
                key = "QAMR"
            legend_handles[key] = bp["boxes"][0]

ax.set_yticks(positions)
ax.set_yticklabels(metric_names, fontsize=22)
ax.set_xlabel("Percentage", fontsize=22)
ax.legend(
    list(legend_handles.values())[::-1],  
    list(legend_handles.keys())[::-1],    
    loc="lower left",
    fontsize=20
)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("Ablation-Study-Results.pdf", bbox_inches="tight")
plt.show()
# ----------------------- Statistical Testing Section -----------------------
# Function to compute A12
ab = ['Base','QueryRewriting','DualChunking','ContextReduction','ContextSelection']
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
for ablation in ab:
    # Prepare a list to store statistical test results
    stat_results_list = []

    # Iterate through each metric and perform statistical tests
    for metric_file, metric_name in zip(metric_files, metric_names):
        our_data = results["QAMR"].get(metric_file, [])
        basic_data = results[ablation].get(metric_file, [])
        
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
    print(f"\nStatistical Test Results for {ablation}:")
    print(stat_results)

    # Optionally, save the results to a CSV file
    stat_results.to_csv(f"statistical_test_results_QAMR_vs_{ablation}.csv", index=False)
