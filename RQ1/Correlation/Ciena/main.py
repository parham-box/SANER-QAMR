import pandas as pd
from scipy.stats import pearsonr

metric_names = [
    "Contextual Relevancy",
    "Contextual Recall",
    "Contextual Precision",
    "Answer Faithfulness",
    "Answer Relevancy",
    "Answer Correctness",
]
metric_files = [
    "crel_res_ciena.csv",
    "crec_res_ciena.csv",
    "cp_res_ciena.csv",
    "f_res_ciena.csv",
    "arel_res_ciena.csv",
    "ac_res_ciena.csv",
]
for metric_name, metric_file in zip(metric_names, metric_files):
    data = pd.read_csv(metric_file)

    # Calculate Pearson correlation and p-value
    correlation, p_value = pearsonr(data['score'], data['domain_experts_assessment'])

    # Print the results
    print(f"{metric_name}:")
    print(f"Pearson Correlation Coefficient: {correlation}")
    print(f"P-value: {p_value}")
