# Only evaluate top 2 and top4
import pandas as pd
import warnings

# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", message="DataFrameGroupBy.apply operated on the grouping columns")

# Read the CSV file
df = pd.read_csv('predictions_with_model_scores.csv')

# Sort by PDB_ID and Vina_Rank to ensure proper ordering
df_sorted = df.sort_values(['PDB_ID', 'Vina_Rank'])

# Group by PDB_ID and calculate Vina success rates for top-N
def calculate_vina_success_rates(group):
    results = {}
    for n in [2, 4]:  # Only top 2 and top 4
        top_n = group[group['Vina_Rank'] <= n]
        if len(top_n) > 0:
            success_rate = top_n['Is_Good_Pose'].max()  # If any in top-N is good pose
            results[f'Vina_Top{n}'] = success_rate
        else:
            results[f'Vina_Top{n}'] = 0.0
    return pd.Series(results)

# Group by PDB_ID and calculate Vina success rates
vina_rates = df_sorted.groupby('PDB_ID', group_keys=False).apply(calculate_vina_success_rates)

# Calculate average Vina success rates
avg_vina_rates = vina_rates.mean()

# For combined (Vina + Model)
def calculate_combined_rates(group):
    results = {}

    # Top2: Vina top1 + Model top1 (where model top1 is not Vina top1)
    vina_top1 = group[group['Vina_Rank'] == 1]
    model_top1 = group[group['Vina_Rank'] != 1].nlargest(1, 'Model_Probability')  # Exclude Vina top1
    top2_combined = pd.concat([vina_top1, model_top1])
    results['Combined_Top2'] = top2_combined['Is_Good_Pose'].max() if len(top2_combined) > 0 else 0.0

    # Top4: Vina top2 + Model top2 (where model top2 are the 2 best by model prob, excluding Vina top1,2)
    vina_top2 = group[group['Vina_Rank'] <= 2]
    others_for_model = group[~group.index.isin(vina_top2.index)].nlargest(2, 'Model_Probability')
    top4_combined = pd.concat([vina_top2, others_for_model])
    results['Combined_Top4'] = top4_combined['Is_Good_Pose'].max() if len(top4_combined) > 0 else 0.0

    return pd.Series(results)

# Group by PDB_ID and calculate combined success rates
combined_rates = df.groupby('PDB_ID', group_keys=False).apply(calculate_combined_rates)

# Calculate average combined success rates
avg_combined_rates = combined_rates.mean()

# Count total number of unique PDBs
total_pdbs = len(df['PDB_ID'].unique())

# Calculate counts of successful cases for Vina
vina_counts = {}
for n in [2, 4]:  # Only top 2 and top 4
    vina_counts[f'Vina_Top{n}'] = int(vina_rates[f'Vina_Top{n}'].sum())

# Calculate counts of successful cases for Combined
combined_counts = {}
for n in [2, 4]:  # Only top 2 and top 4
    combined_counts[f'Combined_Top{n}'] = int(combined_rates[f'Combined_Top{n}'].sum())

# Print results
print("Vina Success Rates:")
for n in [2, 4]:  # Only top 2 and top 4
    rate = avg_vina_rates[f'Vina_Top{n}']
    count = vina_counts[f'Vina_Top{n}']
    print(f"Vina Top{n}: {rate:.4f} ({count}/{total_pdbs}) - {rate*100:.2f}%")

print("\nCombined (Vina + Model) Success Rates:")
for n in [2, 4]:  # Only top 2 and top 4
    rate = avg_combined_rates[f'Combined_Top{n}']
    count = combined_counts[f'Combined_Top{n}']
    print(f"Combined Top{n}: {rate:.4f} ({count}/{total_pdbs}) - {rate*100:.2f}%")

# Restore warnings
warnings.resetwarnings()
