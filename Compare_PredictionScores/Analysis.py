#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 11:15:44 2023

@author: Viktor
"""


# Set the  filename 
file = 'GPT_multimodal.csv'  #change for other files 


# Write updated_filtered_data to the CSV file
with open(file, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['ID', 'Response', 'Video', 'Model_Response'])
    writer.writeheader()
    for entry in final_dict:
        writer.writerow(entry)

print(f'Updated data has been written to {output_file}.')

##############################
####### PRINT CORRS ##########
##############################

# import csv file  PRINT OUT CORRS

import numpy as np
import pandas as pd
import scipy.stats

# List of CSV files
csv_files = ["GPT_multimodal.csv", "GPT_unimodal.csv", "CLIP_ViT32.csv", "CLIP_RN50.csv", "llama_multimodal.csv", "llama_unimodal.csv", "CLIP_Vit32_unimodal.csv"]

for file in csv_files:
    # Read the CSV file
    continuous_data = pd.read_csv(file)

    # Drop NA values
    continuous_data = continuous_data.dropna()

    # Apply square root transformation to Model_Response
 #   continuous_data['Model_Response'] = np.sqrt(continuous_data['Model_Response'])

    # Uncomment the following line to filter the data based on Response values
    # continuous_data = continuous_data[(continuous_data['Response'] <= 0.2) | (continuous_data['Response'] >= 0.8)]

    # Extract model and human data
    model = continuous_data["Model_Response"]
    human = continuous_data["Response"]

    set_A = model.to_numpy()
    set_B = human.to_numpy()

    # Calculate Pearson and Spearman correlations and their p-values
    pearson_corr, pearson_p = scipy.stats.pearsonr(set_A, set_B)
    spearman_corr, spearman_p = scipy.stats.spearmanr(set_A, set_B)

    print(f"File: {file}\nPearson Correlation: {pearson_corr}, p-value: {pearson_p}\nSpearman Correlation: {spearman_corr}, p-value: {spearman_p}\n")



###################################################
####### BOOTSTRAP FOR CONFIDENCE INTERVALS ########
###################################################

import numpy as np
import pandas as pd
import scipy.stats

# Function to perform bootstrapping and calculate correlations
def bootstrap_correlation(data, n_iterations=1000):
    pearson_corrs = []
    spearman_corrs = []

    for _ in range(n_iterations):
        # Resample the data with replacement
        resampled_data = data.sample(frac=1, replace=True)

        # Calculate correlations for the resampled data
        pearson_corr, _ = scipy.stats.pearsonr(resampled_data['Model_Response'], resampled_data['Response'])
        spearman_corr, _ = scipy.stats.spearmanr(resampled_data['Model_Response'], resampled_data['Response'])

        pearson_corrs.append(pearson_corr)
        spearman_corrs.append(spearman_corr)

    return pearson_corrs, spearman_corrs

# List of CSV files
csv_files = ["GPT_multimodal.csv", "GPT_unimodal.csv", "CLIP_ViT32.csv", "CLIP_RN50.csv", "llama_multimodal.csv", "llama_unimodal.csv", "CLIP_Vit32_unimodal.csv"]

# Iterate over each file and calculate bootstrapped correlations
for file in csv_files:
    continuous_data = pd.read_csv(file)

    # Preprocess the data
    continuous_data = continuous_data.dropna()
    continuous_data['Model_Response'] = np.sqrt(continuous_data['Model_Response'])

    # Perform bootstrapping
    pearson_corrs, spearman_corrs = bootstrap_correlation(continuous_data)

    # Calculate the confidence intervals
    pearson_lower, pearson_upper = np.percentile(pearson_corrs, [2.5, 97.5])
    spearman_lower, spearman_upper = np.percentile(spearman_corrs, [2.5, 97.5])

    print(f"File: {file}")
    print(f"Pearson Correlation: 2.5th percentile = {pearson_lower}, 97.5th percentile = {pearson_upper}")
    print(f"Spearman Correlation: 2.5th percentile = {spearman_lower}, 97.5th percentile = {spearman_upper}\n")


###############################
####### TOP QUARTILES ########
##############################

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# List of CSV files
csv_files = ["GPT_multimodal.csv", "GPT_unimodal.csv", "CLIP_ViT32.csv", "CLIP_RN50.csv", "llama_multimodal.csv", "llama_unimodal.csv", "CLIP_Vit32_unimodal.csv"]

for file in csv_files:
    # Read the CSV file
    continuous_data = pd.read_csv(file)

    # Drop NA values
    continuous_data = continuous_data.dropna()

    # Apply square root transformation to Model_Response
    continuous_data['Model_Response'] = np.sqrt(continuous_data['Model_Response'])

    # Extract model and human data
    model = continuous_data["Model_Response"].to_numpy()
    human = continuous_data["Response"].to_numpy()

    # Calculate the value that gives top 25% for model responses
    threshold_model = np.percentile(model, 75)

    # Get boolean mask for top 25% values in model responses
    mask_model = model >= threshold_model

    # Select corresponding elements in model and human responses
    top_model = model[mask_model]
    top_human = human[mask_model]

    # Perform Pearson's correlation
    corr, _ = pearsonr(top_model, top_human)

    print(f"File: {file} - Top Quartile Pearson Correlation: {corr}")
    

##################################################################
####### BOOTSTRAP FOR CONFIDENCE INTERVALS  TOP QUARTILES ########
##################################################################

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Function to perform bootstrapping and calculate correlations for top quartile
def bootstrap_top_quartile_correlation(data, n_iterations=1000):
    top_quartile_corrs = []

    # Calculate the value that gives top 25% for model responses
    threshold_model = np.percentile(data['Model_Response'], 75)

    # Get boolean mask for top 25% values in model responses
    mask_model = data['Model_Response'] >= threshold_model

    # Select corresponding elements in model and human responses
    top_model = data['Model_Response'][mask_model]
    top_human = data['Response'][mask_model]

    for _ in range(n_iterations):
        # Resample the top quartile data with replacement
        sample_indices = np.random.choice(len(top_model), len(top_model), replace=True)
        resampled_model = top_model.iloc[sample_indices]
        resampled_human = top_human.iloc[sample_indices]

        # Calculate Pearson correlation for the resampled data
        corr, _ = pearsonr(resampled_model, resampled_human)
        top_quartile_corrs.append(corr)

    return top_quartile_corrs

# List of CSV files
csv_files = ["GPT_multimodal.csv", "GPT_unimodal.csv", "CLIP_ViT32.csv", "CLIP_RN50.csv", "llama_multimodal.csv", "llama_unimodal.csv", "CLIP_Vit32_unimodal.csv"]

for file in csv_files:
    # Read the CSV file
    continuous_data = pd.read_csv(file)

    # Drop NA values and preprocess data
    continuous_data = continuous_data.dropna()
    continuous_data['Model_Response'] = np.sqrt(continuous_data['Model_Response'])

    # Perform bootstrapping for top quartile
    top_quartile_corrs = bootstrap_top_quartile_correlation(continuous_data)

    # Calculate the confidence intervals
    lower_bound, upper_bound = np.percentile(top_quartile_corrs, [2.5, 97.5])

    print(f"File: {file}")
    print(f"Top Quartile Correlation: 2.5th percentile = {lower_bound}, 97.5th percentile = {upper_bound}\n")





#### write the rows from top 25% to new CSV  to see what the clips are # Load your CSV file

clip_df = pd.read_csv("CLIP_ViT32.csv")

top_clips_df = clip_df[clip_df['Model_Response'].isin(top_set_A)]

# Drop duplicate rows based on the 'model_responses' column, keeping the first occurrence
top_clips_df = top_clips_df.drop_duplicates(subset='Model_Response', keep='first')


# Save the filtered DataFrame to a new CSV file
top_clips_df.to_csv("top_clips.csv", index=False)


# Add a constant term to the independent variable for the intercept
normalized_A_with_constant = sm.add_constant(normalized_A)

# Fit the OLS model
model = sm.OLS(normalized_B, normalized_A_with_constant)
results = model.fit()

results.summary()

# Calculate the predictions
predictions = results.predict(normalized_A_with_constant)

# Plot the original data and predictions
plt.scatter(normalized_A, normalized_B, alpha=0.5, label="Actual Data")
plt.scatter(normalized_A, predictions, color="red", alpha=0.5, label="Predictions")

# Plot the perfect predictions line
max_value = max(normalized_A.max(), normalized_B.max())
plt.plot([0, max_value], [0, max_value], linestyle="--", color="gray", label="Perfect Predictions")

# Display the p-value on the plot
p_value = results.pvalues[1]
#plt.text(0.05, 0.95, f"P-value: {p_value:.4f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

# Set labels and title
plt.xlabel("Set A")
plt.ylabel("Set B")
plt.title("Actual Data vs Predictions")
plt.legend()
plt.show()


# Get the R-squared value
r_squared = results.rsquared

# Plot the R-squared value
plt.figure(figsize=(4, 6))
plt.bar(['R-squared'], [r_squared], color='blue', alpha=0.7)
plt.ylim(0, 1)
plt.ylabel('Value')
plt.title('Variance Explained by the Model', fontsize=16)
plt.text('R-squared', r_squared, f"{r_squared:.2f}", color='black', ha='center', fontsize=12, va='bottom')

plt.tight_layout()
plt.show()




#### PLOT PEARSON CORRELATIOONS:

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Provided data
corrs = np.array([0.22, -0.045, 0.003, 0.37, 0.16])
ceiling = np.array([0.39, 0.48])
names = ["CLIP Overall", "CLIP Distorted\nOverall", "LSTM Overall", "CLIP Top Quartile", "CLIP Bottom Quartile"]
ps = np.array([0.0001, 0.006, 0.88, 0.0001, 0.001])
human_ceilings = ["Human Ceiling Overall", "Human Ceiling Top Quartile"]
bert_baseline = 0.07

# Create color palette: 'red' for "CLIP Top 25%" and "CLIP Low 25%", 'blue' for others
colors = ['blue' if name not in ["CLIP Top Quartile", "CLIP Bottom Quartile"] else 'red' for name in names]

# Calculate transparency levels based on correlation values
corrs_abs = np.abs(corrs)  # use absolute values for size
corrs_norm = (corrs_abs - np.min(corrs_abs)) / (np.max(corrs_abs) - np.min(corrs_abs))  # normalize to [0, 1]
alphas = 0.4 + corrs_norm * 0.6  # map to [0.4, 1] range

# Set style and context to "paper"
sns.set_style("whitegrid")
sns.set_context("paper")

# Create a larger figure
plt.figure(figsize=(10, 6), dpi=500)

# Create the bar plot with custom colors
bar_plot = sns.barplot(x=names, y=corrs, palette=colors, linewidth=0.5)

# Set bar widths and transparency
for bar, alpha in zip(bar_plot.patches, alphas):
    bar.set_width(0.5)  # Set width
    bar.set_alpha(alpha)  # Set transparency

# Add the p-values
for i, p_val in enumerate(ps):
    bar_plot.text(i, corrs[i] + (0.01 if corrs[i] > 0 else -0.01), f"p={p_val:.4f}", ha='center', va='center', fontsize=8)

# Add ceiling values as scatter lines
for i, c_val in enumerate(ceiling):
    color = 'red' if human_ceilings[i] == "Human Ceiling Top Quartile" else 'blue'
    plt.axhline(y=c_val, color=color, linestyle='--')
    plt.text(0 if i == 0 else len(corrs) - 1, c_val + 0.02, human_ceilings[i], ha='left' if i == 0 else 'right', color=color)

# Add BERT baseline as a scatter line
plt.axhline(y=bert_baseline, color='black', linestyle='--')
plt.text(len(corrs) - 1, bert_baseline + 0.02, "GPT-2 baseline", ha='right', color='black')

# Rotate x-axis labels
bar_plot.set_xticklabels(bar_plot.get_xticklabels(), rotation=20, horizontalalignment='right')

# Set y-axis limit
bar_plot.set_ylim([min(corrs) - 0.1, 0.5])

# Add labels and title
plt.xlabel('')
plt.ylabel('Correlation')
plt.title('Correlation values with p-values and human ceiling')

# Show the plot
plt.tight_layout()
plt.savefig("final_corrs.png",dpi=500)
plt.show()

plt.close()





####### Calculate human corrs btw.each other

### For each participant, correlate the maximal set of Videos with each other participant. Take the average of these correlations as ceiling 

import pandas as pd
import itertools
import numpy as np
from scipy.stats import pearsonr

human = continuous_data[["ID","Response","Video"]]

# Group by 'ID'
grouped = human.groupby('ID')

# Initialize a list to store the pairwise correlations
pairwise_correlations = []


# Loop through each pair and calculate correlation
for id1, id2 in participant_pairs:
    # Get data for the two participants and average duplicates
    data1 = grouped.get_group(id1).groupby('Video').mean().reset_index()
    data2 = grouped.get_group(id2).groupby('Video').mean().reset_index()

    # Find common set of videos
    common_videos = set(data1['Video']).intersection(set(data2['Video']))

    # Filter data to only include common videos
    data1_common = data1[data1['Video'].isin(common_videos)]
    data2_common = data2[data2['Video'].isin(common_videos)]

    # Calculate correlation for 'Response' of common videos
    if len(data1_common) == len(data2_common) and len(data1_common) >= 2:  # New check here
#        correlation, p_value = pearsonr(data1_common['Response'], data2_common['Response'])
        correlation, p_value = spearmanr(data1_common['Response'], data2_common['Response'])
        # Check if p-value is below 0.05 and the correlation is not 'nan'
        if p_value < 0.05 and not np.isnan(correlation):
            pairwise_correlations.append(correlation)

# Take the average of pairwise correlations to get the "ceiling"
if pairwise_correlations:  # Check if list is not empty
    ceiling = np.mean(pairwise_correlations)
    print("Ceiling (average of pairwise correlations):", ceiling)
else:
    print("No valid correlations found.")





