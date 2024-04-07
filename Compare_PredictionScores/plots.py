#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 13:51:57 2023

@author: Viktor
"""
########################################################
##### Create Scatter plots to check correlaltions ######
########################################################


# List of CSV files
csv_files1 = ["GPT_multimodal.csv", "GPT_unimodal.csv"]
csv_files2 = ["CLIP_ViT32.csv", "CLIP_RN50.csv", "llama_multimodal.csv", "llama_unimodal.csv"]


for file in csv_files1:
    # Read the CSV into a DataFrame
    df = pd.read_csv(file)
        
    # Group by 'Video' and 'ID', then calculate the mean for 'Response'
    grouped_response = df.groupby(['Video', 'ID'])['Response'].mean().reset_index()
    
    # Group by 'Video' to get the unique 'Model_Response'
    unique_model_response = df.groupby('Video')['Model_Response'].first().reset_index()
    
    # Group by 'Video' to get the average 'Response'
    avg_response = grouped_response.groupby('Video')['Response'].mean().reset_index()
    
    # Merge the two DataFrames on the 'Video' column
    final_df = pd.merge(avg_response, unique_model_response, on='Video')
    
    # Extract the average 'Response' and unique 'Model_Response'
    x_values = final_df['Response']
    y_values = final_df['Model_Response']
    
    # Create a scatter plot
    plt.figure(figsize=(10, 10), dpi=300)
    plt.scatter(x_values, y_values, c='blue', edgecolors='k', alpha=0.6, s=50, label='Model_Response')
    plt.scatter(x_values, x_values, c='red', edgecolors='k', alpha=0.6, s=50, label='Human Response')  # Note that we are using x_values for both x and y here for the "Response"
    
    plt.title(f'Scatter Plot for {file}', fontsize=18)
    plt.xlabel('Average Human Response per Video', fontsize=14)
    plt.ylabel('Model Response per Video', fontsize=14)
    plt.grid(True)
    plt.legend(loc='upper left', fontsize=12)
    
    # Save the figure in high resolution
    plt.savefig(f"{file.split('.')[0]}_scatterplot.png", dpi=300)

    # Show the plot
    plt.show()

for file in csv_files2:
    # Read the CSV into a DataFrame
    df = pd.read_csv(file)
    
    df['Model_Response'] = np.sqrt(df['Model_Response'])

    # Group by 'Video' and 'ID', then calculate the mean for 'Response'
    grouped_response = df.groupby(['Video', 'ID'])['Response'].mean().reset_index()
    
    # Group by 'Video' to get the unique 'Model_Response'
    unique_model_response = df.groupby('Video')['Model_Response'].first().reset_index()
    
    # Group by 'Video' to get the average 'Response'
    avg_response = grouped_response.groupby('Video')['Response'].mean().reset_index()
    
    # Merge the two DataFrames on the 'Video' column
    final_df = pd.merge(avg_response, unique_model_response, on='Video')
    
    # Extract the average 'Response' and unique 'Model_Response'
    x_values = final_df['Response']
    y_values = final_df['Model_Response']
    
    # Create a scatter plot
    plt.figure(figsize=(10, 10), dpi=300)
    plt.scatter(x_values, y_values, c='blue', edgecolors='k', alpha=0.6, s=50, label='Model_Response')
    plt.scatter(x_values, x_values, c='red', edgecolors='k', alpha=0.6, s=50, label='Human Response')  # Note that we are using x_values for both x and y here for the "Response"
    
    plt.title(f'Scatter Plot for {file}', fontsize=18)
    plt.xlabel('Average Human Response per Video', fontsize=14)
    plt.ylabel('Model Response per Video', fontsize=14)
    plt.grid(True)
    plt.legend(loc='upper left', fontsize=12)
    
    # Save the figure in high resolution
    plt.savefig(f"{file.split('.')[0]}_scatterplot.png", dpi=300)

    # Show the plot
    plt.show()


########################################################
##### Create Scatter plots version 2 ###################
########################################################

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# List of CSV files
csv_files1 = ["GPT_multimodal.csv", "GPT_unimodal.csv"]
csv_files2 = ["CLIP_ViT32.csv", "CLIP_RN50.csv", "llama_multimodal.csv", "llama_unimodal.csv", "CLIP_Vit32_unimodal.csv"]

def plot_data(file, sqrt_transform=False):
    # Read the CSV into a DataFrame
    df = pd.read_csv(file)
    
    if sqrt_transform:
        df['Model_Response'] = np.sqrt(df['Model_Response'])

    # Grouping and data extraction
    grouped_response = df.groupby(['Video', 'ID'])['Response'].mean().reset_index()
    unique_model_response = df.groupby('Video')['Model_Response'].first().reset_index()
    avg_response = grouped_response.groupby('Video')['Response'].mean().reset_index()
    final_df = pd.merge(avg_response, unique_model_response, on='Video')
    x_values = final_df['Response']
    y_values = final_df['Model_Response']
    
    # Plotting
    plt.figure(figsize=(10, 10), dpi=300)
    
    # Scatter plot for Model Response
    plt.scatter(x_values, y_values, c='blue', edgecolors='k', alpha=0.6, s=50, label='Model Response')
    
    # Scatter plot for Human Response and regression line for Model Response
    plt.scatter(x_values, x_values, c='red', edgecolors='k', alpha=0.6, s=50, label='Human Response')
    sns.regplot(x=x_values, y=y_values, scatter=False, line_kws={"color": "black", "alpha": 0.7})
    
    title_name = os.path.splitext(file)[0]
    plt.title(f'Scatter Plot for {title_name}', fontsize=18)
    plt.xlabel('Average Human Response per Video', fontsize=14)
    plt.ylabel('Model Response per Video', fontsize=14)
    plt.legend(loc='upper left', fontsize=12)
    
    # Save and show
    plt.savefig(f"{title_name}_scatterplot.png", dpi=300)
    plt.show()

for file in csv_files1:
    plot_data(file, False)

for file in csv_files2:
    plot_data(file, True)



########################################################
##### Create Scatter plots version 2, TOP25% ONLY ######
########################################################

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_data(file, sqrt_transform=False):
    # Read the CSV into a DataFrame
    df = pd.read_csv(file)
    
    if sqrt_transform:
        df['Model_Response'] = np.sqrt(df['Model_Response'])

    # Determine the top 25% threshold for Model Response
    threshold = df['Model_Response'].quantile(0.75)
    
    # Select only the top 25% of Model Responses
    top_df = df[df['Model_Response'] >= threshold]

    # Grouping and data extraction
    grouped_response = top_df.groupby(['Video', 'ID'])['Response'].mean().reset_index()
    unique_model_response = top_df.groupby('Video')['Model_Response'].first().reset_index()
    avg_response = grouped_response.groupby('Video')['Response'].mean().reset_index()
    
    # Merge to match the top 25% Model Responses with their corresponding Human Responses
    final_df = pd.merge(avg_response, unique_model_response, on='Video')
    
    # Get the x and y values
    x_values = final_df['Response']
    y_values = final_df['Model_Response']
    
    # Plotting
    plt.figure(figsize=(10, 10), dpi=300)
    
    # Scatter plot for Model Response
    plt.scatter(x_values, y_values, c='blue', edgecolors='k', alpha=0.6, s=50, label='Top 25% Model Response')
    
    # Scatter plot for Human Response and regression line for Model Response
    plt.scatter(x_values, x_values, c='red', edgecolors='k', alpha=0.6, s=50, label='Human Response')
    sns.regplot(x=x_values, y=y_values, scatter=False, line_kws={"color": "black", "alpha": 0.7})
    
    # Title and labels
    title_name = os.path.splitext(file)[0]
    plt.title(f'Top 25% Scatter Plot for {title_name}', fontsize=18)
    plt.xlabel('Average Human Response per Video', fontsize=14)
    plt.ylabel('Model Response per Video', fontsize=14)
    plt.legend(loc='upper left', fontsize=12)
    
    # Save and show
    plt.savefig(f"{title_name}_top25_scatterplot.png", dpi=300)
    plt.show()

# Assuming csv_files1 and csv_files2 are already defined
for file in csv_files1 + csv_files2:
    sqrt_transform = 'unimodal' not in file
    plot_data(file, sqrt_transform)



########################################################
##### Basic Scatterplot ################################
########################################################

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

# List of CSV files
csv_files1 = ["GPT_multimodal.csv", "GPT_unimodal.csv"]
csv_files2 = ["CLIP_ViT32.csv", "CLIP_RN50.csv", "llama_multimodal.csv", "llama_unimodal.csv", "CLIP_Vit32_unimodal.csv"]

def significance_marker(p_value):
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return ''

def plot_data(file, sqrt_transform=False):
    # Read the CSV into a DataFrame
    df = pd.read_csv(file)
    
    if sqrt_transform:
        df['Model_Response'] = np.sqrt(df['Model_Response'])

    # Simplify the DataFrame
    df = df.groupby('Video').agg({'Model_Response': 'first', 'Response': 'mean'}).reset_index()
    df = df.dropna()

    # Regression and significance testing
    slope, intercept, r_value, p_value, std_err = linregress(df['Model_Response'], df['Response'])
    print(file)
    print(p_value)

    # Plotting
    plt.figure(figsize=(10, 10), dpi=300)
    plt.scatter(df['Model_Response'], df['Response'], c='blue', edgecolors='k', alpha=0.6, s=50)

    # Add regression line if significant
    if p_value < 0.05 or file == "llama_multimodal.csv":
        sns.regplot(x='Model_Response', y='Response', data=df, scatter=False, line_kws={"color": "black", "alpha": 0.7})
        plt.text(0.05, 0.95, significance_marker(p_value), horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes, fontsize=16)

    title_name = os.path.splitext(file)[0]
#    plt.title(f'Scatter Plot for {title_name}', fontsize=18)
#    plt.xlabel('Model Response per Video', fontsize=14)
#    plt.ylabel('Average Human Response per Video', fontsize=14)

    # Save and show
    plt.savefig(f"{title_name}_scatterplot.png", dpi=600)
    plt.show()

for file in csv_files1:
    plot_data(file, False)

for file in csv_files2:
    plot_data(file, True)




########################################################
##### Top25% Basic Scatterplot #########################
########################################################

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

# List of CSV files
csv_files1 = ["GPT_multimodal.csv", "GPT_unimodal.csv"]
csv_files2 = ["CLIP_ViT32.csv", "CLIP_RN50.csv", "llama_multimodal.csv", "llama_unimodal.csv", "CLIP_Vit32_unimodal.csv"]

def significance_marker(p_value):
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return ''

def plot_data(file, sqrt_transform=False):
    # Read the CSV into a DataFrame
    df = pd.read_csv(file)
    
    if sqrt_transform:
        df['Model_Response'] = np.sqrt(df['Model_Response'])

    # Simplify the DataFrame
    df = df.groupby('Video').agg({'Model_Response': 'first', 'Response': 'mean'}).reset_index()
    df = df.dropna()
    
    
    # Determine the top 25% threshold for Model Response
    threshold = df['Model_Response'].quantile(0.75)
    
    # Select only the top 25% of Model Responses
    df = df[df['Model_Response'] >= threshold]

    # Regression and significance testing
    slope, intercept, r_value, p_value, std_err = linregress(df['Model_Response'], df['Response'])
    print(file)
    print(p_value)

    # Plotting
    plt.figure(figsize=(10, 10), dpi=600)
    plt.scatter(df['Model_Response'], df['Response'], c='blue', edgecolors='k', alpha=0.6, s=50)

    # Add regression line if significant
    if p_value < 0.05:
        sns.regplot(x='Model_Response', y='Response', data=df, scatter=False, line_kws={"color": "black", "alpha": 0.7})
        plt.text(0.05, 0.95, significance_marker(p_value), horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes, fontsize=16)

    title_name = os.path.splitext(file)[0]
#    plt.title(f'Scatter Plot for {title_name}', fontsize=18)
#    plt.xlabel('Model Response per Video', fontsize=14)
#    plt.ylabel('Average Human Response per Video', fontsize=14)

    # Save and show
    plt.savefig(f"{title_name}_scatterplot.png", dpi=600)
    plt.show()

for file in csv_files1:
    plot_data(file, False)

for file in csv_files2:
    plot_data(file, True)




####################################################
##### Create Bar plots to show correlaltions ######
###################################################


import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_rgb

# Function to normalize the values for alpha range
def normalize_alpha(values, alpha_min=0.4, alpha_max=1):
    v_min, v_max = min(values), max(values)
    return [(alpha_min + (alpha_max - alpha_min) * (v - v_min) / (v_max - v_min)) for v in values]

# Data
models = [
    'Llama_Multimodal', 'Llama_Unimodal (text only)', 'GPT_4_Multimodal_Prompt', 'GPT_4_Unimodal_Prompt (text only)',
    'CLIP_VIT32', 'CLIP_RN50', 'Top 25% CLIP_VIT32', 'Top 25% CLIP_RN50', 'Top 25% GPT_4 Multimodal',
    'Top 25% Llama_Multimodal', 'Bottom 25% CLIP_VIT32', 'Bottom 25% CLIP_RN50', 'Bottom 25% GPT_4 Multimodal', 
    'Bottom 25% Llama_Multimodal'
]
values = [0.13, -0.10, 0.30, 0.15, 0.23, 0.25, 0.40, 0.19, 0.15, 0.06, 0.26, 0.26, 0.31, 0.11]

# Colors and alpha values
base_colors = sns.color_palette("colorblind")

colors = {
    'Llama': base_colors[0],
    'CLIP_VIT32': base_colors[1],
    'CLIP_RN50': base_colors[2],
    'GPT-4': base_colors[4]
}

alpha_values = normalize_alpha(values)

# Set the style and size
sns.set_style("whitegrid")
plt.figure(figsize=(16, 8))

# Create the bar plot with custom colors and transparency
bars = sns.barplot(x=models, y=values, palette=[colors['Llama'], colors['Llama'], colors['GPT-4'], colors['GPT-4'], 
                                                colors['CLIP_VIT32'], colors['CLIP_RN50'], colors['CLIP_VIT32'], colors['CLIP_RN50'],
                                                colors['GPT-4'], colors['Llama'], colors['CLIP_VIT32'], colors['CLIP_RN50'], 
                                                colors['GPT-4'], colors['Llama']])

# Applying alpha values to the bars
for bar, alpha in zip(bars.patches, alpha_values):
    bar.set_alpha(alpha)

# Draw dotted lines for Human Ceilings
plt.axhline(0.41, color='grey', linestyle='dotted', label="Human Ceiling")
plt.axhline(0.63, color='grey', linestyle='dotted', label="Top 25% Human Ceiling")

# Rotate x-labels for better visibility
bars.set_xticklabels(models, rotation=45, ha='right')

# Labels, title, and legend
plt.xlabel('Models', fontsize=14)
plt.ylabel('Correlation-Value', fontsize=14)
plt.title('Comparison of Models', fontsize=16)
plt.tight_layout()

# Custom legend for groups in top left corner
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color=colors[key], lw=4, label=key) for key in colors.keys()]
plt.legend(handles=legend_elements, loc='upper left')

# Save the plot in high quality
plt.savefig('transparency_chart.png', dpi=300)
plt.show()



##########################
#Plot separate Bar Graphs#
##########################

##### OVERALL
plt.figure(figsize=(16, 8))

# Create the bar plot with adjusted positions
bars_overall = plt.bar(positions_overall, values_overall, width=0.6, color=[colors['Llama'], colors['Llama'], colors['GPT-4'], 
                                                                     colors['GPT-4'], colors['CLIP_VIT32'], colors['CLIP_RN50']])
# Apply transparency based on value
for bar, alpha in zip(bars_overall, alpha_overall):
    bar.set_alpha(alpha)

# Draw Human Ceiling and add to the legend
human_ceiling_line = plt.axhline(0.41, color='grey', linestyle='dotted', label='Human Ceiling')

# Set y-axis limits
plt.ylim(-0.13, 0.65)

# Rotate x-labels for better visibility
plt.xticks(positions_overall, models_overall, rotation=45, ha='right')

# Labels, title, and legend
plt.xlabel('Models', fontsize=14)
plt.ylabel('Correlation-Value', fontsize=14)
plt.title('Models Overall Comparison', fontsize=16)
plt.legend(handles=legend_elements + [human_ceiling_line], loc='upper left')

plt.tight_layout()
plt.show()




#### TOP 25%
plt.figure(figsize=(16, 8))

# Create the bar plot with adjusted positions
bars_top25 = plt.bar(positions_top25, values_top25, width=0.6, color=[colors['CLIP_VIT32'], colors['CLIP_RN50'], colors['GPT-4'], colors['Llama']])

# Apply transparency based on value
for bar, alpha in zip(bars_top25, alpha_top25):
    bar.set_alpha(alpha)

# Draw Human Ceiling for Top 25% and add to the legend
top25_human_ceiling_line = plt.axhline(0.63, color='grey', linestyle='dotted', label='Top 25% Human Ceiling')

# Set y-axis limits
plt.ylim(-0.13, 0.65)

# Rotate x-labels for better visibility
plt.xticks(positions_top25, models_top25, rotation=45, ha='right')

# Labels, title, and legend
plt.xlabel('Models', fontsize=14)
plt.ylabel('Correlation-Value', fontsize=14)
plt.title('Models Top 25% Comparison', fontsize=16)
plt.legend(handles=legend_elements + [top25_human_ceiling_line], loc='upper left')

plt.tight_layout()
plt.show()


### BOTTOM 25% 
plt.figure(figsize=(16, 8))

# Create the bar plot with adjusted positions
bars_bottom25 = plt.bar(positions_bottom25, values_bottom25, width=0.6, color=[colors['CLIP_VIT32'], colors['CLIP_RN50'], colors['GPT-4'], colors['Llama']])

# Apply transparency based on value
for bar, alpha in zip(bars_bottom25, alpha_bottom25):
    bar.set_alpha(alpha)

# Set y-axis limits
plt.ylim(-0.13, 0.65)

# Rotate x-labels for better visibility
plt.xticks(positions_bottom25, models_bottom25, rotation=45, ha='right')

# Labels, title
plt.xlabel('Models', fontsize=14)
plt.ylabel('Correlation-Value', fontsize=14)
plt.title('Models Bottom 25% Comparison', fontsize=16)

plt.tight_layout()
plt.show()


######################################################
#Plot separate Bar Graphs (2) -> Different Groupings #
######################################################

plt.figure(figsize=(16, 8))

# Group 1
values_group1 = [0.13, 0.30, -0.10, 0.15]
max_value = max(abs(min(values_group1)), max(values_group1))
alpha_group1 = [abs(value) / max_value for value in values_group1]
models_group1 = ['llama_multimodal', 'GPT_4_multimodal', 'llama_unimodal', 'GPT_4_unimodal']
colors_group1 = [colors['Llama'], colors['GPT-4'], colors['Llama'], colors['GPT-4']]

# Create narrower bars and adjust positions
width = 0.4  # New, narrower width
positions_group1 = range(len(values_group1))

bars_group1 = plt.bar(positions_group1, values_group1, width=width, color=colors_group1)

for bar, alpha in zip(bars_group1, alpha_group1):
    bar.set_alpha(alpha)

# Add human ceiling line and extend y-axis limits
plt.ylim(-0.11, 0.65)

human_ceiling_line = plt.axhline(0.41, color='grey', linestyle='dotted', label='Human Ceiling')

# Add labels, ticks, and titles
plt.xticks(positions_group1, models_group1, rotation=45, ha='right')
plt.xlabel('Models', fontsize=14)
plt.ylabel('Correlation-Value', fontsize=14)
plt.title('Group 1 Models Comparison', fontsize=16)

# Add legend including bars
plt.legend([bars_group1[0], bars_group1[1], human_ceiling_line], ['Llama Models', 'GPT-4 Models', 'Human Ceiling'], loc='upper left')

plt.tight_layout()

plt.show()




#### GROUP2
plt.figure(figsize=(16, 8))
values_group2 = [0.40, 0.19, 0.15, 0.06]  # Removed human ceiling as a bar
max_value = max(abs(min(values_group2)), max(values_group2))
alpha_group2 = [abs(value) / max_value for value in values_group2]
models_group2 = ['CLIP_ViT32 (Top 25%)', 'CLIP_RN50 (Top 25%)', 'GPT-4 (Top 25%)', 'Llama (Top 25%)']
colors_group2 = [colors['CLIP_VIT32'], colors['CLIP_RN50'], colors['GPT-4'], colors['Llama']]
positions_group2 = range(len(values_group2))

width = 0.4
bars_group2 = plt.bar(positions_group2, values_group2, width=width, color=colors_group2)

for bar, alpha in zip(bars_group2, alpha_group2):
    bar.set_alpha(alpha)

plt.ylim(-0.11, 0.65)
human_ceiling_line = plt.axhline(0.63, color='grey', linestyle='dotted', label='Human Ceiling (Top 25%)')

plt.xticks(positions_group2, models_group2, rotation=45, ha='right')
plt.xlabel('Models', fontsize=14)
plt.ylabel('Correlation-Value', fontsize=14)
plt.title('Group 2 Models Comparison', fontsize=16)

plt.legend([bar for bar in bars_group2] + [human_ceiling_line], models_group2 + ['Human Ceiling (Top 25%)'], loc='upper left')

plt.tight_layout()
plt.show()

##### Group 3
plt.figure(figsize=(16, 8))
values_group3 = [0.26, 0.26, 0.31, 0.11]  # Added Llama and GPT-4 in bottom 25%
max_value = max(abs(min(values_group3)), max(values_group3))
alpha_group3 = [abs(value) / max_value for value in values_group3]
models_group3 = ['CLIP_ViT32 (Bottom 25%)', 'CLIP_RN50 (Bottom 25%)', 'GPT-4 (Bottom 25%)', 'Llama (Bottom 25%)']
colors_group3 = [colors['CLIP_VIT32'], colors['CLIP_RN50'], colors['GPT-4'], colors['Llama']]
positions_group3 = range(len(values_group3))

width = 0.4
bars_group3 = plt.bar(positions_group3, values_group3, width=width, color=colors_group3)

for bar, alpha in zip(bars_group3, alpha_group3):
    bar.set_alpha(alpha)

plt.ylim(-0.11, 0.65)

plt.xticks(positions_group3, models_group3, rotation=45, ha='right')
plt.xlabel('Models', fontsize=14)
plt.ylabel('Correlation-Value', fontsize=14)
plt.title('Group 3 Models Comparison', fontsize=16)

plt.legend([bar for bar in bars_group3], models_group3, loc='upper left')

plt.tight_layout()
plt.show()

