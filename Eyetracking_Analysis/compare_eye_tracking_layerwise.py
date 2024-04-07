import os
import numpy as np
import glob
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr, spearmanr

def apply_gaussian_smoothing(heatmap, sigma=30):
    return gaussian_filter(heatmap, sigma=sigma)

def apply_gaussian_smoothing_model(heatmap, sigma=30):
    return gaussian_filter(heatmap, sigma=sigma)

def calculate_probability_distribution(heatmap, show_plot=True):
    heatmap = apply_gaussian_smoothing(heatmap) #this is done before already 
    # Replace NaNs with 0 for probability distribution calculation
    prob_distribution = np.nan_to_num(heatmap.flatten(), nan=0.0)
    
    if show_plot:
        # Plot the heatmap
        plt.imshow(heatmap, cmap='hot', interpolation='nearest')
        plt.colorbar()  # To show the scale
        plt.title('Smoothed Heatmap Human')
        plt.show()
    
    # Normalize the distribution to sum to 1
    if np.sum(prob_distribution) != 0:
        prob_distribution /= np.sum(prob_distribution)
    return prob_distribution


### Calculate Probability Distribution on model side (add a threshold if necessary)
def calculate_probability_distribution_model(heatmap, show_plot=True):
        # Flatten the heatmap to calculate the percentile on the non-NaN values
    flat_heatmap = heatmap.flatten()
    flat_heatmap_nonan = flat_heatmap[~np.isnan(flat_heatmap)]
    
    # Calculate the 75th percentile value as the threshold
    threshold_value = np.percentile(flat_heatmap_nonan, 95)
    
  
    # Apply the threshold
    heatmap[heatmap < threshold_value] = 0.0 # set if necessary
    
    # Flatten the heatmap and replace NaNs with 0 for probability distribution calculation
    prob_distribution = np.nan_to_num(heatmap.flatten(), nan=0.0)
    
    if show_plot:
        # Plot the heatmap
        plt.imshow(heatmap, cmap='hot', interpolation='nearest')
        plt.colorbar()  # To show the scale
        plt.title('Smoothed Heatmap Model')
        plt.show()
    
    # Normalize the distribution to sum to 1
    total = np.sum(prob_distribution)
    if total != 0:
        prob_distribution /= total
    return prob_distribution

def calculate_correlation(distribution1, distribution2):
    if np.all(np.isnan(distribution1)) or np.all(np.isnan(distribution2)):
        return None
    assert distribution1.shape == distribution2.shape
    distribution1 = np.nan_to_num(distribution1, nan=0.0)
    distribution2 = np.nan_to_num(distribution2, nan=0.0)
    pearson_corr, _ = pearsonr(distribution1, distribution2)
    spearman_corr, _ = spearmanr(distribution1, distribution2)
    return pearson_corr, spearman_corr


# This assumes that the model and human heatmaps are named in a way
# that allows them to be matched by segment number.
human_folder_path = "Numpy_Attention_Matrices_normalised"
model_folder_path = "Numpy_Attention_layerwise"

# Dictionary to hold all correlations by segment
correlations_by_segment = {}


#############################################################
## Compute correlation for averaged participants, word, segment  #######
############################################################

import os
import glob
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Define previously mentioned functions here again if necessary

human_folder_path = "Eyetracking_Analysis/Numpy_Attention_Matrices_normalised"
model_folder_path = "Eyetracking_Analysis/Numpy_Attention_layerwise"
correlations_by_word_segment_layer = {}

# Loop over segment numbers and layers
for segment_number in range(1, 16):
    # Dictionary to accumulate human heatmaps for averaging
    human_heatmaps_accumulator = {}

    # Build the filename pattern for the current segment
    human_pattern = f"*segment_{segment_number}.npy"
    human_files = glob.glob(os.path.join(human_folder_path, human_pattern))

    # Loop over all human files to accumulate heatmaps
    for human_file in human_files:
        # Extract word
        word = os.path.basename(human_file).split('_')[1]

        # Initialize the list for the word if it doesn't exist
        if word not in human_heatmaps_accumulator:
            human_heatmaps_accumulator[word] = []

        # Load and accumulate the heatmaps
        human_heatmap = np.load(human_file)
        human_heatmaps_accumulator[word].append(human_heatmap)

    # Loop over layers
    for layer in range(12):
        segment_number_model = segment_number * 10
        model_pattern = f"*_{segment_number_model}.png_layer_{layer}*.npy"
        model_files = glob.glob(os.path.join(model_folder_path, model_pattern))
        

        # Calculate the average heatmap and correlation for each word
        for word in human_heatmaps_accumulator.keys():
            average_human_heatmap = np.nanmean(human_heatmaps_accumulator[word], axis=0)
            human_prob_dist = calculate_probability_distribution(average_human_heatmap, show_plot=False)
            
            # Find the corresponding model file
            corresponding_model_file = next((f for f in model_files if word in os.path.basename(f)), None)
            print(corresponding_model_file)
            if not corresponding_model_file:
                continue  # If there's no corresponding model file, skip to the next

            # Load the model heatmap
            model_heatmap = np.load(corresponding_model_file)
            model_prob_dist = calculate_probability_distribution_model(model_heatmap, show_plot=False)

            # Calculate the correlation for this specific segment, word, and layer
            correlation = calculate_correlation(human_prob_dist.flatten(), model_prob_dist.flatten())
            
            correlations_by_word_segment_layer[(segment_number, word, layer)] = correlation
            

# Save correlations for each layer
save_path = "results/correlations_by_layer"
os.makedirs(save_path, exist_ok=True)

for key, (pearson_corr, spearman_corr) in correlations_by_word_segment_layer.items():
    segment, word, layer = key
    filename = f"{save_path}/correlation_segment_{segment}_word_{word}_layer_{layer}.npy"
    np.save(filename, {'pearson': pearson_corr, 'spearman': spearman_corr})
    print(f'Saved correlation for Segment: {segment}, Word: {word}, Layer: {layer} to {filename}')




#### LOAD FILES
average_correlations = np.load("Eyetracking_Analysis/average_correlations.npy", allow_pickle=True).item()
#correlations_by_word_segment_layer = np.load("correlations_by_word_segment_layer.npy",allow_pickle=True).item()


###############################
###Model Layerwise Correlations ###
##############################

import numpy as np
from collections import defaultdict

# Assuming correlations_by_word_segment is loaded or defined earlier
# correlations_by_word_segment = ...

# Separate and accumulate correlations by layer
layerwise_pearson = defaultdict(list)
layerwise_spearman = defaultdict(list)

for (segment, word, layer), (pearson_corr, spearman_corr) in correlations_by_word_segment_layer.items():
    layerwise_pearson[layer].append(pearson_corr)
    layerwise_spearman[layer].append(spearman_corr)

# Compute and print nan-mean for each layer
print("Correlations from correlations_by_word_segment:")
for layer in layerwise_pearson:
    pearson_mean = np.nanmean(layerwise_pearson[layer])
    spearman_mean = np.nanmean(layerwise_spearman[layer])
    print(f"Layer: {layer}, Pearson Mean: {pearson_mean}, Spearman Mean: {spearman_mean}")



##########################################################
# BELOW ARE DIFFERENT WAYS OF COMPARING THE HEATMAPS #####
# TO COMPARE VALUES DIRECTLY SEEMS THE MOST CONSERVATIVE # 
# ALL WERE DONE IN OUR PREPRINT ##########################
# BUT OTHER WAYS MAY ALSO MAKE SENSE #####################
# DEPENDING ON WHAT YOU ARE INTERESTED IN ################
##########################################################

##################################
#### Compare values directly######
##################################


import numpy as np
from scipy.stats import spearmanr

# Assuming average_correlations and correlations_by_word_segment_layer are loaded or defined earlier

# Function to remove NaN values and separate values by layer
def filter_and_separate_by_layer(correlations_dict):
    layerwise_values = {}
    for (segment, word, layer), (pearson_corr, spearman_corr) in correlations_dict.items():
        if not np.isnan(pearson_corr) and not np.isnan(spearman_corr):
            if layer not in layerwise_values:
                layerwise_values[layer] = {'pearson': [], 'spearman': []}
            layerwise_values[layer]['pearson'].append(pearson_corr)
            layerwise_values[layer]['spearman'].append(spearman_corr)
    return layerwise_values

# Filter and separate values by layer
layerwise_correlations = filter_and_separate_by_layer(correlations_by_word_segment_layer)

# Iterate through layers to compute and print correlations
for layer, correlations in layerwise_correlations.items():
    # Find common segment-word pairs for the current layer in average_correlations
    common_keys = [(segment, word) for segment, word in average_correlations if (segment, word, layer) in correlations_by_word_segment_layer]

    # Collect correlation values for common keys
    human_human_values_pearson = [average_correlations[key][0] for key in common_keys]
    human_model_values_pearson = [correlations['pearson'][i] for i, key in enumerate(common_keys)]
    human_human_values_spearman = [average_correlations[key][1] for key in common_keys]
    human_model_values_spearman = [correlations['spearman'][i] for i, key in enumerate(common_keys)]

    # Calculate and print direct correlations for the layer
    direct_correlation_pearson, p_value_pearson = spearmanr(human_human_values_pearson, human_model_values_pearson)
    direct_correlation_spearman, p_value_spearman = spearmanr(human_human_values_spearman, human_model_values_spearman)

    print(f"Layer: {layer}")
    print(f"  Direct Pearson correlation: {direct_correlation_pearson}, P-value: {p_value_pearson}")
    print(f"  Direct Spearman correlation: {direct_correlation_spearman}, P-value: {p_value_spearman}")

    # Interpretation of the results
    # Interpretation code here (similar to what you have already provided)
    # Interpret the Pearson result
    if p_value_pearson < 0.05:
        print("The direct Pearson correlation is statistically significant.")
        if direct_correlation_pearson > 0:
            print("Positive significant Pearson correlation.")
        elif direct_correlation_pearson < 0:
            print("Negative significant Pearson correlation.")
    else:
        print("The direct Pearson correlation is not statistically significant.")

    # Interpret the Spearman result
    if p_value_spearman < 0.05:
        print("The direct Spearman correlation is statistically significant.")
        if direct_correlation_spearman > 0:
            print("Positive significant Spearman correlation.")
        elif direct_correlation_spearman < 0:
            print("Negative significant Spearman correlation.")
    else:
        print("The direct Spearman correlation is not statistically significant.")
        
        
        

###############################
## PLOT ONLY POSITIVE SCORES ##
###############################


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict

# Load the dictionaries
# correlations_by_word_segment_layer = np.load("correlations_by_word_segment_layer.npy", allow_pickle=True).item()
average_correlations = np.load("Eyetracking_Analysis/average_correlations.npy", allow_pickle=True).item()

# Function to separate and filter correlations by layer
def separate_and_filter_by_layer(correlations_dict, average_dict):
    layerwise_values = defaultdict(lambda: {'positive_pearson': [], 'positive_spearman': []})
    for (segment, word, layer), (pearson_corr, spearman_corr) in correlations_dict.items():
        if (segment, word) in average_dict and pearson_corr > 0 and spearman_corr > 0:
            layerwise_values[layer]['positive_pearson'].append((pearson_corr, average_dict[(segment, word)][0]))
            layerwise_values[layer]['positive_spearman'].append((spearman_corr, average_dict[(segment, word)][1]))
    return layerwise_values

# Separate and filter values by layer
layerwise_correlations = separate_and_filter_by_layer(correlations_by_word_segment_layer, average_correlations)

# Colorblind-friendly colors
color_model = "#1f77b4"  # Muted blue
color_human = "#ff7f0e"  # Safety orange

# Iterate through layers to plot
for layer, correlations in layerwise_correlations.items():
    # Pearson plot
    human_model_pearson = [corr[0] for corr in correlations['positive_pearson']]
    human_human_pearson = [corr[1] for corr in correlations['positive_pearson']]
    mean_pearson_human_model = np.mean(human_model_pearson)
    mean_pearson_human_human = np.mean(human_human_pearson)
    n_pearson = len(human_model_pearson)

    plt.figure(figsize=(10, 6), dpi=100)
    sns.barplot(x=np.arange(n_pearson), y=human_model_pearson, color=color_model, label=f'Human-Model (Mean: {mean_pearson_human_model:.2f}, N: {n_pearson})', alpha=0.6)
    sns.barplot(x=np.arange(n_pearson), y=human_human_pearson, color=color_human, label=f'Human-Human (Mean: {mean_pearson_human_human:.2f}, N: {n_pearson})', alpha=0.6)
    plt.title(f'Positive Pearson Correlation Comparison for Layer {layer}')
    plt.ylabel('Correlation Value')
    plt.xlabel('Word, Segment Index')
    plt.legend()
    plt.show()

    # Spearman plot
    human_model_spearman = [corr[0] for corr in correlations['positive_spearman']]
    human_human_spearman = [corr[1] for corr in correlations['positive_spearman']]
    mean_spearman_human_model = np.mean(human_model_spearman)
    mean_spearman_human_human = np.mean(human_human_spearman)
    n_spearman = len(human_model_spearman)

    plt.figure(figsize=(10, 6), dpi=100)
    sns.barplot(x=np.arange(n_spearman), y=human_model_spearman, color=color_model, label=f'Human-Model (Mean: {mean_spearman_human_model:.2f}, N: {n_spearman})', alpha=0.6)
    sns.barplot(x=np.arange(n_spearman), y=human_human_spearman, color=color_human, label=f'Human-Human (Mean: {mean_spearman_human_human:.2f}, N: {n_spearman})', alpha=0.6)
    plt.title(f'Positive Spearman Correlation Comparison for Layer {layer}')
    plt.ylabel('Correlation Value')
    plt.xlabel('Word, Segment Index')
    plt.legend()
    plt.show()



    
        
###############################
### PLOT 25% (HUMAN SIDE) #####
###############################
        
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict

# Assuming correlations_by_word_segment_layer and average_correlations are loaded or defined earlier

# Function to separate correlations by layer
def separate_by_layer(correlations_dict):
    layerwise_values = defaultdict(lambda: {'pearson': [], 'spearman': []})
    for (segment, word, layer), (pearson_corr, spearman_corr) in correlations_dict.items():
        if pearson_corr > 0 and spearman_corr > 0:
            layerwise_values[layer]['pearson'].append((segment, word, pearson_corr))
            layerwise_values[layer]['spearman'].append((segment, word, spearman_corr))
    return layerwise_values

# Separate values by layer
layerwise_correlations = separate_by_layer(correlations_by_word_segment_layer)

# Iterate through layers to plot
for layer, correlations in layerwise_correlations.items():
    # Sort and get the top 25% positive correlations
    top_25_pearson = sorted(correlations['pearson'], key=lambda x: x[2], reverse=True)[:len(correlations['pearson'])//4]
    top_25_spearman = sorted(correlations['spearman'], key=lambda x: x[2], reverse=True)[:len(correlations['spearman'])//4]

    # Extract corresponding human-human scores
    top_pearson_human_human = [average_correlations[(seg, word)][0] for seg, word, _ in top_25_pearson]
    top_spearman_human_human = [average_correlations[(seg, word)][1] for seg, word, _ in top_25_spearman]

    # Extract the human-model scores
    top_pearson_human_model = [corr for _, _, corr in top_25_pearson]
    top_spearman_human_model = [corr for _, _, corr in top_25_spearman]

    # Create string labels for plot
    labels_pearson = [f"{seg},{word}" for seg, word, _ in top_25_pearson]
    labels_spearman = [f"{seg},{word}" for seg, word, _ in top_25_spearman]

    # Pearson plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x=labels_pearson, y=top_pearson_human_model, color='blue', label='Human-Model')
    sns.barplot(x=labels_pearson, y=top_pearson_human_human, color='orange', label='Human-Human')
    plt.title(f'Top 25% Positive Pearson Correlations for Layer {layer}')
    plt.ylabel('Correlation Value')
    plt.xlabel('Word, Segment')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()

    # Spearman plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x=labels_spearman, y=top_spearman_human_model, color='green', label='Human-Model')
    sns.barplot(x=labels_spearman, y=top_spearman_human_human, color='red', label='Human-Human')
    plt.title(f'Top 25% Positive Spearman Correlations for Layer {layer}')
    plt.ylabel('Correlation Value')
    plt.xlabel('Word, Segment')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()


##############################
      #  ONLY HUMAN HIGH  PREDICTABILITY #
##############################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict

# Load the CSV file
df = pd.read_csv("Eyetracking_Analysis/CLIP.csv")

# Calculate the 75th percentile (top 25%) for "Model_Response" and "Response"
threshold_model_response = df["Model_Response"].quantile(0.75)
threshold_response = df["Response"].quantile(0.75)

# Filter rows where only "Response" in top 25
top_videos = df[df["Response"] >= threshold_response]

# Extract the 'Video' column values
top_video_words = top_videos["Video"].tolist()

# Create a dictionary with the top 25% words
top_video_words_dict = {word: True for word in top_video_words}

# Load the dictionaries
#correlations_by_word_segment_layer = np.load("correlations_by_word_segment_layer.npy", allow_pickle=True).item()
average_correlations = np.load("Eyetracking_Analysis/average_correlations.npy", allow_pickle=True).item()

# Function to separate correlations by layer
def separate_by_layer(correlations_dict):
    layerwise_values = defaultdict(lambda: {'pearson': [], 'spearman': []})
    for (segment, word, layer), (pearson_corr, spearman_corr) in correlations_dict.items():
        if word in top_video_words_dict:
            layerwise_values[layer]['pearson'].append((segment, word, pearson_corr))
            layerwise_values[layer]['spearman'].append((segment, word, spearman_corr))
    return layerwise_values

# Separate values by layer
layerwise_correlations = separate_by_layer(correlations_by_word_segment_layer)

# Iterate through layers to plot
for layer, correlations in layerwise_correlations.items():
    # Filter correlations for words in top_video_words_dict
    filtered_pearson = [(seg, word, corr) for seg, word, corr in correlations['pearson'] if word in top_video_words_dict]
    filtered_spearman = [(seg, word, corr) for seg, word, corr in correlations['spearman'] if word in top_video_words_dict]

    # Calculate mean correlations for filtered values
    mean_filtered_pearson_human_model = np.mean([corr for _, _, corr in filtered_pearson])
    mean_filtered_pearson_human_human = np.mean([average_correlations[(seg, word)][0] for seg, word, _ in filtered_pearson])
    mean_filtered_spearman_human_model = np.mean([corr for _, _, corr in filtered_spearman])
    mean_filtered_spearman_human_human = np.mean([average_correlations[(seg, word)][1] for seg, word, _ in filtered_spearman])

    # Create string labels for plot
    labels_filtered_pearson = [f"{seg},{word}" for seg, word, _ in filtered_pearson]
    labels_filtered_spearman = [f"{seg},{word}" for seg, word, _ in filtered_spearman]

    # Pearson plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x=labels_filtered_pearson, y=[corr for _, _, corr in filtered_pearson], color='blue', label=f'Human-Model (Mean: {mean_filtered_pearson_human_model:.2f})')
    sns.barplot(x=labels_filtered_pearson, y=[average_correlations[(seg, word)][0] for seg, word, _ in filtered_pearson], color='orange', label=f'Human-Human (Mean: {mean_filtered_pearson_human_human:.2f})')
    plt.title(f'Top Positive Pearson Correlations for Layer {layer} (Filtered Words)')
    plt.ylabel('Correlation Value')
    plt.xlabel('Word, Segment')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()

    # Spearman plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x=labels_filtered_spearman, y=[corr for _, _, corr in filtered_spearman], color='green', label=f'Human-Model (Mean: {mean_filtered_spearman_human_model:.2f})')
    sns.barplot(x=labels_filtered_spearman, y=[average_correlations[(seg, word)][1] for seg, word, _ in filtered_spearman], color='red', label=f'Human-Human (Mean: {mean_filtered_spearman_human_human:.2f})')
    plt.title(f'Top Positive Spearman Correlations for Layer {layer} (Filtered Words)')
    plt.ylabel('Correlation Value')
    plt.xlabel('Word, Segment')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()




##############################
#  BOTH MODEL AND HUMAN HIGH PREDICTABILITY #
##############################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict

# Load the CSV file
df = pd.read_csv("Eyetracking_Analysis/CLIP.csv")

# Calculate the 75th percentile (top 25%) for "Model_Response" and "Response"
threshold_model_response = df["Model_Response"].quantile(0.75)
threshold_response = df["Response"].quantile(0.75)

# Filter rows where both "Model_Response" and "Response" are in the top 25%
top_videos = df[(df["Model_Response"] >= threshold_model_response) & (df["Response"] >= threshold_response)]

# Extract the 'Video' column values
top_video_words = top_videos["Video"].tolist()

# Create a dictionary with the top 25% words
top_video_words_dict = {word: True for word in top_video_words}

# Load the dictionaries
#correlations_by_word_segment_layer = np.load("correlations_by_word_segment_layer.npy", allow_pickle=True).item()
average_correlations = np.load("Eyetracking_Analysis/average_correlations.npy", allow_pickle=True).item()

# Function to separate correlations by layer
def separate_by_layer(correlations_dict):
    layerwise_values = defaultdict(lambda: {'pearson': [], 'spearman': []})
    for (segment, word, layer), (pearson_corr, spearman_corr) in correlations_dict.items():
        if word in top_video_words_dict:
            layerwise_values[layer]['pearson'].append((segment, word, pearson_corr))
            layerwise_values[layer]['spearman'].append((segment, word, spearman_corr))
    return layerwise_values

# Separate values by layer
layerwise_correlations = separate_by_layer(correlations_by_word_segment_layer)

# Iterate through layers to plot
for layer, correlations in layerwise_correlations.items():
    # Filter correlations for words in top_video_words_dict
    filtered_pearson = [(seg, word, corr) for seg, word, corr in correlations['pearson'] if word in top_video_words_dict]
    filtered_spearman = [(seg, word, corr) for seg, word, corr in correlations['spearman'] if word in top_video_words_dict]

    # Calculate mean correlations for filtered values
    mean_filtered_pearson_human_model = np.mean([corr for _, _, corr in filtered_pearson])
    mean_filtered_pearson_human_human = np.mean([average_correlations[(seg, word)][0] for seg, word, _ in filtered_pearson])
    mean_filtered_spearman_human_model = np.mean([corr for _, _, corr in filtered_spearman])
    mean_filtered_spearman_human_human = np.mean([average_correlations[(seg, word)][1] for seg, word, _ in filtered_spearman])

    # Create string labels for plot
    labels_filtered_pearson = [f"{seg},{word}" for seg, word, _ in filtered_pearson]
    labels_filtered_spearman = [f"{seg},{word}" for seg, word, _ in filtered_spearman]

    # Pearson plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x=labels_filtered_pearson, y=[corr for _, _, corr in filtered_pearson], color='blue', label=f'Human-Model (Mean: {mean_filtered_pearson_human_model:.2f})')
    sns.barplot(x=labels_filtered_pearson, y=[average_correlations[(seg, word)][0] for seg, word, _ in filtered_pearson], color='orange', label=f'Human-Human (Mean: {mean_filtered_pearson_human_human:.2f})')
    plt.title(f'Top Positive Pearson Correlations for Layer {layer} (Filtered Words)')
    plt.ylabel('Correlation Value')
    plt.xlabel('Word, Segment')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()

    # Spearman plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x=labels_filtered_spearman, y=[corr for _, _, corr in filtered_spearman], color='green', label=f'Human-Model (Mean: {mean_filtered_spearman_human_model:.2f})')
    sns.barplot(x=labels_filtered_spearman, y=[average_correlations[(seg, word)][1] for seg, word, _ in filtered_spearman], color='red', label=f'Human-Human (Mean: {mean_filtered_spearman_human_human:.2f})')
    plt.title(f'Top Positive Spearman Correlations for Layer {layer} (Filtered Words)')
    plt.ylabel('Correlation Value')
    plt.xlabel('Word, Segment')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()



##############################
#  OVERALL #
##############################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict

# Load the CSV file
df = pd.read_csv("Eyetracking_Analysis/CLIP.csv")

# Calculate the 75th percentile (top 100%) for "Model_Response" and "Response"
threshold_model_response = df["Model_Response"].quantile(0.0)
threshold_response = df["Response"].quantile(0.0)

# Filter rows where both "Model_Response" and "Response" are in the top 25%
top_videos = df[(df["Model_Response"] >= threshold_model_response) & (df["Response"] >= threshold_response)]

# Extract the 'Video' column values
top_video_words = top_videos["Video"].tolist()

# Create a dictionary with the top 25% words
top_video_words_dict = {word: True for word in top_video_words}

# Load the dictionaries
#correlations_by_word_segment_layer = np.load("correlations_by_word_segment_layer.npy", allow_pickle=True).item()
average_correlations = np.load("Eyetracking_Analysis/average_correlations.npy", allow_pickle=True).item()

# Function to separate correlations by layer
def separate_by_layer(correlations_dict):
    layerwise_values = defaultdict(lambda: {'pearson': [], 'spearman': []})
    for (segment, word, layer), (pearson_corr, spearman_corr) in correlations_dict.items():
        if word in top_video_words_dict:
            layerwise_values[layer]['pearson'].append((segment, word, pearson_corr))
            layerwise_values[layer]['spearman'].append((segment, word, spearman_corr))
    return layerwise_values

# Separate values by layer
layerwise_correlations = separate_by_layer(correlations_by_word_segment_layer)

# Iterate through layers to plot
for layer, correlations in layerwise_correlations.items():
    # Filter correlations for words in top_video_words_dict
    filtered_pearson = [(seg, word, corr) for seg, word, corr in correlations['pearson'] if word in top_video_words_dict]
    filtered_spearman = [(seg, word, corr) for seg, word, corr in correlations['spearman'] if word in top_video_words_dict]

    # Calculate mean correlations for filtered values
    mean_filtered_pearson_human_model = np.mean([corr for _, _, corr in filtered_pearson])
    mean_filtered_pearson_human_human = np.mean([average_correlations[(seg, word)][0] for seg, word, _ in filtered_pearson])
    mean_filtered_spearman_human_model = np.mean([corr for _, _, corr in filtered_spearman])
    mean_filtered_spearman_human_human = np.mean([average_correlations[(seg, word)][1] for seg, word, _ in filtered_spearman])

    # Create string labels for plot
    labels_filtered_pearson = [f"{seg},{word}" for seg, word, _ in filtered_pearson]
    labels_filtered_spearman = [f"{seg},{word}" for seg, word, _ in filtered_spearman]

    # Pearson plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x=labels_filtered_pearson, y=[corr for _, _, corr in filtered_pearson], color='blue', label=f'Human-Model (Mean: {mean_filtered_pearson_human_model:.2f})')
    sns.barplot(x=labels_filtered_pearson, y=[average_correlations[(seg, word)][0] for seg, word, _ in filtered_pearson], color='orange', label=f'Human-Human (Mean: {mean_filtered_pearson_human_human:.2f})')
    plt.title(f'Top Positive Pearson Correlations for Layer {layer} (Filtered Words)')
    plt.ylabel('Correlation Value')
    plt.xlabel('Word, Segment')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()

    # Spearman plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x=labels_filtered_spearman, y=[corr for _, _, corr in filtered_spearman], color='green', label=f'Human-Model (Mean: {mean_filtered_spearman_human_model:.2f})')
    sns.barplot(x=labels_filtered_spearman, y=[average_correlations[(seg, word)][1] for seg, word, _ in filtered_spearman], color='red', label=f'Human-Human (Mean: {mean_filtered_spearman_human_human:.2f})')
    plt.title(f'Top Positive Spearman Correlations for Layer {layer} (Filtered Words)')
    plt.ylabel('Correlation Value')
    plt.xlabel('Word, Segment')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()
    
    
# Plot layerwise corrs


import matplotlib.pyplot as plt
import numpy as np

# Sample data: Replace these with your actual data
x_values = np.array([-0.01, -0.005, 0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035])
y_values = np.arange(len(x_values))
errors = np.random.uniform(0.001, 0.003, size=len(x_values))  # Sample error values

# Significance markers: Replace with your actual significance values
significance = ['***', '', '**', '*', '', '', '***', '**', '*', '']

# Create the plot
plt.errorbar(x_values, y_values, xerr=errors, fmt='o', color='darkorange', ecolor='lightgray', capsize=5)

# Add significance markers
for i, sig in enumerate(significance):
    plt.text(x_values[i], y_values[i], sig, ha='center', va='bottom')

# Customize the plot
plt.axvline(x=0, color='grey', linestyle='--')  # Zero line
plt.xlabel('βdistance')
plt.yticks(y_values)  # Set the y-ticks to match the y-values

# Show the plot
plt.show()


    

# Plot layerwise corrs Percentages 

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Data for each category
# Overall Correlations
overall_human_human = np.array([0.33] * 12)
overall_human_model = np.array([0.06, 0.00, 0.14, -0.07, 0.06, 0.12, 0.07, 0.10, 0.14, 0.10, 0.09, 0.05])

# Positive Model-Human Correlations
positive_human_human = np.array([0.32, 0.33, 0.34, 0.34, 0.33, 0.33, 0.32, 0.33, 0.32, 0.32, 0.31, 0.31])
positive_human_model = np.array([0.25, 0.30, 0.28, 0.25, 0.29, 0.31, 0.31, 0.33, 0.37, 0.36, 0.32, 0.34])
positive_human_model_N = np.array([145, 284, 348, 95, 205, 431, 366, 358, 504, 446, 425, 361])  # N values

# Top Quartile Human Predictability
top_quartile_human_predictability_human_human = np.array([0.33] * 12)
top_quartile_human_predictability_human_model = np.array([0.06, 0.02, 0.15, -0.06, 0.06, 0.13, 0.08, 0.11, 0.24, 0.22, 0.09, 0.05])

# Top Quartile Human and Model Predictability
top_quartile_predictability_human_human = np.array([0.32] * 12)
top_quartile_predictability_human_model = np.array([0.07, 0.00, 0.16, -0.08, 0.05, 0.12, 0.07, 0.10, 0.22, 0.21, 0.09, 0.04])


# Function to calculate percentage
def calculate_percentage(human_human, human_model):
    adjusted_human_model = np.maximum(human_model, 0)
    return (adjusted_human_model / human_human) * 100

# Calculating percentages for each category
percentage_overall = calculate_percentage(overall_human_human, overall_human_model)
percentage_positive = calculate_percentage(positive_human_human, positive_human_model)
percentage_top_quartile_human = calculate_percentage(top_quartile_human_predictability_human_human, top_quartile_human_predictability_human_model)
percentage_top_quartile_both = calculate_percentage(top_quartile_predictability_human_human, top_quartile_predictability_human_model)

# Preparing data for Seaborn plot
data = {
    "Layer": np.tile(np.arange(1, 13), 4),
    "Percentage": np.concatenate([percentage_overall, percentage_positive, percentage_top_quartile_human, percentage_top_quartile_both]),
    "Category": ["All Video Clips"] * 12 + 
                ["Video Clips with Positive Human-Model Correlation"] * 12 + 
                ["Video Clips Within Top Quartile of Human Predictability"] * 12 + 
                ["Video Clips Within Top Quartile Human and Model Predictability"] * 12,
    "N": np.concatenate([np.repeat(None, 12), positive_human_model_N, np.repeat(None, 24)])  # Including N values for Positive correlations only
}

# Creating a DataFrame for plotting
df = pd.DataFrame(data)

# Adjusting the plot settings for specified requirements
sns.set(style="white", rc={"figure.dpi": 600, "savefig.dpi": 600})

# Creating the plot
plt.figure(figsize=(15, 8))
plot = sns.lineplot(data=df, x="Layer", y="Percentage", hue="Category", style="Category", 
                    markers=True, ci="sd", palette="deep", linewidth=2.5, markersize=10)

# Removing the labels for x and y axes
plot.set_xlabel('')
plot.set_ylabel('')

# Setting ticks for the x and y axes, including 110% and 120%
plot.set_xticks(np.arange(1, 13))
plot.set_yticks(np.linspace(0, 120, 13))  # Extending the y-axis range to include 110% and 120%

# Enhancing the font size for ticks, labels, and legend
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plot.set_xlabel('', fontsize=14)
plot.set_ylabel('', fontsize=14)
plot.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='large')

# Adding percentage sign on y-axis labels
plot.set_yticklabels(['{}%'.format(int(y)) for y in np.linspace(0, 120, 13)])

# Removing the grid and the box around the plot
plot.grid(False)
plot.spines['top'].set_visible(False)
plot.spines['right'].set_visible(False)
plot.spines['bottom'].set_visible(True)
plot.spines['left'].set_visible(True)

# Adding N values as annotations for "Video Clips with Positive Human-Model Correlation" category
for layer, percentage, n in zip(df[df['Category'] == "Video Clips with Positive Human-Model Correlation"]['Layer'], 
                                df[df['Category'] == "Video Clips with Positive Human-Model Correlation"]['Percentage'], 
                                df[df['Category'] == "Video Clips with Positive Human-Model Correlation"]['N']):
    plt.text(layer, percentage, f"N: {n}", horizontalalignment='left', size='small', color='black', weight='semibold')

# Adding layer numbers on the x-axis
plot.set_xticks(np.arange(1, 13))
plot.set_xticklabels([str(x) for x in range(1, 13)])

plt.show()




##### SEPARATE PLOTS

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Data for each category
# Overall Correlations
overall_human_human = np.array([0.33] * 12)
overall_human_model = np.array([0.06, 0.00, 0.14, -0.07, 0.06, 0.12, 0.07, 0.10, 0.14, 0.10, 0.09, 0.05])

# Positive Model-Human Correlations
positive_human_human = np.array([0.32, 0.33, 0.34, 0.34, 0.33, 0.33, 0.32, 0.33, 0.32, 0.32, 0.31, 0.31])
positive_human_model = np.array([0.25, 0.30, 0.28, 0.25, 0.29, 0.31, 0.31, 0.33, 0.37, 0.36, 0.32, 0.34])
positive_human_model_N = np.array([145, 284, 348, 95, 205, 431, 366, 358, 504, 446, 425, 361])  # N values

# Top Quartile Human Predictability
top_quartile_human_predictability_human_human = np.array([0.33] * 12)
top_quartile_human_predictability_human_model = np.array([0.06, 0.02, 0.15, -0.06, 0.06, 0.13, 0.08, 0.11, 0.24, 0.22, 0.09, 0.05])

# Top Quartile Human and Model Predictability
top_quartile_predictability_human_human = np.array([0.32] * 12)
top_quartile_predictability_human_model = np.array([0.07, 0.00, 0.16, -0.08, 0.05, 0.12, 0.07, 0.10, 0.12, 0.11, 0.09, 0.04])

# Function to calculate percentage
def calculate_percentage(human_human, human_model):
    adjusted_human_model = np.maximum(human_model, 0)
    return (adjusted_human_model / human_human) * 100

# Calculating percentages for each category
percentage_overall = calculate_percentage(overall_human_human, overall_human_model)
percentage_positive = calculate_percentage(positive_human_human, positive_human_model)
percentage_top_quartile_human = calculate_percentage(top_quartile_human_predictability_human_human, top_quartile_human_predictability_human_model)
percentage_top_quartile_both = calculate_percentage(top_quartile_predictability_human_human, top_quartile_predictability_human_model)

# Separate data for Positive Correlation plot
data_positive = {
    "Layer": np.arange(1, 13),
    "Percentage": percentage_positive,
    "Category": ["Video Clips with Positive Human-Model Correlation"] * 12,
    "N": positive_human_model_N
}

# Data for the rest of the categories
data_rest = {
    "Layer": np.tile(np.arange(1, 13), 3),
    "Percentage": np.concatenate([percentage_overall, percentage_top_quartile_human, percentage_top_quartile_both]),
    "Category": ["All Video Clips"] * 12 + 
                ["Video Clips Within Top Quartile of Human Predictability"] * 12 + 
                ["Video Clips with Top Quartile Human and Model Predictability"] * 12
}

# Creating DataFrames for plotting
df_positive = pd.DataFrame(data_positive)
df_rest = pd.DataFrame(data_rest)

# Adjusting the plot settings
sns.set(style="white", rc={"figure.dpi": 600, "savefig.dpi": 600})

# Plot for "Video Clips with Positive Human-Model Correlation"
plt.figure(figsize=(15, 8))
plot_positive = sns.lineplot(data=df_positive, x="Layer", y="Percentage", hue="Category", style="Category", 
                             markers=True, ci="sd", palette="deep", linewidth=2.5, markersize=10)
plot_positive.set_xticks(np.arange(1, 13))
plot_positive.set_yticks(np.linspace(0, 120, 13))
plot_positive.set_yticklabels(['{}%'.format(int(y)) for y in np.linspace(0, 120, 13)])
plt.show()

# Plot for the rest of the categories
plt.figure(figsize=(15, 8))
plot_rest = sns.lineplot(data=df_rest, x="Layer", y="Percentage", hue="Category", style="Category", 
                         markers=True, ci="sd", palette="deep", linewidth=2.5, markersize=10)
plot_rest.set_xticks(np.arange(1, 13))
plot_rest.set_yticks(np.linspace(0, 120, 13))
plot_rest.set_yticklabels(['{}%'.format(int(y)) for y in np.linspace(0, 120, 13)])
plt.show()







###### SECOND PLOT 

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Function to calculate percentage
def calculate_percentage(human_human, human_model):
    adjusted_human_model = np.maximum(human_model, 0)
    return (adjusted_human_model / human_human) * 100

# Data arrays for the rest of the categories
overall_human_human = np.array([0.33] * 12)
overall_human_model = np.array([0.06, 0.00, 0.14, -0.07, 0.06, 0.12, 0.07, 0.10, 0.14, 0.10, 0.09, 0.05])
top_quartile_human_predictability_human_human = np.array([0.33] * 12)
top_quartile_human_predictability_human_model = np.array([0.06, 0.02, 0.15, -0.06, 0.06, 0.13, 0.08, 0.11, 0.24, 0.22, 0.09, 0.05])
top_quartile_predictability_human_human = np.array([0.32] * 12)
top_quartile_predictability_human_model = np.array([0.07, 0.00, 0.16, -0.08, 0.05, 0.12, 0.07, 0.10, 0.22, 0.21, 0.09, 0.04])

# Calculating percentages for each category
percentage_overall = calculate_percentage(overall_human_human, overall_human_model)
percentage_top_quartile_human = calculate_percentage(top_quartile_human_predictability_human_human, top_quartile_human_predictability_human_model)
percentage_top_quartile_both = calculate_percentage(top_quartile_predictability_human_human, top_quartile_predictability_human_model)

# Data for the rest of the categories
data_rest = {
    "Layer": np.tile(np.arange(1, 13), 3),
    "Percentage": np.concatenate([percentage_overall, percentage_top_quartile_human, percentage_top_quartile_both]),
    "Category": ["All Video Clips"] * 12 + 
                ["Video Clips Within Top Quartile of Human Predictability"] * 12 + 
                ["Video Clips with Top Quartile Human and Model Predictability"] * 12
}

# Creating DataFrame for plotting
df_rest = pd.DataFrame(data_rest)

# Adjusting the plot settings
sns.set(style="white", rc={"figure.dpi": 600, "savefig.dpi": 600})

# Plot for the rest of the categories
plt.figure(figsize=(15, 8))
plot_rest = sns.lineplot(data=df_rest, x="Layer", y="Percentage", hue="Category", style="Category", 
                         markers=True, ci="sd", palette="deep", linewidth=2.5, markersize=10)
plot_rest.set_xticks(np.arange(1, 13))
plot_rest.set_yticks(np.linspace(0, 100, 11))
plot_rest.set_yticklabels(['{}%'.format(int(y)) for y in np.linspace(0, 100, 11)])

# Enhancing visibility
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Removing the labels for x and y axes
plot_rest.set_xlabel('', fontsize=16)
plot_rest.set_ylabel('', fontsize=16)

# Thick font for legend
legend = plot_rest.legend()
plt.setp(legend.get_texts(), fontweight='bold')

# Removing the grid and the box around the plot, keeping only x and y axes
plot_rest.grid(False)
plot_rest.spines['top'].set_visible(False)
plot_rest.spines['right'].set_visible(False)
plot_rest.spines['bottom'].set_visible(True)
plot_rest.spines['left'].set_visible(True)

plt.show()




################################# 
##SECOND PLOT FOR SUPP MATERIAL##
#################################

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Function to calculate percentage
def calculate_percentage(human_human, human_model):
    adjusted_human_model = np.maximum(human_model, 0)
    return (adjusted_human_model / human_human) * 100

# New data for Positive Model-Human Correlations and TOP25% Human Correlations
positive_human_human = np.array([0.32, 0.33, 0.34, 0.34, 0.33, 0.33, 0.32, 0.33, 0.32, 0.32, 0.31, 0.31])
positive_human_model = np.array([0.25, 0.30, 0.28, 0.25, 0.29, 0.31, 0.31, 0.33, 0.37, 0.36, 0.32, 0.34])
positive_human_model_N = np.array([145, 284, 348, 95, 205, 431, 366, 358, 504, 446, 425, 361])

top_quartile_human_human = np.array([0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45])
top_quartile_human_model = np.array([0.26, 0.31, 0.29, 0.32, 0.30, 0.34, 0.37, 0.36, 0.38, 0.40, 0.36, 0.33])

# Calculating percentages for each category
percentage_positive = calculate_percentage(positive_human_human, positive_human_model)
percentage_top_quartile = calculate_percentage(top_quartile_human_human, top_quartile_human_model)

# Combining both datasets into a single DataFrame for plotting
data_combined = {
    "Layer": np.tile(np.arange(1, 13), 2),
    "Percentage": np.concatenate([percentage_positive, percentage_top_quartile]),
    "Category": ["Positive Model-Human Correlations"] * 12 + 
                ["TOP25% Human Correlations"] * 12,
    "N": np.concatenate([positive_human_model_N, np.repeat(None, 12)])  # Including N values only for the first category
}

# Creating DataFrame for plotting
df_combined = pd.DataFrame(data_combined)

# Adjusting the plot settings
sns.set(style="white", rc={"figure.dpi": 600, "savefig.dpi": 600})

# Plot for the combined data
plt.figure(figsize=(15, 8))
plot_combined = sns.lineplot(data=df_combined, x="Layer", y="Percentage", hue="Category", style="Category", 
                             markers=True, ci=None, palette="deep", linewidth=2.5, markersize=10)
plot_combined.set_xticks(np.arange(1, 13))
plot_combined.set_yticks(np.linspace(0, 100, 11))
plot_combined.set_yticklabels(['{}%'.format(int(y)) for y in np.linspace(0, 100, 11)])

# Enhancing visibility
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Removing the labels for x and y axes
plot_combined.set_xlabel('', fontsize=16)
plot_combined.set_ylabel('', fontsize=16)

# Thick font for legend
legend = plot_combined.legend()
plt.setp(legend.get_texts(), fontweight='bold')

# Removing the grid and the box around the plot, keeping only x and y axes
plot_combined.grid(False)
plot_combined.spines['top'].set_visible(False)
plot_combined.spines['right'].set_visible(False)
plot_combined.spines['bottom'].set_visible(True)
plot_combined.spines['left'].set_visible(True)

plt.show()


############## 
##FINAL PLOT##
##############


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Function to calculate percentage
def calculate_percentage(human_human, human_model):
    adjusted_human_model = np.maximum(human_model, 0)
    return (adjusted_human_model / human_human) * 100

# Data arrays for the rest of the categories
overall_human_human = np.array([0.33] * 12)
overall_human_model = np.array([0.06, 0.00, 0.14, -0.07, 0.06, 0.12, 0.07, 0.10, 0.14, 0.10, 0.09, 0.05])
top_quartile_human_predictability_human_human = np.array([0.33] * 12)
top_quartile_human_predictability_human_model = np.array([0.06, 0.02, 0.15, -0.06, 0.06, 0.13, 0.08, 0.11, 0.24, 0.22, 0.09, 0.05])
top_quartile_predictability_human_human = np.array([0.32] * 12)
top_quartile_predictability_human_model = np.array([0.07, 0.00, 0.16, -0.08, 0.05, 0.12, 0.07, 0.10, 0.22, 0.21, 0.09, 0.04])
top_quartile_human_human = np.array([0.45] * 12)
top_quartile_human_model = np.array([0.26, 0.31, 0.29, 0.32, 0.30, 0.34, 0.37, 0.36, 0.38, 0.40, 0.36, 0.33])

# Calculating percentages for each category
percentage_overall = calculate_percentage(overall_human_human, overall_human_model)
percentage_top_quartile_human = calculate_percentage(top_quartile_human_predictability_human_human, top_quartile_human_predictability_human_model)
percentage_top_quartile_both = calculate_percentage(top_quartile_predictability_human_human, top_quartile_predictability_human_model)
percentage_top_quartile = calculate_percentage(top_quartile_human_human, top_quartile_human_model)

# Data for the rest of the categories
data_rest = {
    "Layer": np.tile(np.arange(1, 13), 4),
    "Percentage": np.concatenate([percentage_overall, percentage_top_quartile_human, percentage_top_quartile_both, percentage_top_quartile]),
    "Category": ["All Videos"] * 12 + 
                ["Videos Top 25% Human Predictability"] * 12 + 
                ["Videos Top 25% Model Predictability"] * 12 +
                ["Videos Top 25% Human-Human Eyetracking Correlation"] * 12
}

# Creating DataFrame for plotting
df_rest = pd.DataFrame(data_rest)

# Adjusting the plot settings
sns.set(style="white", rc={"figure.dpi": 600, "savefig.dpi": 600})

# Existing code for plotting
plt.figure(figsize=(15, 8))
plot_rest = sns.lineplot(data=df_rest, x="Layer", y="Percentage", hue="Category", style="Category", 
                         markers=True, ci="sd", palette="deep", linewidth=2.5, markersize=10)
plot_rest.set_xticks(np.arange(1, 13))
plot_rest.set_yticks(np.linspace(0, 100, 11))
plot_rest.set_yticklabels(['{}%'.format(int(y)) for y in np.linspace(0, 100, 11)])

# Enhancing visibility
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Removing the labels for x and y axes
plot_rest.set_xlabel('', fontsize=18)
plot_rest.set_ylabel('', fontsize=18)

# Adjust the legend position by specifying bbox_to_anchor
legend = plot_rest.legend(fontsize='x-large', bbox_to_anchor=(0.65, 1.15))
plt.setp(legend.get_texts(), fontweight='bold')

# Removing the grid and the box around the plot, keeping only x and y axes
plot_rest.grid(False)
plot_rest.spines['top'].set_visible(False)
plot_rest.spines['right'].set_visible(False)
plot_rest.spines['bottom'].set_visible(True)
plot_rest.spines['left'].set_visible(True)

plt.show()
