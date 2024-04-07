import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import randint
import os
from scipy.ndimage.filters import gaussian_filter
import imageio
import io

final = pd.read_csv("final.csv")

final['x_pred'] = final['x_pred'] * window_width
final['y_pred'] = final['y_pred'] * window_height

# Assuming the DataFrame is called 'final'
grouped = final.groupby(['ID', 'stimulus'])

# Set the dimensions of the heatmap (e.g., the resolution of the video)
heatmap_width = 1280
heatmap_height = 720


# Loop through the groups
for (participant, stimulus), group in grouped:
    # Load the stimulus video
    video_path = f'/Users/Viktor/Desktop/CLIP_Behavioural_Data/stimulus/{stimulus}'

    if not os.path.isfile(video_path):
        print(f"Video file not found: {video_path}")
        continue

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    if frame_count == 0:
        print(f"Frame count is 0 for video: {video_path}")
        cap.release()
        continue

    # Initialize the averaged heatmap
    averaged_heatmap = np.zeros((heatmap_height, heatmap_width))

    frames_with_heatmaps = []

# Loop through the groups
for (participant, stimulus), group in grouped:
    # Load the stimulus video
    video_path = f'stimulus/{stimulus}'

    if not os.path.isfile(video_path):
        print(f"Video file not found: {video_path}")
        continue

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    if frame_count == 0:
        print(f"Frame count is 0 for video: {video_path}")
        cap.release()
        continue

    # Initialize the averaged heatmap
    averaged_heatmap = np.zeros((heatmap_height, heatmap_width))

    frames_with_heatmaps = []

    # Iterate through the rows in the group
    for _, row in group.iterrows():
        time_elapsed = row['time_elapsed']
        x_pred = row['x_pred']
        y_pred = row['y_pred']

        # Create a 2D histogram for x_pred and y_pred, and normalize it
        heatmap, _, _ = np.histogram2d([y_pred], [x_pred],
                                       bins=[heatmap_height, heatmap_width],
                                       range=[[0, heatmap_height], [0, heatmap_width]])
        
        
        

        # Apply a more robust normalization
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))

        # Apply Gaussian blur to the heatmap 
        sigma = 25  # Increase this value for stronger blurring
        smoothed_heatmap = gaussian_filter(heatmap, sigma)
        averaged_heatmap += smoothed_heatmap


    # Calculate the averaged and smoothed heatmap across frames
    averaged_heatmap = (averaged_heatmap - np.min(averaged_heatmap)) / (np.max(averaged_heatmap) - np.min(averaged_heatmap))
    averaged_smoothed_heatmap = gaussian_filter(averaged_heatmap, sigma=20)
 
    # Save the averaged and smoothed heatmap as a csv file 
    np.save(f'Numpy_Attention_Matrices_normalised/{participant}_{stimulus}_attention_matrix', averaged_heatmap)
    
    
    cap.release()


#########################################################
##Create 15 averaged heatmas per movie (as with Model)##
#########################################################

import cv2
import numpy as np
import os
from scipy.ndimage import gaussian_filter

# Assume 'grouped' is already defined as per the current script
# Define the number of desired time segments
num_segments = 15
segment_duration = 6 / num_segments  # Duration of each segment in seconds

final = pd.read_csv("final.csv")

# Loop through the groups
for (participant, stimulus), group in grouped:
    # Load the stimulus video
    video_path = f'stimulus/{stimulus}'

    if not os.path.isfile(video_path):
        print(f"Video file not found: {video_path}")
        continue

    cap = cv2.VideoCapture(video_path)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    if frame_count == 0:
        print(f"Frame count is 0 for video: {video_path}")
        cap.release()
        continue
    

    # Initialize the list to hold averaged heatmaps for each time segment
    averaged_heatmaps = [np.zeros((heatmap_height, heatmap_width)) for _ in range(num_segments)]
    segment_counts = [0] * num_segments  # Keep track of the number of heatmaps in each segment

    # Iterate through the rows in the group
    for _, row in group.iterrows():
        time_elapsed = row['time_elapsed']
        # Find the segment index, ensuring it is within the range of [0, num_segments-1]
        segment_index = min(int(time_elapsed // (segment_duration * 1000)), num_segments - 1)

        x_pred = row['x_pred']
        y_pred = row['y_pred']

        # Create a 2D histogram for x_pred and y_pred, and normalize it
        heatmap, _, _ = np.histogram2d([y_pred], [x_pred],
                                       bins=[heatmap_height, heatmap_width],
                                       range=[[0, heatmap_height], [0, heatmap_width]])

        # Normalize the heatmap
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))

        # Add the smoothed heatmap to the correct segment and increment the count
        averaged_heatmaps[segment_index] += heatmap
        segment_counts[segment_index] += 1

    # Now average out the heatmaps for each segment
    for i in range(num_segments):
        if segment_counts[i] > 0:
            averaged_heatmaps[i] /= segment_counts[i]

            # Apply additional smoothing after averaging if needed
            smoothed_averaged_heatmap = gaussian_filter(averaged_heatmaps[i], sigma=20)

            # Save the averaged heatmap for this segment
            output_path = f'Numpy_Attention_Matrices_normalised/{participant}_{stimulus}_attention_matrix_segment_{i+1}'
            np.save(output_path, smoothed_averaged_heatmap)

    cap.release()

