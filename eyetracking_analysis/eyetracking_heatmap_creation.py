### combine both eye-tracking folders. No need to run. 

import os
import shutil

source_folder = 'eyetracking2'
destination_folder = 'eyetracking'

# Get the list of files and directories in the source folder
items = os.listdir(source_folder)

# Loop through each item in the source folder
for item in items:
    # Create the full path to the item
    src_path = os.path.join(source_folder, item)
    dst_path = os.path.join(destination_folder, item)
    
    # If the item is a file, copy it to the destination folder
    if os.path.isfile(src_path):
        shutil.copy2(src_path, dst_path)
    # If the item is a directory, recursively merge its contents to the destination folder
    elif os.path.isdir(src_path):
        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)






import os 
import pandas as pd
import numpy as np
import cv2
import random
from matplotlib import pyplot as plt

study1 = "urvi"
study2 = "q7va"
study3 = "ov2z"

stimuli = os.listdir("stimulus")


eye_tracking = os.listdir("eyetracking2")
eye_tracking.remove("CLIP1.xlsx")
eye_tracking.remove("CLIP2.xlsx")
eye_tracking.remove("CLIP3.xlsx")
eye_tracking.remove(".DS_Store")

output = "human_heatmaps"
video_count = 5  # Number of video clips

os.chdir("eyetracking2")


final = pd.DataFrame()

window_width = 1280
window_height = 720

for file in eye_tracking:
    print(file)
    if "calibration" in file:
        pass
    else:
    
        if file.split('task-')[1][:4] == study1:
            excl = pd.read_excel("CLIP1.xlsx",engine='openpyxl')
            
        elif file.split('task-')[1][:4] == study2:
            excl = pd.read_excel("CLIP2.xlsx",engine='openpyxl')
            
        elif file.split('task-')[1][:4] == study3:
            excl = pd.read_excel("CLIP3.xlsx",engine='openpyxl')
            
        df = pd.read_excel(file, engine='openpyxl')
        spreadsheet_row = int(df["spreadsheet_row"][2])
        perp_id = df["participant_id"]
        y_pred = df["y_pred_normalised"]
        x_pred = df["x_pred_normalised"]
        stimulus = excl["Videos"][spreadsheet_row]
        time_elapsed = df["time_elapsed"]
        stimulus_series = pd.Series([stimulus]*len(perp_id))
        
        # Concatenate the 4 series into a DataFrame
        temp_df = pd.concat([perp_id, x_pred, y_pred, stimulus_series, time_elapsed], axis=1)

        # Give the columns proper names
        temp_df.columns = ['ID', 'x_pred', 'y_pred', 'stimulus', "time_elapsed"]

        # Append the temporary DataFrame to the main DataFrame
        final = final.append(temp_df, ignore_index=True)
        

final = final[final['x_pred'] != 0]

# Now save this DataFrame to a CSV file
final.to_csv('final.csv', index=False)  # index=False will not write row indices

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

