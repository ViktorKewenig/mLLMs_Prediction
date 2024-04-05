# mLLMs Prediction

This repository contains code used in the paper "Multimodality and Attention Increase Alignment in Natural Language Prediction Between Humans and Computational Models" by Viktor Kewenig, Andrew Lampinen, Samuel A. Nastase, Christopher Edwards, Quitterie D'Elascombe, Akilles Richardt, Jeremy I. Skipper and Gabriella Vigliocco

## Usage

### Extracting Prediction Scores
1. **Download Llama**: download and extract [llama](https://huggingface.co/huggyllama/llama-7b) into the "Extract Prediction Scores" folder. Download the [llama-adapter](https://github.com/OpenGVLab/LLaMA-Adapter/tree/main/llama_adapter_v2_multimodal7b) into the adapter folder in the same directory.
2. **Set OpenAI API Key Variable**: having an openAI API key is necessary for accessing GPT-4 and GPT-4 multimodal. If you have one, set it as an environmental variable.
3. **Install Requirements**: pip install the requirements.txt file (add requirements for the CLIP model if not already done so). 
4. **Run**: run the code for each model individually

### Comparing Prediction Scores
1. **Example Data**: you can run "Analysis.py" to compare and plot extracted prediction scores from GPT-4 and human data (skipping the extraction step). Exemplary data is provided in the directory.
2. **Plot**: install dependencies (Matplotlib) and run the code to plot scatterplots. 

### Eyetracking Analysis
1. **Heatmap Creation**: this script takes the eye-tracking data provided by the [Gorilla online experiment builder](https://gorilla.sc) and transforms it into normalised and smoothed numpy heatmaps.
2. **Heatmap Comparison**: this script compares eye-tracking data with the attention maps extracted from the CLIP model in step (1) "Extracting Prediction Scores".
3. **Data Availability**: all anonymised eye-tracking data will be available under this link: https://osf.io/6whzq/?view_only=162085f95bab42b5a57b34b386143ba8 (upon publication).

### Follow-Up Eyetracking Analysis
1. **Analysis**: we ran a follow-up analysis with human participants to test whether presence of salient visual cues predict overlap scores between human- and model attention patterns. You can run "Analysis.py" with the provided sample data.
2. **Data Availability**: all anonymised data from this follow-up experiment will be available under this link: https://osf.io/6whzq/?view_only=162085f95bab42b5a57b34b386143ba8 (upon publication).
