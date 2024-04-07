
import numpy as np
import torch
# from pkg_resources import packaging

print("Torch version:", torch.__version__)

from clip import clip

for i,m in enumerate(clip.available_models()):
    print(i,m)

model_name = clip.available_models()[0]
model, preprocess = clip.load(model_name) # it uses ~/.cache/clip by default
model.eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import clip
import numpy as np
import os
import cv2
from torchvision.transforms.functional import to_pil_image

def attach_hooks(model, hook_function):
    for block in model.visual.transformer.resblocks:
        block.attn.register_forward_hook(hook_function)

def get_attention_maps(image_input, model):
    model.eval()
    model.zero_grad()

    # Dictionary to store attention maps from each layer
    layer_attention_maps = {}

    def attn_hook(layer_num):
        def hook(module, input, output):
#            print(f"Hook in layer {layer_num} triggered")
            attention_weights = output[0].detach().cpu()  # output[1] contains the attention weights
            if layer_num not in layer_attention_maps:
                layer_attention_maps[layer_num] = []
            layer_attention_maps[layer_num].append(attention_weights)
        return hook

    # Register the hook to the MultiheadAttention module of each layer
    for i, layer in enumerate(model.visual.transformer.resblocks):
        # The first block in each layer is MultiheadAttention
        multihead_attn = next(layer.children())
        if isinstance(multihead_attn, torch.nn.MultiheadAttention):
            multihead_attn.register_forward_hook(attn_hook(i))

    with torch.no_grad():
        _ = model.encode_image(image_input)

    return layer_attention_maps

def forward_func(image_input, text_input, model):
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_input)
    # Calculate similarity as dot product
    similarity = (image_features @ text_features.T).softmax(dim=-1)
    return similarity

os.chdir("the_usual_suspects")
# Read prestige words
with open("the_usual_suspects_words.txt", "r") as file:
    lines = file.readlines()

# Change directory
all_labels = set()

# Manage directory
directory = os.listdir()
excluded_files = ["the_usual_suspects_words.txt", ".ipynb_checkpoints", "heatmaps", "all_correlations.txt", ".DS_Store", "direct_heatmaps", "gif"]
directory = [item for item in directory if item not in excluded_files]
directory.sort()
device = "cpu"

all_corrs = []

# Process each file
for c in directory:
    label = c.split("_")[0]
    all_labels.add(label)

# Ensure the heatmaps directory exists
heatmaps_dir = "heatmaps"
os.makedirs(heatmaps_dir, exist_ok=True)

for c in directory:
    label = c.split("_")[0]
    prompt = c.split("_")[1]
    image = Image.open(c)
    preprocess = Compose([
        Resize(256, interpolation=3),  # Use 3 instead of InterpolationMode.BICUBIC
        CenterCrop(224),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    image_input = preprocess(image).unsqueeze(0).to(device)
    image_input.requires_grad = True

    prompts_and_labels = [f"{prompt} {label}" for label in all_labels]
    input_tokens = clip.tokenize(prompts_and_labels).to(device)

    # Assuming forward_func is correctly defined
    similarity = forward_func(image_input, input_tokens, model)
    label_index = list(all_labels).index(label)
    likelihood = similarity[0, label_index].item()

    print(f"{label}_{likelihood}")
    all_corrs.append([f"{label}_{likelihood}"])

    layer_attention_maps = get_attention_maps(image_input, model)

    # Process and save attention maps for each layer
    for layer_num, attention_maps in layer_attention_maps.items():
      # Average across all heads in the layer
      stacked_attention_maps = torch.stack(attention_maps)
 #     print(f"Layer {layer_num} stacked attention map shape:", stacked_attention_maps.shape)

      averaged_attention_map = torch.mean(torch.stack(attention_maps), dim=2)
#      print("Averaged attention map shape:", averaged_attention_map.shape)

      # Averaged_attention_map is of shape [1, 1, 50, 50]

      # First, remove the batch and head dimensions, keeping only the spatial dimensions
      attention_map_2d = averaged_attention_map.squeeze()

      # Resize the attention map to match the internal grid size
      # For example, if the internal grid size is 7x7 for a 224x224 input, and the original image is 1280x720
      grid_size = 7  # Adjust based on model's internal representation
      resized_grid_map = cv2.resize(attention_map_2d.numpy(), (grid_size, grid_size))

      # Now, resize this grid map to match the original image dimensions
      resized_attention_map = cv2.resize(resized_grid_map, (1280, 720))

      # Save the attention map as a numpy array
      np.save(f"heatmaps/{c}_layer_{layer_num}_attention_map.npy", resized_attention_map)

# Loop through files in the current directory
os.chdir("the_usual_suspects")
os.chdir("heatmaps")

for filename in os.listdir('.'):
    if filename.endswith('.zip'):
        # Delete the file
        os.remove(filename)
        print(f"Deleted file: {filename}")
        break  # Remove this line if you want to delete all .zip files
