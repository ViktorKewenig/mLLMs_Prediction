# -*- coding: utf-8 -*-

##############################
# Unimodal Prediction Scores #
##############################

from typing import Text
import cv2
import llama
import torch
import numpy as np
import os
from PIL import Image
import torch.nn.functional as F
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer


device = "cuda"
tokenizer = LlamaTokenizer.from_pretrained("tokenizer.model")  #replace with huggingface repository if file is not stored locally
model = LlamaForCausalLM.from_pretrained("7B").to(device) #replace with huggingface repository if file is not stored locally



pic_dir = os.listdir("/content/inference/prestige")
print(pic_dir)

#pic_dir.remove(".ipynb_checkpoints")
pic_dir.remove("ckpts")
pic_dir.remove(".DS_Store")

model = model.float()
model.eval()

all_corrs = []
all_labels = []

os.chdir("/content/inference/prestige")

for pic in pic_dir:
  label = pic.split("_")[0]
  all_labels.append(label)

all_labels = list(set(all_labels))

# Tokenize the labels
all_label_tokens = [tokenizer.encode(label, bos=False, eos=False) for label in all_labels]

# Flatten the list of token lists to get all unique token IDs
all_label_ids_flat = [token_id for token_list in all_label_tokens for token_id in token_list]


os.chdir("/content/inference/prestige")
for pic in pic_dir:
  print(pic)
  label = pic.split("_")[0]
  prompt = pic.split("_")[1]
  condition = pic.split("_")[2]

  print(label)
  print(prompt)
  print(condition)

  tokens_for_word = tokenizer.encode(label)
  tokens_for_word = torch.tensor(tokens_for_word).unsqueeze(0).to(device)
  token_indices = tokens_for_word[0].tolist()

  tokens = tokenizer.encode(prompt)
  tokens = torch.tensor(tokens).unsqueeze(0).to(device)

  with torch.no_grad():
      logits = model.forward(tokens)  # No backward pass needed
      probabilities = F.softmax(logits.logits, dim=-1)
      probabilities = probabilities [0,-1]
      probabilities = probabilities[all_label_ids_flat]


  joint_probability = 1.0
  for token_list in all_label_tokens:
      if all(token in token_list for token in tokens_for_word[0]):
          for token_id in tokens_for_word[0]:
              position_in_all_label_ids = all_label_ids_flat.index(token_id)
              joint_probability *= probabilities[position_in_all_label_ids].item()
          break

  print(f"{label}_{joint_probability}")
  all_corrs.append([f"{label}_{joint_probability}", condition])

  # Release GPU memory
  del logits, probabilities, tokens_for_word, tokens
  torch.cuda.empty_cache()

# Save to file
with open('all_correlations_unimodal_llama.txt', 'w') as fp:
    for item in all_corrs:
        fp.write("%s\n" % item)
    print('Done')



################################
# Multimodal Prediction Scores #
################################

from typing import Text
import cv2
import llama
import torch
import numpy as np
import os
from PIL import Image
import torch.nn.functional as F
from llama import Tokenizer

device = "cuda"

llama_dir = "/content/inference"
pic_dir = os.listdir("/content/inference/prestige")
print(pic_dir)

pic_dir.remove(".DS_Store")

model, preprocess = llama.load("BIAS-7B", llama_dir, device) #replace with huggingface repository if file is not stored locally
model = model.float()
model.eval()

tokenizer = Tokenizer("tokenizer.model") #replace with huggingface repository if file is not stored locally

all_corrs = []

os.chdir("/content/inference/prestige")
for pic in pic_dir:
    print(pic)
    label = pic.split("_")[0]
    prompt = pic.split("_")[1]
    condition = pic.split("_")[2]

    print(label)
    print(prompt)
    print(condition)

    img = Image.fromarray(cv2.imread(pic))
    img = preprocess(img).unsqueeze(0).to(device)

    tokens_for_word = tokenizer.encode(s=label, bos=False, eos=False)
    tokens_for_word = torch.tensor(tokens_for_word).unsqueeze(0).to(device)
    token_indices = tokens_for_word[0].tolist()

    tokens = tokenizer.encode(s=prompt, bos=True, eos=False)
    tokens = torch.tensor(tokens).unsqueeze(0).to(device)

    # Use forward_visual to get the visual query
    visual_query = model.forward_visual(img)

    with torch.no_grad():
        logits = model.forward_inference(visual_query, tokens, start_pos=len(tokens)-1)  # No backward pass needed
        probabilities = F.softmax(logits, dim=-1)
        print(probabilities.shape)

    joint_probability = 1.0
    for token_id in token_indices:
        joint_probability *= probabilities[0, token_id].item()

    print(f"{label}_{joint_probability}")
    all_corrs.append([f"{label}_{joint_probability}", condition])

    # Release GPU memory
    del logits, probabilities, tokens_for_word, tokens, img
    torch.cuda.empty_cache()

# Save to file
with open('all_correlations_multimodal_llama.txt', 'w') as fp:
    for item in all_corrs:
        fp.write("%s\n" % item)
    print('Done')


