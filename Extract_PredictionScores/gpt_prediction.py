
########################
# Unimodal Predictions #
########################


from typing import Text
import os
from openai import OpenAI
from dotenv import load_dotenv

# Assuming your OpenAI API key is stored in an .env file for security
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI()

device = "cuda"

# Paths and directories might need to be adapted based on where you're running this script
pic_dir = os.listdir("/content/inference/prestige")
print(pic_dir)

# Update these lines to match the directories or files you're actually working with
pic_dir.remove("ckpts")
pic_dir.remove(".DS_Store")

all_corrs = []
all_labels = []

os.chdir("/content/inference/prestige")

for pic in pic_dir:
  label = pic.split("_")[0]
  all_labels.append(label)

all_labels = list(set(all_labels))

for pic in pic_dir:
  print(pic)
  label = pic.split("_")[0]
  prompt = pic.split("_")[1]
  condition = pic.split("_")[2]

  print(label)
  print(prompt)
  print(condition)

  # Construct the prompt
  api_prompt = f"What is the relevance of the following linguistic information [{prompt}] for predicting the word [{label}]"

  # Query the OpenAI API
  response = client.chat.completions.create(
    model="gpt-4 ",  
    prompt=api_prompt,
    max_tokens=100,
    n=1,
    seed=1,
    temperature=0,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    messages=[
    {"role": "system", "content": api_prompt}]
  )

  output_text = response.choices[0].text.strip()
  print(f"{label}_{output_text}")
  all_corrs.append([f"{label}_{output_text}", condition])

# Save to file
with open('all_correlations_unimodal_gpt4.txt', 'w') as fp:
    for item in all_corrs:
        fp.write("%s\n" % item)
    print('Done')


##########################
# Multimodal Predictions #
##########################

import os
import base64
import requests
from dotenv import load_dotenv

# Load OpenAI API key from .env 
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Directory for GIFs
gif_dir = "/content/inference/prestige_gifs"
print(os.listdir(gif_dir))

# Set headers for the API request
headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}

# Navigate to the directory where the images are stored
os.chdir(gif_dir)

all_corrs = []

# prompt
prompt_text = "Pay careful attention to this gif and the associated dialogue. How relevant do you judge this visual-linguistic information for the word: “workhouse”? Give me a percentage between 0 and 100 please. Only respond with the percentage number."

for gif_filename in os.listdir(gif_dir):
    # Encode the current image to base64
    base64_image = encode_image(gif_filename)
    
    # Construct the payload for the API request
    payload = {
      "model": "gpt-4-vision-preview",
      "messages": [
        {
          "role": "user",
          "content": prompt_text
        },
        {
          "role": "system",
          "content": {
            "type": "image_url",
            "image_url": f"data:image/gif;base64,{base64_image}"
          }
        }
      ],
      "max_tokens": 300
    }

    # Make the API request
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    # Process the response
    output_text = response.json()['choices'][0]['message']['content']
    print(f"{gif_filename}_{output_text}")

    all_corrs.append([f"{gif_filename}_{output_text}"])

# Save the correlations to a file
with open('all_correlations_multimodal_gpt4.txt', 'w') as fp:
    for item in all_corrs:
        fp.write("%s\n" % item)
    print('Done')



