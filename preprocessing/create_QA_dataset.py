import multiprocessing as mp
import os, shutil, json
import pandas as pd
import tqdm

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import random


"""
Questions:

how many tools are operating?
what is the phase of image?
'is ' + tools[i] + ' used in ' + phases + '?' >> tools[i] + ' is used in ' + phases OR yes
'is ' + label + ' used in ' + phases + '?' >> label + ' is not used in ' + phases  OR no

We start with the simpler version with yes and no. Later we can introduce other option as well.
"""    

transform = Compose([
    Resize((224, 224)),  # Resize the image to 224x224 pixels
    ToTensor(),          # Convert the image to a PyTorch tensor
    Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
               std=[0.26862954, 0.26130258, 0.2757772])  # Normalize the tensor
])

# Load pre-trained model and tokenizer
model_name = "gpt2"  # You can use other models like "EleutherAI/gpt-neo-125M"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)


final_data = {}

count = 0

def main(ann, categories, fname):
    
    for i,v in ann.items():
        if len(v) == 1:
            _v = v[0]
            instruments = categories["instrument"]
            verbs = categories["verb"]
            targets = categories["target"]
            phases = categories["phase"] 
            triplets = categories["triplet"]

            temp = {}
            temp["image"] = f"/data/shared/CholecT50/CholecT50/videos/{fname}/{int(i):06d}.png"
            temp["image_tensors"] = f"/data/shared/CholecT50/CholecT50/image_tensors/{fname}/{int(i):06d}.pt"
            temp["number_of_tools"] = len(v)

            temp["tools"] = []
            temp["phase"] = []
            temp["action verb"] = []
            temp["target organ"] = []

            # for _v in v:
            if _v[2] == 1 and _v[9] == 1: # groundtruth only
                triplet_id = str(_v[0])
                phase_id = str(_v[-1])
                ins_id = str(_v[1])
                vrb_id = str(_v[7])
                tgt_id = str(_v[8])
                ins,vrb,tgt = triplets[triplet_id].split(",")
                phs = phases[phase_id]

                t_ins_id = next((key for key, value in instruments.items() if value == ins), None)
                t_vrb_id = next((key for key, value in verbs.items() if value == vrb), None)
                t_tgt_id = next((key for key, value in targets.items() if value == tgt), None)
                assert ins_id == t_ins_id
                assert vrb_id == t_vrb_id
                assert tgt_id == t_tgt_id

                temp["tools"].append(ins)
                temp["phase"].append(phs)
                temp["action_verb"].append(vrb)
                temp["target_organ"].append(tgt)

                # temp["caption_1:"] = f"The current phase of the procedure is {phs}. \
                #                         During this phase, {temp["number of tools"]} tools are being utilized: {ins}. \
                #                         The primary focus of the procedure is on the {tgt}, and the action being performed is to {vrb} the organ."
                

                # Convert structured input to a string

                input_text = (
                    f"number of tools: {temp['number_of_tools']}, "
                    f"tools: {temp['tools']}, "
                    f"phase: {temp['phase']}, "
                    f"target organ: {temp['target_organ']}, "
                    f"action verb: {temp['action_verb']}"
                )

                # Generate captions
                captions = generate_captions(input_text, num_captions=5)
                
                # Print the generated captions
                for i, caption in enumerate(captions, 1):
                    print(f"Caption {i}: {caption}")
                    temp[f"caption_{i}"] = caption

# Generate multiple captions
def generate_captions(input_text, num_captions=5, max_length=100):
    captions = []
    for _ in range(num_captions):
        input_ids = tokenizer.encode(f"Input: {input_text}\nOutput:", return_tensors="pt")
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,  # To avoid repetition
            temperature=0.7,  # Controls the creativity of the output
            top_p=0.9,  # Nucleus sampling
            do_sample=True
        )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        # Extract the generated caption
        caption = generated_text.split("Output:")[1].strip()
        captions.append(caption)
    return captions


def create_tensors(ann, fname):
    for i in ann:
        img = f"/data/shared/CholecT50/CholecT50/videos/{fname}/{int(i):06d}.png"
        image = Image.open(img)
        transformed_image = transform(image)
        torch.save(transformed_image,
            f"/data/shared/CholecT50/CholecT50/image_tensors/{fname}/{int(i):06d}.pt")

if __name__ == "__main__":
    files = os.listdir("/data/shared/CholecT50/CholecT50/labels")
    
    create_image_tensor = False

    for _file in tqdm.tqdm(range(1,3)):#len(files))):
        _file_name = f"VID{int(_file):02d}.json"
        _file_name = f"/data/shared/CholecT50/CholecT50/labels/{_file_name}"

        if os.path.exists(_file_name):
            data = json.load(open(_file_name))
            
            # categories
            categories = data["categories"]

            # annotations
            ann = data["annotations"]
            
            if create_image_tensor:
                create_tensors(ann, f"VID{int(_file):02d}")

            main(ann, categories, f"VID{int(_file):02d}")