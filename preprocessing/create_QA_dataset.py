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
            temp["question_answers"] = {}
            qa = temp["question_answers"]

            qa["How many tools are being used?"] = len(v)
            qa["What tools are being used?"] = []
            qa["What is the current phase of the surgery?"] = []
            qa["What action is going on?"] = []
            qa["Which organ is under surgery?"] = []
            
            temp["captions"] = {}

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

                qa["What tools are being used?"].append(ins)
                qa["What is the current phase of the surgery?"].append(phs)
                qa["What action is going on?"].append(vrb)
                qa["Which organ is under surgery?"].append(tgt)
                
                temp["caption"]["caption_1"] = f"The current phase of the procedure is {phs}. \
                                        During this phase, {temp["number of tools"]} tools are being utilized: {ins}. \
                                        The primary focus of the procedure is on the {tgt}, \
                                        and the action being performed is to {vrb} the organ."
                

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