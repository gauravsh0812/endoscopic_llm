import os, torch
from models.clip import ClipVisionEncoder
from models.roberta import RobertaEncoder

CLIPENC = ClipVisionEncoder()
# ROBENC = RobertaEncoder()   

image_path = "/data/gauravs/combine_data/images"
# qtn_path = "/data/gauravs/combine_data/questions"

for i in os.listdir(image_path):
    img = f"{image_path}/{i}"
    tnsr = CLIPENC(img, "cuda:0")
    
    torch.save(f"/data/gauravs/combine_data/clip_image_tensors/{int(i.split('.')[0])}.pt")
