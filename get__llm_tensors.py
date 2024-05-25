import os, torch, tqdm
from models.clip import ClipVisionEncoder
from models.roberta import RobertaEncoder

CLIPENC = ClipVisionEncoder()
# ROBENC = RobertaEncoder()   

image_path = "/data/gauravs/combine_data/images"
# qtn_path = "/data/gauravs/combine_data/questions"

tset = tqdm.tqdm(os.listdir(image_path))
for i in tset:
    img = f"{image_path}/{i}"
    tnsr = CLIPENC(img, "cuda:0")
    
    torch.save(f"/data/gauravs/combine_data/clip_image_tensors/{int(i.split('.')[0])}.pt")
