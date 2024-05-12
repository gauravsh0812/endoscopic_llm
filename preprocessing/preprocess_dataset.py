import os
import shutil
import yaml
import torch
from box import Box
from PIL import Image
from torchvision import transforms

# reading config file
with open("config/config.yaml") as f:
    cfg = Box(yaml.safe_load(f))

# create an image_tensors folder
if not os.path.exists(f"{cfg.dataset.path_to_data}/images"):
    os.mkdir(f"{cfg.dataset.path_to_data}/images")
# if not os.path.exists(f"{cfg.dataset.path_to_data}/image_tensors"):
#     os.mkdir(f"{cfg.dataset.path_to_data}/image_tensors")
# if not os.path.exists(f"{cfg.dataset.path_to_data}/questions"):
#     os.mkdir(f"{cfg.dataset.path_to_data}/questions") 
# if not os.path.exists(f"{cfg.dataset.path_to_data}/answers"):
#     os.mkdir(f"{cfg.dataset.path_to_data}/answers")
# if not os.path.exists("logs"):
#     os.mkdir('logs')

def preprocess_data(vid,img,num,qa):
    
    # IMAGE = Image.open(f"{cfg.dataset.path_to_data}/{vid}/imgs/{img}")

    # saving the image
    shutil.copyfile(f"{cfg.dataset.path_to_data}/{vid}/imgs/{img}",
                    f"{cfg.dataset.path_to_data}/images/{img}")

    # converting and saving to tensor
    # convert = transforms.ToTensor()
    # IMAGE = convert(IMAGE)
    # torch.save(IMAGE, f"{cfg.dataset.path_to_data}/image_tensors/{num}.pt")

    # # wrting questions and answers
    # qa = open(qa).readlines()
    # q = open(f"{cfg.dataset.path_to_data}/questions/question_{num}.lst", "w") 
    # a = open(f"{cfg.dataset.path_to_data}/answers/answer_{num}.lst", "w")
    # for _qa in qa:
    #     _qas = _qa.split("|")
    #     _q,_a = _qas[0], _qas[1]
    #     q.write(_q + "\n")
    #     a.write(_a + "\n")

def preprocess():
    
    print("creating image tensors...")
    
    for v in os.listdir(cfg.dataset.path_to_data):
        if "VID" in v:
            print(v)
            imgs = os.listdir(f"{cfg.dataset.path_to_data}/{v}/imgs")
            qas = f"{cfg.dataset.path_to_data}/{v}/qas"
            for im in imgs:
                num = int(im.split(".")[0])
                qa = os.path.join(qas, f"{num}.txt")
                preprocess_data(v,im,num,qa)

if __name__ == "__main__":
    preprocess()