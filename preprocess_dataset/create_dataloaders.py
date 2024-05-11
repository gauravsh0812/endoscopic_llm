import os
import yaml
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from collections import Counter
from torchtext.vocab import Vocab
from torch.utils.data import SequentialSampler
from box import Box
from transformers import RobertaTokenizer
import torch.multiprocessing as mp

mp.set_start_method('spawn', force=True)

# reading config file
with open("config/config.yaml") as f:
    cfg = Box(yaml.safe_load(f))

def get_max_len(train, test, val):
    qtns = train["QUESTION"].to_list() + \
           test["QUESTION"].to_list() + \
           val["QUESTION"].to_list()
    
    c = 0
    for _q in qtns:
        l = len(_q.split())
        if l > c:
            c=l
    return c

class Img2MML_dataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        qtn = self.dataframe.iloc[index, 1]
        img = self.dataframe.iloc[index, 0] 
        lbl = self.dataframe.iloc[index,2]
        tmp = self.dataframe.iloc[index,-1]
        return img,qtn,lbl,tmp
        

class My_pad_collate(object):
    def __init__(self, device, max_len):
        self.device = device
        self.max_len = max_len
        self.tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")

    def __call__(self, batch):
        _img, _qtns, _lbls, _tmps = zip(*batch)

        padded_tokenized_qtns = self.tokenizer(
                                _qtns, 
                                return_tensors="pt",
                                padding='max_length',
                                truncation=True,
                                max_length=self.max_len)

        # the labels will be stored as tensor
        # 3 will be stored as [0.,0.,0.,1.]
        lbls = []
        for _l in _lbls:
            _l = int(_l.replace("\n",""))
            z = torch.zeros(11)
            z[_l] = 1.0
            lbls.append(z)
        
        # tensors
        _img = torch.Tensor(_img)
        _lbls = torch.stack(lbls)
        input_ids = torch.Tensor(padded_tokenized_qtns["input_ids"])
        attn_masks = torch.Tensor(padded_tokenized_qtns["attention_mask"])

        return (
            _img.to(self.device),
            input_ids.to(self.device),
            attn_masks.to(self.device),
            _lbls.to(self.device),
            _tmps,
        )
    
def data_loaders(batch_size):

    print("creating dataloaders...")
    q = open(f"{cfg.dataset.path_to_data}/questions.lst").readlines()
    l = open(f"{cfg.dataset.path_to_data}/labels.lst").readlines()
    t = open(f"{cfg.dataset.path_to_data}/templates.lst").readlines()

    assert len(q) == len(l) == len(t)

    image_num = range(0, len(q))

    # split the image_num into train, test, validate
    train_val_images, test_images = train_test_split(
        image_num, test_size=0.1, random_state=42
    )
    train_images, val_images = train_test_split(
        train_val_images, test_size=0.1, random_state=42
    )

    for t_idx, t_images in enumerate([train_images, test_images, val_images]):
        qi_data = {
            "IMG": [num for num in t_images],
            "QUESTION": [
                ("<sos> " + q[num].strip() + " <eos>") for num in t_images
            ],
            "LABEL": [l[num].strip().replace("\n","") for num in t_images],
            "TEMPLATE": [t[num].strip().replace("\n","") for num in t_images],
        }
    
        if t_idx == 0:
            train = pd.DataFrame(qi_data, columns=["IMG", "QUESTION", "LABEL", "TEMPLATE"])
        elif t_idx == 1:
            test = pd.DataFrame(qi_data, columns=["IMG", "QUESTION", "LABEL", "TEMPLATE"])
        else:
            val = pd.DataFrame(qi_data, columns=["IMG", "QUESTION", "LABEL", "TEMPLATE"])
    
    
    print(f"saving dataset files to {cfg.dataset.path_to_data}/ folder...")
    train.to_csv(f"{cfg.dataset.path_to_data}/train.csv", index=False)
    test.to_csv(f"{cfg.dataset.path_to_data}/test.csv", index=False)
    val.to_csv(f"{cfg.dataset.path_to_data}/val.csv", index=False)

    # get max_len 
    max_len = get_max_len(train, test, val)    
    print("the max length: ", max_len)
    
    # build vocab 
    print("building vocab...")
    vocab = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base").get_vocab()
    with open(f"{cfg.dataset.path_to_data}/vocab.txt", 'w') as f:
        for word, idx in vocab.items():
            f.write(f"{word} {idx}\n")

    # initializing pad collate class
    mypadcollate = My_pad_collate(cfg.general.device, max_len)

    print("building dataloaders...")

    # initailizing class Img2MML_dataset: train dataloader
    imml_train = Img2MML_dataset(train)
    # creating dataloader
    if cfg.general.ddp:
        train_sampler = DistributedSampler(
            dataset=imml_train,
            num_replicas=cfg.general.world_size,
            rank=cfg.general.rank,
            shuffle=cfg.dataset.shuffle,
        )
        sampler = train_sampler
        shuffle = False

    else:
        sampler = None
        shuffle = cfg.dataset.shuffle
        
    train_dataloader = DataLoader(
        imml_train,
        batch_size=batch_size,
        num_workers=cfg.dataset.num_workers,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=mypadcollate,
        pin_memory=cfg.dataset.pin_memory,
    )

    # initailizing class Img2MML_dataset: val dataloader
    imml_val = Img2MML_dataset(val)

    if cfg.general.ddp:
        val_sampler = SequentialSampler(imml_val)
        sampler = val_sampler
        shuffle = False
    else:
        sampler = None
        shuffle = cfg.dataset.shuffle

    val_dataloader = DataLoader(
        imml_val,
        batch_size=batch_size,
        num_workers=cfg.dataset.num_workers,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=mypadcollate,
        pin_memory=cfg.dataset.pin_memory,
    )

    # initailizing class Img2MML_dataset: test dataloader
    imml_test = Img2MML_dataset(test)
    if cfg.general.ddp:
        test_sampler = SequentialSampler(imml_test)
        sampler = test_sampler
        shuffle = False
    else:
        sampler = None
        shuffle = cfg.dataset.shuffle

    test_dataloader = DataLoader(
        imml_test,
        batch_size=batch_size,
        num_workers=cfg.dataset.num_workers,
        shuffle=shuffle,
        sampler=None,
        collate_fn=mypadcollate,
        pin_memory=cfg.dataset.pin_memory,
    )

    return (train_dataloader, 
            test_dataloader, 
            val_dataloader, 
            vocab, 
            max_len)