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
        l = len(_q.replace("\n","").strip().split())
        if l > c:
            c=l
    return c

class Img2MML_dataset(Dataset):
    def __init__(self, dataframe, ans_vocab):
        self.dataframe = dataframe
        self.ans_vocab = ans_vocab

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        qtn = self.dataframe.iloc[index, 1]
        img = self.dataframe.iloc[index, 0] 
        ans = self.dataframe.iloc[index,2]

        indexed_ans = []
        for token in ans.split():
            if self.ans_vocab.stoi[token] is not None:
                indexed_ans.append(self.ans_vocab.stoi[token])
            else:
                indexed_ans.append(self.ans_vocab.stoi["<unk>"])

        return img,qtn,torch.Tensor(indexed_ans)
        
class My_pad_collate(object):
    def __init__(self, device, max_len):
        self.device = device
        self.max_len = max_len
        self.tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")

    def __call__(self, batch):
        _img, _qtns, _ans = zip(*batch)

        padded_tokenized_qtns = self.tokenizer(
                                _qtns, 
                                return_tensors="pt",
                                padding='max_length',
                                truncation=True,
                                max_length=self.max_len)
        
        # tensors
        _qtn_input_ids = torch.Tensor(padded_tokenized_qtns["input_ids"])
        _qtn_attn_masks = torch.Tensor(padded_tokenized_qtns["attention_mask"])
        
        return (
            _img,
            _qtn_input_ids.to(self.device),
            _qtn_attn_masks.to(self.device),
            _ans,
        )
    
def data_loaders(batch_size):

    print("creating dataloaders...")
    
    questions = os.listdir(f"{cfg.dataset.path_to_data}/questions")
    answers = os.listdir(f"{cfg.dataset.path_to_data}/answers")
    questions_num = range(0, len(questions))

    # split the image_num into train, test, validate
    train_val_images, test_images = train_test_split(
        questions_num, test_size=0.1, random_state=42
    )
    train_images, val_images = train_test_split(
        train_val_images, test_size=0.1, random_state=42
    )

    for t_idx, t_qtns in enumerate([train_images, test_images, val_images]):
        QTNS = [questions[num] for num in t_qtns]
        IMGS = list()
        ALL_QTNS = list()
        ALL_ANS = list()
        for _Q in QTNS:
            _idx = int(_Q.split(".")[0].split("_")[1])
            _qtns = open(f"{cfg.dataset.path_to_data}/questions/question_{_idx}.lst").readlines()
            _ans = open(f"{cfg.dataset.path_to_data}/answers/answer_{_idx}.lst").readlines()
            for _q,_a in zip(_qtns,_ans):
                # keeping qtns thhat has one word answer only
                _alist = _a.split(",")
                if len(_alist) == 1:
                    print(_alist[0])
                    ALL_QTNS.append(_q)
                    ALL_ANS.append(f"<sos> {_alist[0]} <eos>")
                    IMGS.append(f"{cfg.dataset.path_to_data}/images/{_idx}.png")

        print("total number of samples: ", len(ALL_ANS))
        qi_data = {
            "IMG": IMGS,
            "QUESTION": ALL_QTNS,
            "ANSWER": ALL_ANS
        }
    
        if t_idx == 0:
            train = pd.DataFrame(qi_data, columns=["IMG", "QUESTION", "ANSWER"])
        elif t_idx == 1:
            test = pd.DataFrame(qi_data, columns=["IMG", "QUESTION", "ANSWER"])
        else:
            val = pd.DataFrame(qi_data, columns=["IMG", "QUESTION", "ANSWER"])
    
    print(f"saving dataset files to {cfg.dataset.path_to_data}/ folder...")
    train.to_csv(f"{cfg.dataset.path_to_data}/train.csv", index=False)
    test.to_csv(f"{cfg.dataset.path_to_data}/test.csv", index=False)
    val.to_csv(f"{cfg.dataset.path_to_data}/val.csv", index=False)

    # get max_len 
    max_len = get_max_len(train, test, val)    
    print("the max length: ", max_len)
    
    # build vocab 
    print("building questions vocab...")
    qtn_vocab = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base").get_vocab()
    with open(f"{cfg.dataset.path_to_data}/vocab.txt", 'w') as f:
        for word, idx in qtn_vocab.items():
            f.write(f"{word} {idx}\n")

     # build vocab
    print("building answers vocab...")

    counter = Counter()
    for line in train["ANSWER"]:
        counter.update(line.split())

    # <unk>, <pad> will be prepended in the vocab file
    ans_vocab = Vocab(
        counter,
        min_freq=cfg.dataset.vocab_freq,
        specials=["<pad>", "<unk>", "<sos>", "<eos>"],
    )

    # writing answers vocab file...
    vfile = open("ans_vocab.txt", "w")
    for vidx, vstr in ans_vocab.stoi.items():
        vfile.write(f"{vidx} \t {vstr} \n") # build vocab

    # initializing pad collate class
    mypadcollate = My_pad_collate(cfg.general.device, max_len)

    print("building dataloaders...")

    # initailizing class Img2MML_dataset: train dataloader
    imml_train = Img2MML_dataset(train, ans_vocab)
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
    imml_val = Img2MML_dataset(val, ans_vocab)

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
    imml_test = Img2MML_dataset(test, ans_vocab)
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
            qtn_vocab, 
            ans_vocab,
            max_len)