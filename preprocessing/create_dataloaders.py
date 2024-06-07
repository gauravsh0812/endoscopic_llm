import os
import yaml
import pandas as pd
import torch
import multiprocessing
from PIL import Image
from torchvision import transforms
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
with open("config/config_sgpt.yaml") as f:
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
        return img,qtn,ans
        
class My_pad_collate(object):
    def __init__(self, device, max_len, tokenizer, ans_vocab):
        self.device = device
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.ans_vocab = ans_vocab

    def __call__(self, batch):
        _img, _qtns, _ans = zip(*batch)

        print(_img)
        print(_qtns)
        print(_ans)

        padded_tokenized_qtns = self.tokenizer(
                                _qtns, 
                                return_tensors="pt",
                                padding='max_length',
                                truncation=True,
                                max_length=self.max_len)
        
        # qtn tensors
        _qtn_input_ids = torch.Tensor(padded_tokenized_qtns["input_ids"])
        _qtn_attn_masks = torch.Tensor(padded_tokenized_qtns["attention_mask"])

        # ans tensors
        ans = [] 
        for _a in _ans:
            ans.append(int(self.ans_vocab[_a]))

        return (
            _img,
            _qtn_input_ids.to(self.device),
            _qtn_attn_masks.to(self.device),
            ans,
        )

def img_tnsr(im):
    num = int(im.split(".")[0])
    if not os.path.exists(f"{cfg.dataset.path_to_data}/image_tensors/{num}.pt"):
        IMAGE = Image.open(f"{cfg.dataset.path_to_data}/images/{im}")
        convert = transforms.ToTensor()
        IMAGE = convert(IMAGE).numpy()
        torch.save(IMAGE, f"{cfg.dataset.path_to_data}/image_tensors/{num}.pt")

def data_loaders(batch_size):

    print("creating dataloaders...")
    
    N = int(cfg.dataset.sample_size)

    questions = open(f"{cfg.dataset.path_to_data}/questions.lst").readlines()
    answers = open(f"{cfg.dataset.path_to_data}/answers.lst").readlines()
    questions_num = range(0, len(questions))[:N]

    assert len(questions) == len(answers)

    # split the image_num into train, test, validate
    train_val_images, test_images = train_test_split(
        questions_num, test_size=0.1, random_state=42
    )
    train_images, val_images = train_test_split(
        train_val_images, test_size=0.1, random_state=42
    )
    
    target_labels = []

    for t_idx, t_df in enumerate([train_images, test_images, val_images]):
        IMGS = list()
        ALL_QTNS = list()
        ALL_ANS = list()
        for i in t_df:
            n, txt = questions[i].split("\t")
            n = n.strip().replace("QTN","")
            IMGS.append(f"{cfg.dataset.path_to_data}/{n}.png")
            ALL_QTNS.append(txt.replace("\n","").strip())

            _lbl = answers[i].replace("\n","").strip()
            ALL_ANS.append(_lbl)
            
            # building vocab of ans
            if _lbl not in target_labels:
                target_labels.append(_lbl)

        # reshuffling the lists to avoid repeating images in a batch
        qi_data = {
            "IMG": IMGS,
            "QUESTION": ALL_QTNS,
            "ANSWER": ALL_ANS
        }

        # creating image_tensors;
        if cfg.dataset.create_image_tensors:
            os.makedirs(f"{cfg.dataset.path_to_data}/image_tensors", exist_ok=True)
            with multiprocessing.Pool(cfg.dataset.cpu_count) as pool:
                pool.map(img_tnsr, IMGS) 

        if t_idx == 0:
            train = pd.DataFrame(qi_data, columns=["IMG", "QUESTION", "ANSWER"])

            # shuffling multiple times
            train = train.sample(frac=1, random_state=42)
            train = train.sample(frac=1, random_state=42)
            train = train.sample(frac=1, random_state=42)

        elif t_idx == 1:
            test = pd.DataFrame(qi_data, columns=["IMG", "QUESTION", "ANSWER"])

            # shuffling multiple times
            test = test.sample(frac=1, random_state=42)
            test = test.sample(frac=1, random_state=42)
            test = test.sample(frac=1, random_state=42)

        else:
            val = pd.DataFrame(qi_data, columns=["IMG", "QUESTION", "ANSWER"])
            
            # shuffling multiple times
            val = val.sample(frac=1, random_state=42)
            val = val.sample(frac=1, random_state=42)
            val = val.sample(frac=1, random_state=42)

    print(f"saving dataset files to {cfg.dataset.path_to_data}/ folder...")
    train.to_csv(f"{cfg.dataset.path_to_data}/train.csv", index=False)
    test.to_csv(f"{cfg.dataset.path_to_data}/test.csv", index=False)
    val.to_csv(f"{cfg.dataset.path_to_data}/val.csv", index=False)

    print("training dataset size: ", len(train))
    print("testing dataset size: ", len(test))
    print("validation dataset size: ", len(val))

    # get max_len 
    max_len = get_max_len(train, test, val)    
    print("the max length: ", max_len)
    
    # build vocab 
    print("building questions vocab...")
    qtn_tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")
    qtn_vocab = qtn_tokenizer.get_vocab()
    with open(f"{cfg.dataset.path_to_data}/vocab.txt", 'w') as f:
        for word, idx in qtn_vocab.items():
            f.write(f"{word} {idx}\n")

    # building answers vocab
    ans_vocab = {}
    for i,l in enumerate(target_labels):
        ans_vocab[l] = i

    # writing answers vocab file...
    vfile = open("ans_vocab.txt", "w")
    for vidx, vstr in ans_vocab.items():
        vfile.write(f"{vidx} \t {vstr} \n") # build vocab

    # initializing pad collate class
    mypadcollate = My_pad_collate(cfg.general.device, max_len, qtn_tokenizer, ans_vocab)

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
            qtn_tokenizer,
            ans_vocab,
            max_len)