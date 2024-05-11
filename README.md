## Conda Env

```
conda create -n endoscopic_llm python=3.9 -y ; conda activate endoscopic_llm
```

## Prepare dataset

Re-arranging the dataset to a different format that we can use with our model.
Need to be run intialliy only. 
```
python preprocessing/rearrange_dataset.py
```

Preprocess the image and texts to save them as tensors and questions answers, respectively. 
```
python preprocessing/preprocess_dataset.py
```