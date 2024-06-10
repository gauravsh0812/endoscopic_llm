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
## Training
Check and modify the model name as per needed.
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py
```

## Models

#### Clip Roberta Adaptor Clf (clip_roberta_adaptor_clf)
This model uses Clip and Roberta to encode the image and corresponding text questions. These encoded tensors are then concatenated and passed through a series of linear adaptor layers. these layers serves two important functionalities: Firstly, they are required to bring doem the higher dimensional tenors to lower one. Secondly, these are used to bring both image and text lower dimensional tensors to same domain were they will be passed through the final classification layers.

#### ResNet18 Clip - Roberta GPT2 - Adaptor Clf (clip_gpt_emb_clf)
