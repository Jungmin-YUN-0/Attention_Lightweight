# TopAttn: Focusing on Core Tokens With Token Pruned Self-Attention


 "TopAttn: Focusing on Core Tokens With Token Pruned Self-Attention" (ICEIC 2023)

## Summary

Attention mechanism of transformer requires computational complexity that grows quadratically with input sequence length, which restricts its application to textremely long sequences. In this paper, we present TOken Pruned selfATTeNtion(TopAttn), which improves efficiency with low computation on attention operations and a loss of expressiveness. TopAttn introduces a token pruning method to the self-attention 
gradually through transformer layers. 


![image](https://user-images.githubusercontent.com/76892989/204658198-23128ffc-96ce-4c68-8b0a-59aa53be4b5d.png)



## Dependencies

This code is written in Python, Dependencies include 

* spacy
(python -m spacy download en_core_web_sm, python -m spacy download de_core_news_sm)

* torchtext==0.4 (or 0.6)


## Dataset

We conduct experiments on binary classification dataset(IMDB and SST-2) and multi-class classification dataset(SST-5 and YELP-5)



* IMDB (https://aclanthology.org/P11-1015.pdf)

* SST-2 (https://aclanthology.org/D13-1170.pdf)

* SST-5 (https://aclanthology.org/D13-1170.pdf)

* YELP-5 (https://proceedings.neurips.cc/paper/2015/file/250cf8b51c773f3f8dc8b4be867a9a02-Paper.pdf)

## Usage

### [DATA PREPROCESSING]

**preprocess.py** is for preprocessing step before training the model.

Example command lines:

```Python
python preprocess.py -data_task [MT / CF] -data_dir [wmt16 / imdb / yelp5 / sst2 / sst5] -data_ext csv -data_pkl [pickleName.pickle]
```

(NOTE: **MT** is for machine translation and **CF** is for classification task)

### [MAIN]

**main.py** is for model training and inference.

Example command lines:

```Python
python main.py -gpu 1 -option [BASE / LR / CT] -task [TRAIN / TEST] -data_pkl [pickleName.pickle] -model_save [modelName.pt] -pred_save [predictionName.txt] -data_task [MT / CF]
```

(BASE: vanilla transformer, LR: low-rank attention(linformer), CT: core-token attention(proposed model))

[bleu score] De -> En
