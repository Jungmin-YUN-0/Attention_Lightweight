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
**Arguments are as follows:**

* data_task: ```MT``` is for machine translation(De→En) and ```CF``` is for classification (default: CF)
* data_dir: directory of dataset
* data_ext: extension of dataset (default: csv)
* data_pkl: file name of preprocessed data(pickle file)


### [MAIN]

**main.py** is for model training and inference.

Example command lines:

```Python
python main.py -gpu 1 -option [BASE / LR / CT] -task [TRAIN / TEST] -data_task [MT / CF] -data_pkl [pickleName.pickle] -model_save [modelName.pt] -pred_save [predictionName.txt] 
```
**Arguments are as follows:**

* gpu: gpu number
* option: ```BASE``` is for vanilla transformer, ```LR``` is for low-rank attention(linformer) and ```CT``` is for TopAttn (our proposed method) (default: CT)
* task: ```TRAIN``` is for training, and ```TEST``` is for inference
* data_task: ```MT``` is for machine translation and ```CF``` is for classification (default: CF)
* data_pkl: file name of preprocessed data 
* model_save: name of best model
* pred_save: file name of prediction reesult (This is for machine translation task)

**Additional Arguments are as follows:**

* batch_size: batch size (default: 16)
* num_dpoch : # of epoch (default: 8)
* learning_rate: learning rate (default: 1e-4)
* num_warmup: # of steps for warmup (default: 4000)
* hidden_dim: hidden dimension (default: 512)
* n_layer: # of encoder and decoder layer (default: 6)
* n_head: # of head(for multi-head attention) (default: 8)
* ff_dim: dimension of feed-forward neural network (default: 2048)
* dropout: ratio of dropout (default: 0.1)

## Experiment
**1) Performance comparison of different token pruning ratios**
![image](https://user-images.githubusercontent.com/76892989/204666495-c428e6b8-9ac0-4719-aa6a-36b76d75a101.png)
**2) Training memory with different token pruning ratios**
![image](https://user-images.githubusercontent.com/76892989/204666620-6038b129-b49e-4398-8b75-9de8d6007062.png)
**3) Comparison with vanilla transformer on various datasets**
![image](https://user-images.githubusercontent.com/76892989/204667299-b44a97e5-9b8f-4031-a22e-7632523b641f.png)

## Contact
cocoro357@cau.ac.kr
