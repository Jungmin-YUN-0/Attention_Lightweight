# TopAttn: Focusing on Core Tokens With Token Pruned Self-Attention


"TopAttn: Focusing on Core Tokens With Token Pruned Self-Attention" (ICEIC 2023)


Attention mechanism of transformer requires computational complexity that grows quadratically with input sequence length, which restricts its application to textremely long sequences. In this paper, we present TOken Pruned selfATTeNtion(TopAttn), which improves efficiency with low computation on attention operations and a loss of expressiveness. TopAttn introduces a token pruning method to the self-attention 
gradually through transformer layers.


![image](https://user-images.githubusercontent.com/76892989/204658198-23128ffc-96ce-4c68-8b0a-59aa53be4b5d.png)




**[설치필요]**

python -m spacy download en_core_web_sm

python -m spacy download de_core_news_sm

(torchtext==0.4)
안되면 0.6


**[실행]**

- DATA PREPROCESSING

python preprocess.py -data_task [MT / CF] -data_dir [wmt16 / imdb / yelp5 / sst2 / sst5] -data_ext csv -data_pkl [pickleName.pickle]

(MT: machine translation, CF: classification)

- MAIN

python main.py -gpu 1 -option [BASE / LR / CT] -task [TRAIN / TEST] -data_pkl [pickleName.pickle] -model_save [modelName.pt] -pred_save [predictionName.txt] -data_task [MT / CF]

(BASE: vanilla transformer, LR: low-rank attention(linformer), CT: core-token attention(proposed model))

[bleu score] De -> En

vanilla transformer : 0.2998 linformer : 0.0408 ,,,
