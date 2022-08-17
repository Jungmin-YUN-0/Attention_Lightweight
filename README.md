# Core-token-Attention

Core-token Attention: 토큰 프루닝 기반 셀프 어텐션 경량화 메커니즘

Core-token Attention: Token Pruning-based Lightweight Self-Attention Mechanism


![figure](https://user-images.githubusercontent.com/76892989/185070394-3a0543d0-ec7e-4513-8218-a7675de33a94.png)




**[설치필요]**

python -m spacy download en_core_web_sm

python -m spacy download de_core_news_sm




**[실행]**

**DATA PREPROCESSING**

python preprocess.py -data_task [MT / CF] -data_dir [wmt16 / imdb / yelp5 / sst2 / sst5] -data_ext csv -data_pkl [pickleName.pickle]

(MT: machine translation, CF: classification)

**MAIN**

python main.py -gpu 1 -option [BASE / LR / CT] -task [TRAIN / TEST] -data_pkl [pickleName.pickle] -model_save [modelName.pt] -pred_save [predictionName.txt] -data_task [MT / CF]

(BASE: vanilla transformer, LR: low-rank attention(linformer), CT: core-token attention(proposed model))

[bleu score] De -> En

vanilla transformer : 0.2998 linformer : 0.0408 ,,,
