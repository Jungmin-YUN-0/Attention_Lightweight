from numpy import datetime_as_string
import pandas as pd
import spacy
from torchtext.data import TabularDataset, Field, BucketIterator
import dill 

def preprocess():
    #############
    # tokenizer #
    #############
    spacy_en = spacy.load('en_core_web_sm')  # en tokenization
    spacy_de = spacy.load('de_core_news_sm')  # de tokenization

    def tokenize_en(text):
        return [token.text for token in spacy_en.tokenizer(text)]
    def tokenize_de(text):
        return [token.text for token in spacy_de.tokenizer(text)]

    # (source:영어, target:독일어) _ 데이터전처리(token, 소문자 등)
    SRC = Field(tokenize=tokenize_en, init_token="<sos>", eos_token="<eos>", pad_token="<blank>", lower=True) #!# batch_first=True => [배치크기, length]
    TRG = Field(tokenize=tokenize_de, init_token="<sos>", eos_token="<eos>", pad_token="<blank>", lower=True) #!#

    ################
    # make dataset #
    ################
    train_data = TabularDataset(path='./wmt16/train.csv', format='csv', fields=[('src', SRC), ('trg', TRG)])
    valid_data = TabularDataset(path='./wmt16/val.csv', format='csv', fields=[('src', SRC), ('trg', TRG)])
    test_data = TabularDataset(path='./wmt16/test.csv', format='csv', fields=[('src', SRC), ('trg', TRG)])

    ###############
    # build vocab #    
    ###############
    SRC.build_vocab(train_data.src, min_freq=2) #!#
    TRG.build_vocab(train_data.trg, min_freq=2) #!#
    # [.vocab.stoi] token index ex(없는 단어: 0, padding:1, <sos>:2, <eos>:3)
    
    ##################
    # save to pickle #
    ##################
    data = {
        'vocab' : {'src':SRC, 'trg':TRG},
        'train_data' : train_data.examples,
        'valid_data' : valid_data.examples,
        'test_data' : test_data.examples    
            }
    print('[Info] Dumping the processed data to pickle file')
    
    dill.dump(data, open("data.pickle", 'wb'))
    print('[Info] Done..')
    
##################################################################################################################
    
    
if __name__ == '__main__':
    preprocess()
