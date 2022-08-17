from numpy import datetime_as_string
import spacy
from torchtext.data import TabularDataset, Field, BucketIterator
import dill 
import argparse
from torchtext.data.utils import get_tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', default='yelp5', help="directory of dataset")
    parser.add_argument('-data_ext', default='csv', help="extension of dataset")
    parser.add_argument('-data_pkl', default='data_yelp5_256.pickle', help="file name of preprocessed data(pickle file)")
    parser.add_argument('-data_task', default='CF', help="task of dataset" )
  
    opt = parser.parse_args()
    
    task = opt.data_task

    #############
    # tokenizer #
    #############
    spacy_en = spacy.load('en_core_web_sm')  # en tokenization
    spacy_de = spacy.load('de_core_news_sm')  # de tokenization
    
    def tokenize_en(text):
        return [token.text for token in spacy_en.tokenizer(text)]
    def tokenize_de(text):
        return [token.text for token in spacy_de.tokenizer(text)]

    if task == 'MT' :
        # (source:DE, target:EN) _ 데이터전처리(token, 소문자 등)
        SRC = Field(tokenize=tokenize_de, init_token="<sos>", eos_token="<eos>", pad_token="<blank>", lower=True, batch_first=True, fix_length=512) #!# batch_first=True => [배치크기, length]
        TRG = Field(tokenize=tokenize_en, init_token="<sos>", eos_token="<eos>", pad_token="<blank>", lower=True, batch_first=True, fix_length=512) #!#
        #SRC = Field(tokenize=tokenize_en, init_token="<sos>", eos_token="<eos>", pad_token="<blank>", lower=True, batch_first=True) #!# batch_first=True => [배치크기, length]
        #TRG = Field(tokenize=tokenize_de, init_token="<sos>", eos_token="<eos>", pad_token="<blank>", lower=True, batch_first=True) #!#
    elif task == 'CF':
        # TEXT(SRC)
        #SRC = Field(tokenize=tokenize_en, init_token="<sos>", eos_token="<eos>", pad_token="<blank>", lower=True, batch_first=True, fix_length=256)
        #SRC = Field(tokenize=tokenize_en, init_token="<sos>", eos_token="<eos>", pad_token="<blank>", lower=True, batch_first=True, fix_length=256, include_lengths=True)
        #for token_figure# 
        SRC = Field(tokenize=tokenize_en, init_token="<sos>", eos_token="<eos>", pad_token="<blank>", lower=True, batch_first=True, fix_length=256, include_lengths=True)
        # LABEL(TRG)
        #TRG = Field(tokenize=tokenize_en, init_token="<sos>", eos_token="<eos>", pad_token="<blank>", lower=True, batch_first=True, fix_length=1)
        TRG = Field(sequential=False, use_vocab=False, is_target=True, unk_token=None)

    print('[Info] Loading dataset ...')
    ################
    # make dataset #
    ################
    train_data = TabularDataset(path= './'+opt.data_dir+"/train."+opt.data_ext, format=opt.data_ext, fields=[('src', SRC), ('trg', TRG)], skip_header=True)
    valid_data = TabularDataset(path='./'+opt.data_dir+"/val."+opt.data_ext, format=opt.data_ext, fields=[('src', SRC), ('trg', TRG)], skip_header=True)
    test_data = TabularDataset(path='./'+opt.data_dir+"/test."+opt.data_ext, format=opt.data_ext, fields=[('src', SRC), ('trg', TRG)], skip_header=True)

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
    
    dill.dump(data, open(opt.data_pkl, 'wb'))
    print('[Info] Done..')
    
##################################################################################################################
    
    
if __name__ == '__main__':
    main()
