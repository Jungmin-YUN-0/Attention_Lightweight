from tqdm import tqdm
import dill
import time
import math
import os
from transformers import get_scheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext.data import Dataset, BucketIterator
# import torchmetrics import BLEUScore
import hparams
from model.Transformer import Transformer
from Optim import ScheduledOptim

#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import wandb
wandb.init(project="transformer", entity="jungminy")

wandb.config = {
  "learning_rate": 0.001,
  "epochs": 100,
  "batch_size": 128
}
############################################################################################\
def patch_src(src, pad_idx):
    src = src.transpose(0, 1)
    return src

def patch_trg(trg, pad_idx):
    trg = trg.transpose(0, 1)
    trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
    return trg, gold

def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
    gold = gold.contiguous().view(-1)
    loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')

    pred = pred.max(1)[1]
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()
    return loss, n_correct, n_word
############################################################################################

#############################
# model training (per epoch)# 
#############################
def train(model, iterator, optimizer, device, SRC_PAD_IDX, TRG_PAD_IDX):
    model.train() # 학습 모드
    total_loss, n_word_total, n_word_correct = 0, 0, 0 
        
    for idx,batch in enumerate(tqdm(iterator)):  ##leave=False
        # prepare data
        src_seq = patch_src(batch.src, SRC_PAD_IDX).to(device)
        trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch.trg, TRG_PAD_IDX))
        
        # forward
        optimizer.zero_grad()
        pred = model(src_seq, trg_seq)
        
        # backward & update parameters
        loss, n_correct, n_word = cal_performance(pred, gold, TRG_PAD_IDX)
        loss.backward()
        
        optimizer.step()
        lr_scheduler.step()

        # note keeping
        n_word_total += n_word
        n_word_correct += n_correct
        total_loss += loss.item()

        #if idx % 50 == 0:
        #print(f"{idx} : loss = {total_loss/n_word_total} | acc = {n_word_correct/n_word_total}")
        #trg_tokens = [data['vocab']['trg'].vocab.itos[i] for i in [data['vocab']['trg'].vocab.stoi[data['vocab']['trg'].init_token]]]
        #print(trg_tokens[1:])
    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy
    
###############################
# model evaluation (per epoch)# 
###############################
def evaluate(model, iterator, device, SRC_PAD_IDX, TRG_PAD_IDX):
    model.eval() # 평가 모드
    total_loss, n_word_total, n_word_correct = 0, 0, 0
    
    with torch.no_grad():
        for batch in tqdm(iterator):  #leave=False
            # prepare data
            src_seq = patch_src(batch.src, SRC_PAD_IDX).to(device)
            trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch.trg, TRG_PAD_IDX))
            
            # forward
            pred = model(src_seq, trg_seq)
            loss, n_correct, n_word = cal_performance(pred, gold, TRG_PAD_IDX)

            # note keeping
            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += loss.item()
            
    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy
 
###############
# start train #
###############
def start_train(model, train_iterator, valid_iterator, optimizer, device, N_EPOCHS):
    def print_performances(header, accu, start_time, loss):
        print('  - {header:12}  accuracy: {accu:3.3f} %, loss: {loss:3.3f} '\
                'elapse: {elapse:3.3f} min'.format(
                    header=f"({header})",
                    accu=100*accu, loss=loss, elapse=(time.time()-start_time)/60))
    valid_losses = []
    for epoch in range(N_EPOCHS):
        print(f'[ Epoch | {epoch}:{N_EPOCHS}]')
        
        start_time = time.time() # 시작 시간
        train_loss, train_accu = train(model, train_iterator, optimizer, device, SRC_PAD_IDX, TRG_PAD_IDX)
        print_performances('Training', train_accu, start_time, train_loss)

        start_time = time.time()
        valid_loss, valid_accu = evaluate(model, valid_iterator, device, SRC_PAD_IDX, TRG_PAD_IDX)
        print_performances('Validation', valid_accu, start_time, valid_loss)
        
        valid_losses += [valid_loss]

        if valid_loss <= min(valid_losses):
            torch.save(model.state_dict(), "transformer_e2g.pt")
            print(f'[Info] Model has been updated - epoch: {epoch}')
        
        wandb.log({"loss": train_loss})
#########################################################################################################################
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')  # cuda(gpu) 사용
    
#==================== PREPARING ====================#

batch_size = 128
data = dill.load(open("data.pickle", 'rb'))

SRC_PAD_IDX = data['vocab']['src'].vocab.stoi["<blank>"]
TRG_PAD_IDX = data['vocab']['trg'].vocab.stoi["<blank>"]
#SRC_PAD_IDX = data['vocab']['src'].vocab.stoi[data['vocab']['src'].pad_token]
#TRG_PAD_IDX = data['vocab']['trg'].vocab.stoi[data['vocab']['trg'].pad_token]

INPUT_DIM = len(data['vocab']['src'].vocab)  # src_vocab_size
OUTPUT_DIM = len(data['vocab']['trg'].vocab)  # trg_vocab_size

fields = {'src': data['vocab']['src'], 'trg' : data['vocab']['trg']}

train_data = Dataset(examples=data['train_data'], fields=fields)
val_data = Dataset(examples=data['valid_data'], fields=fields)
#test_data = Dataset(examples=data['test_data'], fields=fields)

train_iterator = BucketIterator(train_data, batch_size=batch_size, device=device, train=True)
valid_iterator = BucketIterator(val_data, batch_size=batch_size, device=device)

#==================== CREATE MODEL ===================#
## Transformer
model = Transformer(INPUT_DIM, OUTPUT_DIM, SRC_PAD_IDX, TRG_PAD_IDX,hparams.HIDDEN_DIM,
                                hparams.ENC_LAYERS, hparams.DEC_LAYERS, hparams.ENC_HEADS, hparams.DEC_HEADS,
                                hparams.ENC_PF_DIM, hparams.DEC_PF_DIM, hparams.ENC_DROPOUT, hparams.DEC_DROPOUT, device).to(device)
#=====================================================#
N_EPOCHS = 50
#=====================================================#                           

## optimizer(Adam)
LEARNING_RATE = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=8000,
        num_training_steps = N_EPOCHS * len(train_iterator)
    ) 

#==================== START TRAIN ===================#
start_train(model, train_iterator, valid_iterator, optimizer, device, N_EPOCHS)

wandb.finish()

# Optional
wandb.watch(model)

