from dataclasses import fields
from email.policy import default
from tqdm import tqdm, tqdm_notebook
import dill
import time
from transformers import get_scheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext.data import Dataset, BucketIterator
from model.Transformer import Transformer
from model.Classification import CF_Transformer
import argparse
import wandb
import numpy as np



class Train():
    def __init__(self, gpu, opt, batch_size, n_epoch, data_pkl, model_save, learning_rate, num_warmup, hidden_dim, n_layer, n_head, ff_dim, dropout, data_task):

        gpu = "cuda:"+gpu
        device = torch.device(gpu if torch.cuda.is_available() else 'cpu')  # cuda(gpu) 사용

        HIDDEN_DIM = hidden_dim
        ENC_LAYERS = DEC_LAYERS = n_layer
        ENC_HEADS = DEC_HEADS = n_head
        ENC_PF_DIM = DEC_PF_DIM = ff_dim
        ENC_DROPOUT = DEC_DROPOUT = dropout
        saved_model = model_save
        N_EPOCHS = n_epoch
        LEARNING_RATE  = learning_rate
        attn_option = opt

        SEED=42
        torch.manual_seed(SEED) # torch 
        torch.cuda.manual_seed(SEED)
        np.random.seed(SEED) # numpy
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)


    ########################################################################################################################################################
        if data_task == "MT":
            def patch_trg(trg, pad_idx):
                trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
                return trg, gold

            def cal_performance(pred, gold, TRG_PAD_IDX, criterion):
                gold = gold.contiguous().view(-1)
                loss = criterion(pred, gold)
                #loss = F.cross_entropy(pred, gold, ignore_index=TRG_PAD_IDX, reduction='sum')

                pred = pred.max(1)[1]

                non_pad_mask = gold.ne(TRG_PAD_IDX)    # ne is =!, eq is ==
                n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
                n_word = non_pad_mask.sum().item()
                return loss, n_correct, n_word    

            #############################
            # model training (per epoch)# 
            #############################
            def train(model, iterator, optimizer, device, SRC_PAD_IDX, TRG_PAD_IDX, criterion, data):
                model.train() # 학습 모드
                total_loss, n_word_total, n_word_correct = 0, 0, 0 
                
                ####

                for idx,batch in enumerate(tqdm(iterator)):  ##leave=False
                    # prepare data
                    #src_tensor = torch.zeros(512, dtype=torch.long) #src_max_len = 512
                    #src_tensor[:len(src)] = torch.tensor(src, dtype=torch.long)
                    #batch.src 

                    src_seq = batch.src.to(device)
                    trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch.trg, TRG_PAD_IDX))
                    
                    # forward
                    optimizer.zero_grad()
                    pred = model(src_seq, trg_seq)
                    

                    # backward & update parameters
                    loss, n_correct, n_word = cal_performance(pred, gold, TRG_PAD_IDX, criterion)
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
            def evaluate(model, iterator, device, SRC_PAD_IDX, TRG_PAD_IDX, criterion):
                model.eval() # 평가 모드
                total_loss, n_word_total, n_word_correct = 0, 0, 0
                
                with torch.no_grad():
                    for batch in tqdm(iterator):  #leave=False
                        # prepare data
                        src_seq = batch.src.to(device)
                        trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch.trg, TRG_PAD_IDX))
                        
                        # forward
                        pred = model(src_seq, trg_seq)
                        loss, n_correct, n_word = cal_performance(pred, gold, TRG_PAD_IDX, criterion)

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
            def start_train(model, train_iterator, valid_iterator, optimizer, device, N_EPOCHS, criterion):
                def print_performances(header, accu, start_time, loss):
                    print('  - {header:12}  accuracy: {accu:3.3f} %, loss: {loss:3.3f} '\
                            'elapse: {elapse:3.3f} min'.format(
                                header=f"({header})",
                                accu=100*accu, loss=loss, elapse=(time.time()-start_time)/60))
                
                best_valid_loss = float('inf')
                #valid_losses = []
                time_check = []
                for epoch in range(N_EPOCHS):
                    print(f'[ Epoch | {epoch}:{N_EPOCHS}]')
                    start_time_check = time.time()
                    start_time = time.time() # 시작 시간
                    train_loss, train_accu = train(model, train_iterator, optimizer, device, SRC_PAD_IDX, TRG_PAD_IDX, criterion, data)
                    print_performances('Training', train_accu, start_time, train_loss)
                    
                    start_time = time.time()
                    valid_loss, valid_accu = evaluate(model, valid_iterator, device, SRC_PAD_IDX, TRG_PAD_IDX, criterion)
                    print_performances('Validation', valid_accu, start_time, valid_loss)
                    
                    end_time_check = time.time()
                    time_check.append(end_time_check - start_time_check)
                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        torch.save(model.state_dict(), saved_model)
                        print(f'[Info] Model has been updated - epoch: {epoch}')
                    #valid_losses += [valid_loss]
                
                    wandb.log({"train_loss": train_loss})
                    wandb.log({"valid_loss": valid_loss})
                    wandb.log({"train_accuracy": train_accu})   
                    wandb.log({"valid_accuracy": valid_accu})
                print(time_check/8)
        ########################################################################################################################
        ########################################################################################################################
        
        elif data_task == "CF":

            def train(model, train_iterator, optimizer, device, criterion):  #def train_model(self, train_iterator):
                model.train()
                epoch_losses = 0
                epoch_accs = 0

                for batches in tqdm(train_iterator):
                    input_seq, _ = batches.src
                    target = batches.trg
                    pred = model(input_seq)
                    loss = criterion(pred, target)  #target.contiguous().view(-1)
                    
                    pred_class = pred.argmax(dim=-1)
                    correct_pred = pred_class.eq(target).sum()
                    accuracy = correct_pred / pred.shape[0] #pred.shape[0]: batch_size
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()             
                    lr_scheduler.step()

                    epoch_losses += loss.item()
                    epoch_accs += accuracy.item()

                return epoch_losses/len(train_iterator), epoch_accs/len(train_iterator)

                
            def evaluate(model, iterator, device, criterion):
                model.eval()

                epoch_losses = 0
                epoch_accs = 0

                with torch.no_grad():
                    for batch in tqdm(iterator):
                        input_seq, _ = batch.src
                        target = batch.trg
                        pred = model(input_seq)
                        loss = criterion(pred, target)

                        pred_class = pred.argmax(dim=-1)
                        correct_pred = pred_class.eq(target).sum()
                        accuracy = correct_pred / pred.shape[0] #pred.shape[0]: batch_size

                        epoch_losses += loss.item()
                        epoch_accs += accuracy.item()

                return epoch_losses/len(iterator), epoch_accs/len(iterator)


            def start_train(model, train_iterator, valid_iterator, optimizer, device, N_EPOCHS, criterion):
                best_valid_loss = float('inf')
                train_losses = []
                train_accs = []
                valid_losses = []
                valid_accs = []

                for epoch in range(n_epoch):
                    train_loss, train_acc = train(model, train_iterator, optimizer, device, criterion)
                    valid_loss, valid_acc = evaluate(model, valid_iterator, device, criterion)
                    
                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        torch.save(model.state_dict(), saved_model)
                        print(f'[Info] Model has been updated - epoch: {epoch}')

                    print(f'epoch: {epoch+1}')
                    print(f'train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f}')
                    print(f'valid_loss: {valid_loss:.3f}, valid_acc: {valid_acc:.3f}')

                    wandb.log({"train_loss": train_loss})
                    wandb.log({"valid_loss": valid_loss})
                    wandb.log({"train_accuracy": train_acc})   
                    wandb.log({"valid_accuracy": valid_acc})
                
            
    ########################################################################################################################################################
        
        #==================== PREPARING ====================#
        data = dill.load(open(data_pkl, 'rb'))

        SRC_PAD_IDX = data['vocab']['src'].vocab.stoi["<blank>"]
        if data_task == "MT":
            TRG_PAD_IDX = data['vocab']['trg'].vocab.stoi["<blank>"]
            #SRC_PAD_IDX = data['vocab']['src'].vocab.stoi[data['vocab']['src'].pad_token]
            #TRG_PAD_IDX = data['vocab']['trg'].vocab.stoi[data['vocab']['trg'].pad_token]

        INPUT_DIM = len(data['vocab']['src'].vocab)  # src_vocab_size
        OUTPUT_DIM = len(data['vocab']['trg'].vocab)  # trg_vocab_size

        fields = {'src': data['vocab']['src'], 'trg' : data['vocab']['trg']}

        train_data = Dataset(examples=data['train_data'], fields=fields)
        val_data = Dataset(examples=data['valid_data'], fields=fields)
        ##test_data = Dataset(examples=data['test_data'], fields=fields) #token figure#

        #train_iterator = torch.utils.data.DataLoader(train_data, batch_size=batch_size, drop_last=True)
        #valid_iterator = torch.utils.data.DataLoader(val_data, batch_size=batch_size, drop_last=True)
        train_iterator = BucketIterator(train_data, batch_size=batch_size, device=device, train=True)
        valid_iterator = BucketIterator(val_data, batch_size=batch_size, device=device)
        ##test_iterator = BucketIterator(test_data, batch_size=batch_size, device=device) #token figure#
        ##train_iterator = test_iterator #token figure#
        '''
        import spacy
        from torchtext.data import TabularDataset, Field
        spacy_en = spacy.load('en_core_web_sm')  # en tokenization
        def tokenize_en(text):
            return [token.text for token in spacy_en.tokenizer(text)]
        
        SRC = Field(tokenize=tokenize_en, init_token="<sos>", eos_token="<eos>", pad_token="<blank>", lower=True, batch_first=True, fix_length=512, include_lengths=True)
        TRG = Field(sequential=False, use_vocab=False, is_target=True, unk_token=None)
        
        train_data = TabularDataset(path= './hyperpartisan/train.csv', format="csv", fields=[('src', SRC), ('trg', TRG)], skip_header=True)
        valid_data = TabularDataset(path='./hyperpartisan/val.csv', format="csv", fields=[('src', SRC), ('trg', TRG)], skip_header=True)
        
        SRC.build_vocab(train_data.src, min_freq=2) #!#
        TRG.build_vocab(train_data.trg, min_freq=2) #!#
        
        INPUT_DIM = len(SRC.vocab)
        OUTPUT_DIM = len(TRG.vocab)

        train_iterator = BucketIterator(train_data, batch_size=batch_size, device=device, train=True)
        valid_iterator = BucketIterator(valid_data, batch_size=batch_size, device=device)

        SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
        #TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
        '''

        #==================== CREATE MODEL ===================#
        ## Transformer
        if data_task == "MT":
            model = Transformer(INPUT_DIM, OUTPUT_DIM, SRC_PAD_IDX, TRG_PAD_IDX, HIDDEN_DIM,
                                ENC_LAYERS, DEC_LAYERS, ENC_HEADS, DEC_HEADS,
                                ENC_PF_DIM, DEC_PF_DIM, ENC_DROPOUT, DEC_DROPOUT, device, attn_option).to(device)
        elif data_task == "CF":
            model = CF_Transformer(INPUT_DIM, OUTPUT_DIM, SRC_PAD_IDX, HIDDEN_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device, attn_option).to(device)
        
        LEARNING_RATE = 0.0001
        ## optimizer(Adam)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        
        lr_scheduler = get_scheduler(
                "linear",
                optimizer=optimizer,
                num_warmup_steps=num_warmup,
                num_training_steps = N_EPOCHS * len(train_iterator)
            ) 
        
        if data_task == "MT":
            criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)
        elif data_task == 'CF':
            criterion = nn.CrossEntropyLoss()

        
        #==================== START TRAIN ====================#
        wandb.init(project="transformer", entity="jungminy")
        #wandb.run.name = f"{saved_model}_{LEARNING_RATE}_{N_EPOCHS}_{batch_size}"
        #print(wandb.run.name)
        wandb.config = {
            "learning_rate": learning_rate,
            "epochs": n_epoch,
            "batch_size": batch_size
        }
        
        #optional
        #wandb.watch(model)
        
        
        start_train(model, train_iterator, valid_iterator, optimizer, device, N_EPOCHS, criterion)
        
##################################################################################################################

if __name__ == "__main__":
    Train()
