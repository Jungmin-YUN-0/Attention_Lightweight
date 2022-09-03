import torch
import dill
from tqdm import tqdm
from torchtext.data import Dataset, BucketIterator
from torchtext.data.metrics import bleu_score  # BLEU
from model.Transformer import Transformer
from model.Classification import CF_Transformer
from model.translator import Translator
import argparse
import wandb
import torch.nn.functional as F
#from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
#from sklearn.metrics import precision_score
#from sklearn.metrics import recall_score
from sklearn import metrics

class Test():
    def __init__(self, gpu, opt, batch_size, data_pkl, model_save, pred_save, hidden_dim, n_layer, n_head, ff_dim, dropout, data_task):

        self.data_pkl = data_pkl
        self.saved_model = model_save
        self.saved_result = pred_save

        gpu = "cuda:"+gpu
        device = torch.device(gpu if torch.cuda.is_available() else 'cpu')  # cuda(gpu) 사용

        self.HIDDEN_DIM = hidden_dim
        self.ENC_LAYERS = self.DEC_LAYERS = n_layer
        self.ENC_HEADS = self.DEC_HEADS = n_head
        self.ENC_PF_DIM = self.DEC_PF_DIM = ff_dim
        self.ENC_DROPOUT = self.DEC_DROPOUT = dropout
        attn_option = opt
   
        ##########################################################################################################
        data = dill.load(open(self.data_pkl, 'rb'))

        SRC, TRG = data['vocab']['src'], data['vocab']['trg']
        SRC_PAD_IDX = SRC.vocab.stoi["<blank>"]
        if data_task == "MT":
            TRG_PAD_IDX = TRG.vocab.stoi["<blank>"]
            TRG_SOS_IDX = TRG.vocab.stoi["<sos>"]
            TRG_EOS_IDX = TRG.vocab.stoi["<eos>"]

        INPUT_DIM = len(SRC.vocab)
        OUTPUT_DIM = len(TRG.vocab)

        test_data = Dataset(examples=data['test_data'], fields={'src': SRC, 'trg': TRG})
        test_iterator = BucketIterator(test_data, batch_size=batch_size, device=device)

        if data_task == "MT":
            model = Transformer(INPUT_DIM, OUTPUT_DIM, SRC_PAD_IDX, TRG_PAD_IDX, self.HIDDEN_DIM,
                            self.ENC_LAYERS, self.DEC_LAYERS, self.ENC_HEADS, self.DEC_HEADS,
                            self.ENC_PF_DIM, self.DEC_PF_DIM, self.ENC_DROPOUT, self.DEC_DROPOUT, device, attn_option).to(device)
        elif data_task == "CF":
            model = CF_Transformer(INPUT_DIM, OUTPUT_DIM, SRC_PAD_IDX, self.HIDDEN_DIM, self.ENC_LAYERS, self.ENC_HEADS, self.ENC_PF_DIM, self.ENC_DROPOUT, device, attn_option).to(device)

        model.load_state_dict(torch.load(self.saved_model))
        print('[Info] Trained model state loaded.')

        

        #wandb.init(project="transformer", entity="jungminy")

        #import torchvision.models as models
        #from ptflops import get_model_complexity_info

        #dummy_size = (256, 1)

        #macs, params = get_model_complexity_info(model, dummy_size, as_strings=True, print_per_layer_stat=False, verbose=False)
                                        
        #print('computational complexity: ', macs)
        #print('number of parameters: ', params)

        #===============================================================#
        if data_task == 'MT':
            translator = Translator(model=model, beam_size=5, max_seq_len=512, src_pad_idx=SRC_PAD_IDX, trg_pad_idx=TRG_PAD_IDX, trg_bos_idx=TRG_SOS_IDX, trg_eos_idx=TRG_EOS_IDX, device=device).to(device)
            unk_idx = SRC.vocab.stoi[SRC.unk_token]

            pred_trgs = []
            trgs= []
            index=0
            
            print('[Info] Inference ...')
            if attn_option == 'BASE':
                with open(self.saved_result, 'w') as f:
                    for example in tqdm(test_data, desc='  - (Test)', leave=False):
                        #print(' '.join(example.src))
                        src_seq = [SRC.vocab.stoi.get(word, unk_idx) for word in example.src]

                        src_seq_ext = torch.zeros((1,512), dtype=torch.long) #!#
                        src_seq_ext[:, :len(src_seq)] = torch.LongTensor([src_seq]) #!#
                        pred_seq = translator.translate_sentence(src_seq_ext.to(device)) #!#

                        pred_seq = translator.translate_sentence(torch.LongTensor([src_seq]).to(device))
                        
                        pred_line = ' '.join(TRG.vocab.itos[idx] for idx in pred_seq)
                        #pred_line = pred_line.replace("<sos>", '').replace("<eos>", '')
                        pred_line = pred_line.replace("<sos>", '').replace("<unk>", '').replace("<eos>", '')
                        #print(pred_line)
                        f.write(pred_line.strip() + '\n')
                        #f.write(' '.join(vars(example)['trg']) + '\n')

                        ## for BLEU_score
                        pred_trgs.append(pred_line.split(" ")) ##예측            
                        trgs.append([vars(example)['trg']]) ##정답
                        if (index%100)==0:
                            print(f"[{index} / {len(test_data)}")
                            print(f'예측: {pred_line}')
                            print(f"정답: {' '.join(vars(example)['trg'])}")
                        index += 1
                bleu = bleu_score(pred_trgs, trgs, max_n=4, weights=[0.25,0.25,0.25,0.25])
                print('[Info] Finished.')
                print(f'BLEU score : {bleu}')
            
            elif attn_option == 'LR' or attn_option =='CT':
                with open(self.saved_result, 'w') as f:
                    for example in tqdm(test_data, desc='  - (Test)', leave=False):
                        src_seq = [SRC.vocab.stoi.get(word, unk_idx) for word in example.src]
                        
                        src_seq_ext = torch.zeros((1,512), dtype=torch.long) #!#
                        src_seq_ext[:, :len(src_seq)] = torch.LongTensor([src_seq]) #!#
                        pred_seq = translator.translate_sentence(src_seq_ext.to(device)) #!#

                        pred_line = ' '.join(TRG.vocab.itos[idx] for idx in pred_seq)
                        pred_line = pred_line.replace("<sos>", '').replace("<unk>", '').replace("<eos>", '')
                        #print(pred_line)
                        f.write(pred_line.strip() + '\n')

                        ## for BLEU_score
                        pred_trgs.append(pred_line.split(" ")) ##예측            
                        trgs.append([vars(example)['trg']]) ##정답
                        if (index%100)==0:
                            print(f"[{index} / {len(test_data)}")
                            print(f'예측: {pred_line}')
                            print(f"정답: {' '.join(vars(example)['trg'])}")
                        index += 1
                        
                bleu = bleu_score(pred_trgs, trgs, max_n=4, weights=[0.25,0.25,0.25,0.25])
                print('[Info] Finished.')
                print(f'BLEU score : {bleu}')
            #wandb.log({"bleu_score": bleu})

            return
##################################################################################################################
        if data_task == 'CF':
            def get_predictions(model, iterator, device):
                model.eval()

                epoch_accuracy = 0
                
                targets = torch.tensor([]).to(device)
                preds= torch.tensor([]).to(device)

                with torch.no_grad():
                    for batch in tqdm(iterator):
                        input_seq, _ = batch.src
                        
                        target = batch.trg
                        pred = model(input_seq)

                        pred_class = pred.argmax(dim=-1)
                        correct_pred = pred_class.eq(target).sum()
                        accuracy = correct_pred / pred.shape[0]

                        epoch_accuracy += accuracy.item()         

                        targets = torch.cat([targets,target])
                        preds = torch.cat([preds, pred_class])

                        #F1_score = f1_score(target.cpu(), pred_class.cpu(), average='binary')
                        #Recall_score =  recall_score(target.cpu(), pred_class.cpu(), average='binary')
                        #Precision_score = precision_score(target.cpu(), pred_class.cpu(), average='binary')
                        #F1_score = f1_score(target.cpu(), pred_class.cpu(), average='weighted')#!!#average=binary
                        #F1_score_t = tfa.metrics.F1Score(num_classes=2, average="weighted")
                        
                        #print(metrics.classification_report(target.cpu(), pred_class.cpu(), digits=4))
                    #F1_score = f1_score(targets.cpu(), preds.cpu(), average='binary')# number of class = 2
                    F1_score = f1_score(targets.cpu(), preds.cpu(), average='weighted') # number of class > 2
                return epoch_accuracy / len(iterator), F1_score

            acc, F1_score = get_predictions(model, test_iterator, device)

            print(f"accuracy: {acc}")
            print(f"f1_score: {F1_score}")
            #print(f"f1_score(tensorflow): {F1_score_t}")

            #print(f"recall_score: {Recall_score}")
            #print(f"precision_score: {Precision_score}")
        # from thop import profile
        # flops, params = profile(model, inputs=(torch.randn(1,64).to(device).long(),), verbose=False)
        # print(flops, params)
##################################################################################################################

if __name__ == "__main__":
    Test()
