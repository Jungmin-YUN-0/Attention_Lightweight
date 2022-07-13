import torch.nn as nn
import torch
import torch.nn.functional as F
from model.Encoder import Encoder
from model.Decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, INPUT_DIM, OUTPUT_DIM, SRC_PAD_IDX, TRG_PAD_IDX, HIDDEN_DIM, ENC_LAYERS, DEC_LAYERS, ENC_HEADS, DEC_HEADS, ENC_PF_DIM, DEC_PF_DIM, ENC_DROPOUT, DEC_DROPOUT, device):
        super().__init__()

        self.device = device
        self.src_pad_idx = SRC_PAD_IDX
        self.trg_pad_idx = TRG_PAD_IDX

        self.encoder = Encoder(INPUT_DIM, SRC_PAD_IDX, HIDDEN_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device)
        self.decoder = Decoder(OUTPUT_DIM, TRG_PAD_IDX, HIDDEN_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device)
        self.trg_word_prj = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)    # HIDDEN_DIM=d_model
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                                        
    # source의 <pad> token -> mask=0 설정
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(-2)
        ## unsqueeze -> 차원을 늘릴 때 사용
        ### torch.unsqueeze(input, dim) → Tensor. 지정된 위치에 1차원 크기가 삽입된 새 텐서를 반환
        return src_mask
         
    # target의 미래시점 단어 -> mask=0 설정
    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(-2)
        
        trg_len = trg.size(1)
        trg_sub_mask = (1-torch.triu(torch.ones((1, trg_len, trg_len), device=self.device), diagonal=1)).bool()
        
        trg_mask = trg_pad_mask & trg_sub_mask  # element-wise
        return trg_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)  # src_mask : [batch_size, 1, 1, src_len]
        trg_mask = self.make_trg_mask(trg)  # trg_mask : [batch_size, 1, trg_len, trg_len]

        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(trg, enc_output, trg_mask, src_mask)
        
        seq_logit = self.trg_word_prj(dec_output)
        output = seq_logit.view(-1, seq_logit.size(2))
        return output
        #return F.log_softmax(output, dim=-1)    # softmax


#model.apply(utils.initialize_weights)

'''

class Transformer(nn.Module):

    def __init__(
            self, n_src_vocab, n_trg_vocab, src_pad_idx, trg_pad_idx,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
            trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True,
            scale_emb_or_prj='prj'):

        super().__init__()

        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx

        assert scale_emb_or_prj in ['emb', 'prj', 'none']
        scale_emb = (scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False
        self.scale_prj = (scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False
        self.d_model = d_model

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout, scale_emb=scale_emb)

        self.decoder = Decoder(
            n_trg_vocab=n_trg_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=trg_pad_idx, dropout=dropout, scale_emb=scale_emb)

        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

    def forward(self, src_seq, trg_seq):

        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)

        enc_output, *_ = self.encoder(src_seq, src_mask)
        dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
        seq_logit = self.trg_word_prj(dec_output)
        if self.scale_prj:
            seq_logit *= self.d_model ** -0.5

        return seq_logit.view(-1, seq_logit.size(2))
'''