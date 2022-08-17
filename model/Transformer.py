import torch.nn as nn
import torch
import torch.nn.functional as F
from model.Encoder import Encoder
from model.Decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, INPUT_DIM, OUTPUT_DIM, SRC_PAD_IDX, TRG_PAD_IDX, HIDDEN_DIM, ENC_LAYERS, DEC_LAYERS, ENC_HEADS, DEC_HEADS, ENC_PF_DIM, DEC_PF_DIM, ENC_DROPOUT, DEC_DROPOUT, device, attn_option):
        super().__init__()

        self.device = device
        self.src_pad_idx = SRC_PAD_IDX
        self.trg_pad_idx = TRG_PAD_IDX

        self.encoder = Encoder(INPUT_DIM, SRC_PAD_IDX, HIDDEN_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device, attn_option)
        self.decoder = Decoder(OUTPUT_DIM, TRG_PAD_IDX, HIDDEN_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device, attn_option)

        self.trg_word_prj = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)    # HIDDEN_DIM=d_model
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                                        
    # source의 <pad> token -> mask=0 설정
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(-2) # unsqueeze -> 차원을 늘릴 때 사용
        ### torch.unsqueeze(input, dim) → Tensor. 지정된 위치에 1차원 크기가 삽입된 새 텐서를 반환
        return src_mask
         
    # target의 미래시점 단어 -> mask=0 설정
    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(-2)
        
        trg_len = trg.shape[1]
        trg_sub_mask = (1-torch.triu(torch.ones((1, trg_len, trg_len), device=self.device), diagonal=1)).bool()

        trg_mask = trg_pad_mask & trg_sub_mask  # element-wise
        
        return trg_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)  # src_mask : [batch_size, 1, 1, src_len]
        trg_mask = self.make_trg_mask(trg)  # trg_mask : [batch_size, 1, trg_len, trg_len]

        enc_src = self.encoder(src, src_mask)
        dec_output = self.decoder(trg, enc_src, trg_mask, src_mask)
        
        seq_logit = self.trg_word_prj(dec_output)
        output = seq_logit.view(-1, seq_logit.size(2))
        return F.log_softmax(output, dim=-1)    # softmax


#model.apply(utils.initialize_weights)
