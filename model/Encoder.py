import torch.nn as nn
import torch
from model.EncoderLayer import EncoderLayer
from model.PositionalEncoding import PositionalEncoding
import copy


# encoder

##embedding + encoder layer
class Encoder(nn.Module):
    def __init__(self, INPUT_DIM, SRC_PAD_IDX, HIDDEN_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device):
        super().__init__()

        self.device = device
        self.tok_embedding = nn.Embedding(INPUT_DIM, HIDDEN_DIM, padding_idx=SRC_PAD_IDX)
        self.pos_embedding = PositionalEncoding(HIDDEN_DIM, n_position=512)

        encoder_layer = EncoderLayer(HIDDEN_DIM, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device)
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(ENC_LAYERS)])

        self.dropout = nn.Dropout(ENC_DROPOUT)
        self.HIDDEN_DIM = HIDDEN_DIM # (=d_model)
        self.layer_norm = nn.LayerNorm(HIDDEN_DIM)

        self.d_model = HIDDEN_DIM
#d_model, n_head, ffn_hidden, drop_prob, device):
    def forward(self, src_seq, src_mask):
        
        tok_emb = self.tok_embedding(src_seq)
        tok_emb *= self.HIDDEN_DIM ** 0.5  ## normalization (root(d_model)을 곱함)
        pos_emb = self.dropout(self.pos_embedding(tok_emb))
        output = self.layer_norm(pos_emb)

        for layer in self.layers:
            output, attn_score = layer(output, src_mask)
        return output


        