import torch.nn as nn
import torch
import copy
from model.DecoderLayer import DecoderLayer
from model.PositionalEncoding import PositionalEncoding


# decoder

## embedding + encoder layer + linear transformation
class Decoder(nn.Module):
    def __init__(self, OUTPUT_DIM, TRG_PAD_IDX, HIDDEN_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROUPOUT, device):
        super().__init__()
        self.tok_embedding = nn.Embedding(OUTPUT_DIM, HIDDEN_DIM, padding_idx=TRG_PAD_IDX)
        self.pos_encoding = PositionalEncoding(HIDDEN_DIM, n_position=512)
        
        decoder_layer = DecoderLayer(HIDDEN_DIM, DEC_HEADS, DEC_PF_DIM, DEC_DROUPOUT, device)
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(DEC_LAYERS)])
        
        self.dropout = nn.Dropout(DEC_DROUPOUT)
        self.HIDDEN_DIM = HIDDEN_DIM # (=d_model)
        self.layer_norm = nn.LayerNorm(HIDDEN_DIM)
        #self.scale = torch.sqrt(torch.FloatTensor([HIDDEN_DIM])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        tok_emb = self.tok_embedding(trg)
        tok_emb *= self.HIDDEN_DIM ** 0.5  ## normalization (root(d_model)을 곱함)
        pos_emb = self.dropout(self.pos_encoding(tok_emb))
        output = self.layer_norm(pos_emb)

        for layer in self.layers:
            output, dec_slf_attn, dec_enc_attn = layer(output, enc_src, trg_mask, src_mask)

        return output
