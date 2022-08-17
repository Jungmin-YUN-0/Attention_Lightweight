from doctest import OutputChecker
import torch.nn as nn
import torch
from model.EncoderLayer import EncoderLayer
from model.PositionalEncoding import PositionalEncoding
import copy


# encoder

##embedding + encoder layer
class Encoder(nn.Module):
    def __init__(self, enc_voc_size, SRC_PAD_IDX, d_model, n_layers, n_head, ffn_hidden, drop_prob, device, attn_option, n_position=512):
        super().__init__()
        self.device = device
        self.attn_option = attn_option

        # input dimenstion → embedding dimension
        self.tok_embedding = nn.Embedding(enc_voc_size, d_model, padding_idx=SRC_PAD_IDX)
        # positional embedding 학습 (sinusoidal x) =>> self.pos_embedding = nn.Embedding(max_len, d_model)
        self.pos_encoding = PositionalEncoding(d_model, n_position=n_position)
        #self.layers = nn.ModuleList([EncoderLayer.EncoderLayer(d_model, n_head, ffn_hidden, drop_prob, device) for _ in range(n_layers)])
        encoder_layer = EncoderLayer(d_model, n_head, ffn_hidden, drop_prob, device, attn_option, n_position)
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(n_layers)])
        self.dropout = nn.Dropout(drop_prob)
        #self.d_model = d_model
        self.scale = torch.sqrt(torch.FloatTensor([d_model])).to(device)  ## normalization (가중치에 d_model을 곱함)
        
    def forward(self, src, src_mask):

        x = src
        
        x = self.tok_embedding(x)
        x = x * self.scale
        #src *= self.d_model**0.5
        x = self.dropout(self.pos_encoding(x))
        
        
        if self.attn_option == "CT":
            topk_indices = None
            output = x
            for layer in self.layers:
                output, topk_indices = layer(output, src_mask, topk_indices)
            x = output
        else:
            for layer in self.layers :
                x = layer(x, src_mask)     

        return x        
        
