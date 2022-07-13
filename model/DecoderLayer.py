import torch.nn as nn
from model.MultiHeadAttention import MultiHeadAttention
from model.FeedForward import FeedForward


# decoder layer

##masked_, cross_, feedforward_
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, ffn_hidden, drop_prob, device):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, n_head, drop_prob, device)
        self.enc_dec_attention = MultiHeadAttention(d_model, n_head, drop_prob, device)
        self.ffn = FeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        
    def forward(self, trg, enc_src, src_mask=None, trg_mask=None):
        out, dec_slf_attn = self.self_attention(trg, trg, trg, mask=src_mask)
        out, dec_enc_attn = self.enc_dec_attention(out, enc_src, enc_src, mask=trg_mask)
        out = self.ffn(out)
        # residual connection, layer normalization은 각각의 sublayer에서 미리 수행하였음
        return out, dec_slf_attn, dec_enc_attn
