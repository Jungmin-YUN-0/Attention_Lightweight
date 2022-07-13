import torch.nn as nn
from model.MultiHeadAttention import MultiHeadAttention
from model.FeedForward import FeedForward


# encoder layer

## multihead attention + feed-forward neural network
class EncoderLayer(nn.Module):
    # 입력dim = 출력dim (=> 여러 개 중첩 가능)
    def __init__(self, d_model, n_head, ffn_hidden, drop_prob, device):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, n_head, drop_prob, device)
        self.ffn = FeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)


    def forward(self, src, src_mask=None):
        out, attn_score = self.attention(src, src, src, mask=src_mask)
        out = self.ffn(out)
        # residual connection, layer normalization은 각각의 sublayer에서 미리 수행하였음
        return out, attn_score
        