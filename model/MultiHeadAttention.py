import torch.nn as nn
import torch

# multi-head attention
## input: key, query, value
### d_model : 하나의 단어에 대한 임베딩 차원

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, drop_prob, device):
        super().__init__()
        assert d_model % n_head == 0  # 필요조건

        self.d_model = d_model  # 각 word에서의 임베딩 차원
        self.n_head = n_head
        self.head_dim = d_model // n_head  # 각 head에서의 임베딩 차원 

        self.weight_q = nn.Linear(d_model, d_model, bias=False)  # query weight(FC layer) ## Linear(Q)=Q*W_Q
        self.weight_k = nn.Linear(d_model, d_model, bias=False)  # key weight(FC layer)
        self.weight_v = nn.Linear(d_model, d_model, bias=False)  # value weight(FC layer)
        self.fc_concat = nn.Linear(d_model, d_model, bias=False)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        self.dropout1 = nn.Dropout(drop_prob)
        self.dropout2 = nn.Dropout(drop_prob)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.shape[0]
        
        residual = Q
        
        # query, key, value -> Q, K, V  
        Q, K, V = self.weight_q(Q), self.weight_k(K), self.weight_v(V)
        
        ## multi-head attention (d_model -> n_head*head_dim 형태로 변환) / (n_head개 마다 각각 head_dim을 갖도록      ##[batch, length, n_head, head_dim] -> [batch, n_head, length, head_dim]
        Q = Q.view(batch_size, -1, self.n_head, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_head, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_head, self.head_dim).transpose(1, 2)
        # -1 => 나머지 값들에 의해 결정됨
        
        #=================================================================================================
        ## compute similarity(dot-product)      ##[batch, n_head, query_len, key_len]
        attn_score = torch.matmul(Q / (self.head_dim**0.5), K.transpose(2,3))
        
        ## option_masking 
        if mask is not None:
            mask = mask.unsqueeze(1)
            attn_score = attn_score.masked_fill(mask == 0, -1e10)
                    
        attn_dstn = torch.softmax(attn_score, dim=-1)
        # dim=-1 => 현재 input이 tensor(vector였으면 dim 지정 필요 x) last dimension에 대하여 softmax 적용
        # ex. nn.Softmax(dim=-1) torch.randn(2,3) -> 3     
                 
        ## scaled dot-product attention
        out = torch.matmul(self.dropout1(attn_dstn), V)
        #=================================================================================================
        out = out.permute(0,2,1,3).contiguous()
        out = out.view(batch_size, -1, self.d_model)
        out = self.fc_concat(out)
        out = self.dropout2(out)

        out += residual
        
        q = self.layer_norm(out)
        
        return out, attn_score
    