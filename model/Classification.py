import torch.nn as nn
import torch
import torch.nn.functional as F
from model.MultiHeadAttention import MultiHeadAttention_CF, LinformerSelfAttention_CF, LinformerSelfAttention_CF_test
from model.PositionalEncoding import PositionalEncoding
import copy


class Classification_block(nn.Module):
    def __init__(self, HIDDEN_DIM, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device, attn_option, n_position, topk_indices=None):
        super().__init__()
        heads = ENC_HEADS 
        ffnn_hidden_size = ENC_PF_DIM 
        dmodel = HIDDEN_DIM
        self.topk_indices = topk_indices
        self.attn_option = attn_option
        #self.idx = idx

        if self.attn_option == "BASE":
            self.attention = MultiHeadAttention_CF(dmodel, heads, ENC_DROPOUT, device)
        elif self.attn_option == "LR":
            self.attention = LinformerSelfAttention_CF(dmodel, heads, ENC_DROPOUT, device, n_position, k=128)
        elif self.attn_option == 'CT' :
            #self.attention = LinformerSelfAttention_CF(dmodel, heads, ENC_DROPOUT, device, n_position, k=128, pruning=True)  #mixture model
            self.attention = LinformerSelfAttention_CF_test(dmodel, heads, ENC_DROPOUT, device, n_position, k=128, pruning=True)  #only core-token attention
        self.layer_norm1 = nn.LayerNorm(dmodel)
        self.layer_norm2 = nn.LayerNorm(dmodel)
        
        self.ffnn = nn.Sequential(
                nn.Linear(dmodel, ffnn_hidden_size),
                nn.ReLU(),
                nn.Dropout(ENC_DROPOUT),
                nn.Linear(ffnn_hidden_size, dmodel))

    def forward(self, inputs, topk_indices=None): #!!#
        """Forward propagate through the Transformer block.
        Parameters
        ----------
        inputs: torch.Tensor
            Batch of embeddings.
        Returns
        -------
        torch.Tensor
            Output of the Transformer block (batch_size, seq_length, dmodel)
        """
        # Inputs shape (batch_size, seq_length, embedding_dim = dmodel)
        if self.attn_option == 'CT':
            if topk_indices is None:
                attn_output, topk_indices = self.attention(inputs)
            elif topk_indices is not None:
                attn_output, topk_indices = self.attention(inputs, topk_indices)
        else:
            attn_output = self.attention(inputs)
        
        output = inputs + attn_output           
        output = self.layer_norm1(output)            
        output = output + self.ffnn(output)            
        output = self.layer_norm2(output)

        # Output shape (batch_size, seq_length, dmodel)
        if self.attn_option == 'CT' :
            return output, topk_indices
        else:
            return output


class CF_Transformer(nn.Module):
    """Implementation of the Transformer model for classification.
    
    Parameters
    ----------
    vocab_size: int
        The size of the vocabulary.
    dmodel: int
        Dimensionality of the embedding vector.
    max_len: int
        The maximum expected sequence length.
    padding_idx: int, optional (default=0)
        Index of the padding token in the vocabulary and word embedding.
    n_layers: int, optional (default=4)
        Number of the stacked Transformer blocks.    
    ffnn_hidden_size: int, optonal (default=dmodel * 4)
        Position-Wise-Feed-Forward Neural Network hidden size.
    heads: int, optional (default=8)
        Number of the self-attention operations to conduct in parallel.
    pooling: str, optional (default='max')
        Specify the type of pooling to use. Available options: 'max' or 'avg'.
    dropout: float, optional (default=0.2)
        Probability of an element of the tensor to be zeroed.
    """
    
    def __init__(self, enc_voc_size, dec_voc_size, SRC_PAD_IDX, dmodel, n_layers, n_head, ffnn_hidden, drop_prob, device, attn_option, n_position=512):#512
        
        super().__init__()
        self.tok_embedding = nn.Embedding(enc_voc_size, dmodel, padding_idx=SRC_PAD_IDX)
        self.pos_encoding = PositionalEncoding(dmodel, n_position=n_position)
        self.tnf_blocks = nn.ModuleList()
        self.attn_option = attn_option

        layer = Classification_block(dmodel, n_head, ffnn_hidden, drop_prob, device, attn_option, n_position)
        self.tnf_blocks = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])   

        self.dropout = nn.Dropout(drop_prob)            
        self.linear = nn.Linear(dmodel, dec_voc_size)
        self.scale = torch.sqrt(torch.FloatTensor([dmodel])).to(device)

    def forward(self, inputs):
        """Forward propagate through the Transformer.
        Parameters ---
        inputs: torch.Tensor
            Batch of input sequences.
        input_lengths: torch.LongTensor
            Batch containing sequences lengths.
            
        Returns ---
        torch.Tensor
            Logarithm of softmaxed class tensor.
        """
        self.batch_size = inputs.size(0)

        output = self.tok_embedding(inputs)
        output = output*self.scale
        output = self.dropout(self.pos_encoding(output))

        if self.attn_option == "CT":
            topk_indices = None
            tnf_output = output
            for layer in self.tnf_blocks :
                tnf_output, topk_indices = layer(tnf_output, topk_indices)
            output = tnf_output
        else:
            for layer in self.tnf_blocks :
                output = layer(output)
        # Output dimensions (batch_size, seq_length, dmodel)
        
        ## opt1 max pooling
        # Permute to the shape (batch_size, dmodel, seq_length)
        # Apply max-pooling, output dimensions (batch_size, dmodel)
        output = F.adaptive_max_pool1d(output.permute(0,2,1), (1,)).view(self.batch_size,-1)
        ## opt2 avg pooling
        # Sum along the batch axis and divide by the corresponding lengths (FloatTensor)
        # Output shape: (batch_size, dmodel)
        #output = torch.sum(output, dim=1) / input_lengths.view(-1,1).type(torch.FloatTensor) 
        
        #seq_logit = self.linear(output)
        #output = seq_logit.view(-1, seq_logit.size(2))

        output = self.linear(output)
        #print(output)
        #print(output.shape)
        #print(F.log_softmax(output, dim=-1).shape)
        #print(F.log_softmax(output, dim=-1))
        return F.log_softmax(output, dim=-1)
        
