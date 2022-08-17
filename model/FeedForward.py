import torch.nn as nn
import torch

# feed-forward neural network
## FFN(x)=max(0, xW1+b1)W2+b2

class FeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob):
        super().__init__()
        # 1# linear transformation1_f1 = xW1+b1
        self.linear1 = nn.Linear(d_model, hidden)
        # 2# RELU_f2 = max(0, f1)
        # 3# linear transformation2_f3 = f2W2+b2
        self.linear2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.linear2(self.dropout(torch.relu(self.linear1(x))))
        #x = self.dropout(x)
        return x
