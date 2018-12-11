import torch
import torch.nn as nn
from torch.autograd.variable import Variable
import torch.functional as F

import math

class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding for
    non-recurrent neural networks.
    Implementation based on "Attention Is All You Need"
    Args:
       dropout (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, dropout, dim, max_len=1000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2) * -(math.log(10000.0) / dim)).float())
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, pe_size)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb):
        """ emb (seq_len, BS, dim)
            pe (seq_len, 1, dim)
        """
        self.pe = self.pe.cuda()
        emb = emb * math.sqrt(self.dim)
        emb = emb + self.pe[:emb.size(0)]
        emb = self.dropout(emb)
        return emb
