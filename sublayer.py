import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

from sublayers import MultiHeadAttention, MLP, DocRepAttention

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_dim)
        )

    def forward(self, x):
        x = self.layers(x)
        return x

class Attention(nn.Module):
    def __init__(self, word_dim, query_length, doc_length, num_heads, 
                bias_mask=None, dropout=0.0, input_dim, hidden_size, batch_size):
    """
    Parameters:
    word_dim: The embedding dim of word     
    query_length: the size of the query
    doc_length: the size of the document
    num_heads: the number of the num_heads
    output_dim: the output dim of the multi-layer attention
    batch_size: size of batch
    hidden_size: Hidden size of the mlp at the second layer
    total_key_depth: Size of last dimension of keys. Must be divisible by num_head
    """
        
        super(EncoderLayer, self).__init__()
        
        self.multi_head_attention = MultiHeadAttention(word_dim, query_length, doc_length, 
                    num_heads, bias_mask, dropout)
        self.MLP = MLP(word_dim, hidden_size, word_dim)
        
        self.DocRepAttention = DocRepAttention(batch_size, hidden_size, query_length, word_dim)

    def forward(self, queries, doc):
        
        # Multi-head attention
        y = self.multi_head_attention(queries, doc)  #shape=[bs, query_len, dim]
        
        # MLP layer
        y = self.MLP(y)

        # Document global representation
        Drep = DocRepAttention(y)  # shape=[bs, 1, dim]
        
        return Drep

