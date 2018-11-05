import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

from sublayers import MultiHeadAttention, MLP, DocRepAttention


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

    def forward(self, inputs):
	    x = inputs
	    
	    # Layer Normalization
	    x_norm = self.layer_norm_mha(x)
	    
	    # Multi-head attention
	    y = self.multi_head_attention(x_norm, x_norm, x_norm)
	    
	    # Dropout and residual
	    x = self.dropout(x + y)
	    
	    # Layer Normalization
	    x_norm = self.layer_norm_ffn(x)
	    
	    # Positionwise Feedforward
	    y = self.positionwise_feed_forward(x_norm)
	    
	    # Dropout and residual
	    y = self.dropout(x + y)
	    
	    return y

