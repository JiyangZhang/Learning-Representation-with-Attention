from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

from .sublayers import MultiHeadAttention, PositionwiseFeedForward
from ..normalization import LayerNorm


class EncoderLayer(nn.Module):
  """
  Represents one Encoder layer of the Transformer Encoder
  Refer Fig. 1 in https://arxiv.org/pdf/1706.03762.pdf
  NOTE: The layer normalization step has been moved to the input as per latest version of T2T
  """
  def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads,
         bias_mask=None, layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0):
    """
    Parameters:
      hidden_size: Hidden size
      total_key_depth: Size of last dimension of keys. Must be divisible by num_head
      total_value_depth: Size of last dimension of values. Must be divisible by num_head
      output_depth: Size last dimension of the final output
      filter_size: Hidden size of the middle layer in FFN
      num_heads: Number of attention heads
      bias_mask: Masking tensor to prevent connections to future elements
      layer_dropout: Dropout for this layer
      attention_dropout: Dropout probability after attention (Should be non-zero only during training)
      relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
    """
    
    super(EncoderLayer, self).__init__()
    
    self.multi_head_attention = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth, 
                             hidden_size, num_heads, bias_mask, attention_dropout)
    
    self.positionwise_feed_forward = PositionwiseFeedForward(hidden_size, filter_size, hidden_size,
                                 layer_config='cc', padding = 'both', 
                                 dropout=relu_dropout)
    self.dropout = nn.Dropout(layer_dropout)
    self.layer_norm_mha = LayerNorm(hidden_size)
    self.layer_norm_ffn = LayerNorm(hidden_size)
    
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