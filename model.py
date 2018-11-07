import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from lib.torch_utils import Dot, Gaussian_ker, GenDotM, cossim, cossim1, MaxM_fromBatch
import numpy as np
from sublayers import MultiHeadAttention, MLP, DocRepAttention, QueryRep


class Attention(nn.Module):
    def __init__(self, word_dim, query_length, doc_length, num_heads, kernel_size, filter_size, vocab_size
                bias_mask=None, dropout=0.0, qrep_dim, hidden_size,  batch_size, preemb, emb_path):
	    
        super(EncoderLayer, self).__init__()
	    
        # multi-head-attention
	    self.multi_head_attention = MultiHeadAttention(word_dim, query_length, doc_length, 
	    			num_heads, bias_mask, dropout)
        # mlp layer
	    self.MLP = MLP(word_dim, hidden_size, word_dim)
	    # document representation layer
	    self.DocRepAttention = DocRepAttention(batch_size, hidden_size, query_length, word_dim)
        # query representation layer
        self.QueryRep = QueryRep(batch_size, query_length, kernel_size, filter_size,
                    hidden_size, qrep_dim, vocab_size, word_dim, preemb, preemb_path)

    def generate_mask(self, q, d_pos, d_neg):
        """ generate mask for q, d_pos, d_neg seperately
            and pack them into Variable
            q: LongTensor Variable (BS, qlen)
            d_pos: LongTensor Variable (BS, dlen)
            d_neg: LongTensor Variable (BS, d_len)
            returns: q_mask, d_pos_mask, d_neg_mask
        """
        q_mask = torch.ne(q.data, 0).unsqueeze(2).float() # (BS, qlen, 1)
        q_mask = Variable(q_mask, requires_grad=False)
        d_pos_mask = torch.ne(d_pos.data, 0).unsqueeze(2).float()  # (BS, dlen, 1)
        d_pos_mask = Variable(d_pos_mask, requires_grad=False)
        d_neg_mask = torch.ne(d_neg.data, 0).unsqueeze(2).float()  # (BSm dlen, 1)
        d_neg_mask = Variable(d_neg_mask, requires_grad=False)
        return q_mask, d_pos_mask, d_neg_mask


    def forward(self, q, d_pos, d_neg, q_mask, d_pos_mask, d_neg_mask):
        # Multi-head attention
        y = self.multi_head_attention(queries, doc)  #shape=[bs, query_len, dim]

        # Document self-attention representation
        Drep = self.DocRepAttention(y)  # shape=[bs, 1, dim]

        # Query global representation
        Qrep = self.QueryRep(queries, q_mask) # shape= [bs, 1, dim]

        # calculate the sim score
        Score = cossim(Qrep, Drep) # shape = [bs, 1, 1]
        
        return Score_pos, Score_neg



