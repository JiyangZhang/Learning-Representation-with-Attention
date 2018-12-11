import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from lib.torch_utils import Dot, Gaussian_ker, GenDotM, cossim, cossim1, MaxM_fromBatch
import numpy as np
from sublayer import MultiHeadAttention, MLP, GloRepAttention, MultiHeadQAttention, GloRepQAttention, GloRepWinAttention, MultiHeadWinAttention

from lib.data_utils import list_shuffle, load
from math import floor



class Attention(nn.Module):
    def __init__(self, emb_size, query_length, doc_length, num_heads, kernel_size, filter_size, vocab_size,
                dropout, qrep_dim, hidden_size,  batch_size,
                preemb, emb_path):
        """
        Parameters:
        vocab_size: the embedding matrix param
        emb_size: the embedding matrix param
        preemb: the flag
        emb_path: the pretrained embeddding path   
        """

        super(Attention, self).__init__()
	    
        # embedding matrix
        self.emb_mod = nn.Embedding(vocab_size, emb_size)
        if preemb is True:
            emb_data = load(emb_path)
            self.emb_mod.weight = nn.Parameter(torch.from_numpy(emb_data))
        else:
            init_tensor = torch.randn(vocab_size, emb_size).normal_(0, 5e-3)
            self.emb_mod.weight = nn.Parameter(init_tensor)

        
        # multi-head-attention
        self.multi_head_attention = MultiHeadAttention(emb_size, query_length, doc_length, 
	    			            num_heads, self.emb_mod, dropout)
        self.multi_head_Qattention = MultiHeadQAttention(emb_size, query_length, doc_length, 
                                num_heads, self.emb_mod, dropout)
        self.multi_head_Winattention = MultiHeadWinAttention(emb_size, query_length, doc_length, 
                                num_heads, self.emb_mod, dropout)
        # mlp layer
        self.MLP = MLP(emb_size, hidden_size, emb_size)
        # document representation layer
        self.GloRepAttention = GloRepAttention(batch_size, hidden_size, query_length, emb_size)
        self.GloRepQAttention = GloRepQAttention(batch_size, hidden_size, query_length, emb_size)
        self.GloRepWinAttention = GloRepWinAttention(batch_size, hidden_size, query_length, emb_size)



    def generate_mask(self, q, d_pos, d_neg):
        """
        Parameters:
        generate mask for q, d_pos, d_neg seperately
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
        """
        Parameters:
        q: LongTensor (BS, qlen) Variable input
        d_pos, d_neg: LongTensor (BS, dlen) Variable input
        q_mask: non learnable Variable (BS, qLen, 1)
        d_pos_mask: non learnable Variable (BS, dLen, 1)
        """

        # term level attention
        d_pos, d_neg = self.multi_head_attention(q, d_pos, d_neg, q_mask, d_pos_mask, d_neg_mask, pe_flag=True)  
        #shape=[bs, win_number, q_len, dim]
        D_pos = self.GloRepAttention(d_pos).squeeze(2)
        D_neg = self.GloRepAttention(d_neg).squeeze(2)
        #shape=[bs, win_number, dim]
        D_pos_rep = torch.mean(D_pos, dim=1, keepdim=True) 
        D_neg_rep = torch.mean(D_neg, dim=1, keepdim=True) 
        # shape=[bs, 1, dim]

        q_att, _ = self.multi_head_Qattention(q, q, q, q_mask, q_mask, q_mask, pe_flag=False)  
        #shape=[bs, win_number, query_len, dim]
        Q = self.GloRepQAttention(q_att).squeeze(2)
        #shape=[bs, win_number, dim]
        Q_rep = torch.mean(Q, dim=1, keepdim=True) 
        #shape=[bs, 1, dim]
        
        # calculate the sim score
        Score_pos = cossim(Q_rep, D_pos_rep) # shape = [bs, 1, 1]
        Score_neg = cossim(Q_rep, D_neg_rep)

        return Score_pos, Score_neg



  