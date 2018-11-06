import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from sublayers import MultiHeadAttention, MLP, DocRepAttention

from lib.data_utils import list_shuffle, load
from lib.torch_utils import Dot, Gaussian_ker, GenDotM, cossim, cossim1, MaxM_fromBatch
from lib.torch_utils import MaxM_1dConv
from math import floor


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

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention as per https://arxiv.org/pdf/1706.03762.pdf
    Refer Figure 2
    """
    def __init__(self, word_dim, query_length, doc_length, num_heads, #output_dim
        bias_mask=None, dropout=0.0):
        """
        Parameters:
        word_dim: The dimension of word embedding
        query_length: The number of words in the query
        doc_length: The number of words in the document
        num_heads: Number of attention heads
        bias_mask: Masking tensor to prevent connections to future elements
          dropout: Dropout probability (Should be non-zero only during training)
        """
        super(MultiHeadAttention, self).__init__()
        # Checks borrowed from 
          
        self.num_heads = num_heads  # Number of attention heads
        self.query_scale = (word_dim//num_heads)**-0.5
        self.bias_mask = bias_mask


        self.linear_trans = nn.Linear(word_dim, word_dim, bias=False)

        self.output_linear = nn.Linear(word_dim, word_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
  
    def _split_heads(self, x):
        """
        Split x such to add an extra num_heads dimension
        Input:
          x: a Tensor with shape [batch_size, seq_length, word_dim]
        Returns:
          A Tensor with shape [batch_size, num_heads, seq_length, word_dim/num_heads]
        """
        if len(x.shape) != 3:
          raise ValueError("x must have rank 3")
        shape = x.shape
        return x.view(shape[0], shape[1], self.num_heads, shape[2]//self.num_heads).permute(0, 2, 1, 3)
  
    def _merge_heads(self, x):
        """
        Merge the extra num_heads into the last dimension
        Input:
          x: a Tensor with shape [batch_size, num_heads, seq_length, word<_dim/num_heads]
        Returns:
          A Tensor with shape [batch_size, seq_length, word_dim]
        """
        if len(x.shape) != 4:
          raise ValueError("x must have rank 4")
        shape = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(shape[0], shape[2], shape[3]*self.num_heads)

    def forward(self, queries, doc):  
        # queries Shape [BS, Q_len, word_dim]
        # doc Shape [BS, D_len, word_dim]  
        # Do a linear for each component
        queries = self.linear_trans(queries) 
        doc = self.linear_trans(doc)

        # Split into multiple heads
        queries = self._split_heads(queries) # SHAPE=[batch_size, num_heads, query_length, dim/num_heads]
        doc = self._split_heads(doc)  # SHAPE=[batch_size, num_heads, doc_length, dim/num_heads]

        # Scale queries
        queries *= self.query_scale

        # Combine queries and keys
        logits = torch.matmul(queries, doc.permute(0, 1, 3, 2))  #SHAPE=[bs, num_heads, query_length, doc_length]

        # Add bias to mask future values
        if self.bias_mask is not None:
            logits += Variable(self.bias_mask[:, :, :logits.shape[-2], :logits.shape[-1]].type_as(logits.data))

        # Convert to probabilites
        weights = nn.functional.softmax(logits, dim=2)  # PROBLEM ! dim 

        # Dropout
        weights = self.dropout(weights)

        # Combine with values to get context
        contexts = torch.matmul(weights, doc)  #SHAPE=[bs, num_heads, query_length, dim/num_heads]

        # Merge heads
        contexts = self._merge_heads(contexts)   # SHAPE= [batch_size, query_length, dim]
        #contexts = torch.tanh(contexts)

        # Linear to get output
        outputs = self.output_linear(contexts) # SHAPE = [BS, query_length, word_dim]

        return outputs

class DocRepAttention(nn.Module):
    
    def __init__(self, batch_size, hidden_size, query_length, word_dim):        
        super(AttentionWordRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.word_dim = word_dim
        self.query_length = query_length
        self.u_w = nn.Parameter(torch.Tensor((batch_size, query_length, word_dim)))
        self.mlp = MLP(word_dim, hidden_size, word_dim)
        self.uw.data.uniform_(-0.1, 0.1)

    def forward(self, attention_mat):
        """
        attention_mat: The output of the multi-head attention layer (BS, Q_len, dim)
        """
        u_t = self.mlp(attention_mat) # shape=[bs, q_len, word_dim]
        logits = torch.matmul(u_t, u_w.permute(0, 2, 1))  #SHAPE=[bs, query_length, query_length]
        weights = nn.functional.softmax(logits, dim=1)  # PROBLEM ! dim SHAPE = [BS, q_l, q_l] 
        gloDoc = torch.matmul(weights, attention_mat)
        return gloDoc


class QueryRep(nn.Module):
    """MultiMatch Model"""
    def __init__(self, BS, q_len, kernel_size, filter_size,
                 hidden_size, output_dim, vocab_size, emb_size,
                 sim_type="Cos", attn_type="Cos", preemb=False, preemb_path=''):
        """
        BS: batch size
        q_len: the query length
        kernel_size: the size of kernel
        filter_size: the number of filters
        hidden_size: the hidden layer of mlp
        output_dim: the output dim of the doc representation
        
        """

        super(MultiMatch, self).__init__()
        self.BS = BS
        self.q_len = q_len
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.preemb = preemb
        self.sim_type = sim_type

        # embedding matrix
        self.emb_mod = nn.Embedding(vocab_size, emb_size)
        if preemb is True:
            emb_data = load(preemb_path)
            self.emb_mod.weight = nn.Parameter(torch.from_numpy(emb_data))
        else:
            init_tensor = torch.randn(vocab_size, emb_size).normal_(0, 5e-3)
            self.emb_mod.weight = nn.Parameter(init_tensor)
        
        # The 1-d convolution layer
        self.q_conv1 = nn.Conv1d(in_channels=emb_size, out_channels=filter_size,
                                kernel_size=kernel_size, stride=1, padding=0,
                                dilation=1, bias=True)
        # nonlinear layer
        self.tanh = nn.Tanh()
        # maxpooling layer
        self.Maxpool = nn.MaxPool1d(kernel_size=q_len)
        # mlp layer
        self.mlp = MLP(filter_size, hidden_size, output_dim)

        
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


    def forward(self, q, q_mask):
        """ apply rel
            q: LongTensor (BS, qlen) Variable input
            d: LongTensor (BS, dlen) Variable input
            q_mask: non learnable Variable (BS, qLen, 1)
            d_mask: non learnable Variable (BS, dLen, 1)
            returns R1, R2, R3: relevance of 3 level (BS,)
        """
        # mask out padding's variable embeddings
        q_emb = self.emb_mod(q) * q_mask  # (BS, qlen, emb_size)
        # shuffle the dim
        q_shuffle = q_emb.permute(0, 2, 1)  # (BS, emb_size, qlen)

        # conv1d layer
        q_conv = self.q_conv1(q_emb) # (BS, filter_size, q_len)

        # activation layer
        q_tanh = self.tanh(q_conv) # (BS, filter_size, q_len)

        # max-pooling layer
        q_pool = self.Maxpool(q_tanh) # (BS, filter_size, 1)
        
        # mlp layer 
        q_rep = self.mlp(q_pool.permute(0, 2, 1)) # (BS, 1, outputdim)

        return q_rep


