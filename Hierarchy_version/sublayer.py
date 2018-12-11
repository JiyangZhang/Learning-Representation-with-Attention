
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math


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

class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding for
    document embedding
    Args:
       dropout (float): dropout parameter
       word_dim (int): embedding size
       doc_len = the max length of doc
    Returns:
       the embedding
    """

    def __init__(self, dropout, word_dim, doc_len=1100):
        pe = Variable(torch.zeros(doc_len, word_dim).cuda())
        position = torch.arange(0, doc_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, word_dim, 2) * -(math.log(10000.0) / word_dim)).float())
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, pe_size)
        super(PositionalEncoding, self).__init__()
        self.pe = pe
        self.dropout = nn.Dropout(p=dropout)
        self.word_dim = word_dim

    def forward(self, emb):
        """ 
        emb (seq_len, BS, dim) variable
        pe (seq_len, 1, dim)
        """

        #self.pe = self.pe.repeat(1, emb.size(1), 1)
        
        emb = emb * math.sqrt(self.word_dim)
        emb = torch.add(emb, self.pe)
        emb = self.dropout(emb)
        return emb.permute(1, 0, 2).cuda()


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention as per https://arxiv.org/pdf/1706.03762.pdf
    Refer Figure 2
    """
    def __init__(self, word_dim, query_length, key_len, num_heads, emb_mod, dropout=0.0):
        """
            Parameters:
            word_dim: The dimension of word embedding
            query_length: The number of words in the query
            key_len: The number of words in the document
            num_heads: Number of attention heads
            dropout: Dropout probability (Should be non-zero only during training)
        """
        super(MultiHeadAttention, self).__init__()
        
        # check the number of heads
        if word_dim % num_heads != 0:
            raise ValueError("embedding dim (%d) must be divisible by the number of "
               "attention heads (%d)." % (word_dim, num_heads))
          
        self.num_heads = num_heads  # Number of attention heads
        self.query_scale = (word_dim//num_heads)**-0.5

        # load the embedding
        self.emb_mod = emb_mod
        # Position embedding
        self.PositionalEncoding = PositionalEncoding(dropout, word_dim, key_len)
        #self.bias_mask = bias_mask

        self.linear_trans = nn.Linear(word_dim, word_dim, bias=False)

        self.output_linear = nn.Linear(word_dim, word_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
  
    def _split_heads(self, x):
        """
        Split x such to add an extra num_heads dimension
        Input:
          x: a Tensor with shape [batch_size, win_size, seq_length, word_dim]
        Returns:
          A Tensor with shape [batch_size, win_size, num_heads, seq_length, word_dim/num_heads]
        """
        if len(x.shape) != 4:
          raise ValueError("x must have rank 4")
        shape = x.shape
        return x.view(shape[0], shape[1], self.num_heads, shape[2], shape[3]//self.num_heads)
  
    def _merge_heads(self, x):
        """
        Merge the extra num_heads into the last dimension
        Input:
          x: a Tensor with shape [bs, 100, num_heads, query_length, dim/num_heads]
        Returns:
          A Tensor with shape [bs, 100, query, dim]
        """
        if len(x.shape) != 5:
          raise ValueError("x must have rank 4")
        shape = x.shape
        return x.contiguous().view(shape[0], shape[1], shape[3], shape[4]*self.num_heads)

    def forward(self, q, d_pos, d_neg, q_mask, d_pos_mask, d_neg_mask, pe_flag):  
        # input q Shape [BS, q_len]
        # doc Shape [BS, 1100]

        # mask out padding's variable embeddings
        q = self.emb_mod(q) * q_mask  # (BS, qlen, emb_size)

        # get the embedding
        d_pos = self.emb_mod(d_pos)
        d_neg = self.emb_mod(d_neg) # (bs, qlen, emb_size)
        # adding the pe
        
        if pe_flag == True:
            d_pos = self.PositionalEncoding(d_pos.permute(1, 0, 2))
            d_neg = self.PositionalEncoding(d_neg.permute(1, 0, 2))
        
        # add mask
        d_pos = d_pos  * d_pos_mask  # (BS, dlen, emb_size)
        d_neg = d_neg  * d_neg_mask  # (BS, 1100, emb_size)
        
        # resize
        d_pos = d_pos.view(d_pos.size(0), d_pos.size(1)//11, 11, d_pos.size(2))
        d_neg = d_neg.view(d_neg.size(0), d_neg.size(1)//11, 11, d_neg.size(2))
        q = q.unsqueeze(1)
        q = q.expand(q.size(0), d_neg.size(1), q.size(2), q.size(3))
        
        # Do a linear for each component
        queries = self.linear_trans(q)  # shape [BS, 100, q_len, emb_size]
        d_pos = self.linear_trans(d_pos) # shape [BS, 100, win_size, emb_size]
        d_neg = self.linear_trans(d_neg)

        # Split into multiple heads
        queries = self._split_heads(queries) #[BS, 100, num_heads, query_length, dim/num_heads]
        d_pos = self._split_heads(d_pos)  # [BS, 100, num_heads, doc_win, dim/num_heads]
        d_neg = self._split_heads(d_neg)
        
        # Scale queries
        queries *= self.query_scale

        # calculate two docs
        d_list = [d_pos, d_neg]
        d_output = []
        # Combine queries and keys

        for doc in d_list:
            # calculate the probability
            logits = torch.matmul(queries, doc.permute(0, 1, 2, 4, 3))  
            #SHAPE=[bs, 100, num_heads, query_length, win_size]

            # Convert to probabilites
            weights = nn.functional.softmax(logits, dim=-1)
            #SHAPE=[bs, 100, num_heads, query_length, win_size]

            # Dropout
            weights = self.dropout(weights)

            # Combine with values to get context
            contexts = torch.matmul(weights, doc)  #SHAPE=[bs, 100, num_heads, query_length, dim/num_heads]

            # Merge heads
            contexts = self._merge_heads(contexts) # SHAPE= [batch_size, 100, query_length, dim]
            # Linear to get output
            outputs = self.output_linear(contexts) # SHAPE = [batch_size, 100, query_length, dim]
            d_output.append(outputs)

        return d_output[0], d_output[1]

class MultiHeadWinAttention(nn.Module):
    """
    Multi-head attention as per https://arxiv.org/pdf/1706.03762.pdf
    Refer Figure 2
    """
    def __init__(self, word_dim, query_length, key_len, num_heads, emb_mod, dropout=0.0):
        """
            Parameters:
            word_dim: The dimension of word embedding
            query_length: The number of words in the query
            key_len: The number of words in the document
            num_heads: Number of attention heads
            dropout: Dropout probability (Should be non-zero only during training)
        """
        super(MultiHeadWinAttention, self).__init__()
        
        # check the number of heads
        if word_dim % num_heads != 0:
            raise ValueError("embedding dim (%d) must be divisible by the number of "
               "attention heads (%d)." % (word_dim, num_heads))
          
        self.num_heads = num_heads  # Number of attention heads
        self.query_scale = (word_dim//num_heads)**-0.5
        self.emb_mod = emb_mod

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

    def forward(self, q, d_pos, d_neg, q_mask, d_pos_mask, d_neg_mask, pe_flag):  
        # queries Shape [BS, Q_len, word_dim]
        # doc Shape [BS, Win_size,1, word_dim]
        # q shape [BS, Win_size, 1, word_dim]
        # mask out padding's variable embeddings
        if len(q.shape) == 4:
            q = q.squeeze(2)  # (BS, 5, emb_size)
        
        d_pos = d_pos.squeeze(2)  #[BS, 550, word_dim]
        d_neg = d_neg.squeeze(2)  #[BS, Win_number,word_dim]
        
        # Do a linear for each component
        queries = self.linear_trans(q) 
        d_pos = self.linear_trans(d_pos)
        d_neg = self.linear_trans(d_neg)

        # Split into multiple heads
        queries = self._split_heads(queries) # SHAPE=[batch_size, num_heads, query_length, dim/num_heads]
        d_pos = self._split_heads(d_pos)  # SHAPE=[batch_size, num_heads, doc_length, dim/num_heads]
        d_neg = self._split_heads(d_neg)
        
        # Scale queries
        queries *= self.query_scale

        # calculate two docs
        d_list = [d_pos, d_neg]
        d_output = []
        # Combine queries and keys

        for doc in d_list:
            # calculate the probability
            logits = torch.matmul(queries, doc.permute(0, 1, 3, 2))  
            #SHAPE=[bs, num_heads, query_length, key_len]

            # Convert to probabilites
            weights = nn.functional.softmax(logits, dim=-1)  # PROBLEM ! dim 

            # Dropout
            weights = self.dropout(weights)

            # Combine with values to get context
            contexts = torch.matmul(weights, doc)  #SHAPE=[bs, num_heads, query_length, dim/num_heads]

            # Merge heads
            contexts = self._merge_heads(contexts)   # SHAPE= [batch_size, query_length, dim]

            # Linear to get output
            outputs = self.output_linear(contexts) # SHAPE = [BS, query_length, word_dim]
            d_output.append(outputs)

        return d_output[0], d_output[1]


class MultiHeadQAttention(nn.Module):
    """
    Multi-head attention as per https://arxiv.org/pdf/1706.03762.pdf
    Refer Figure 2
    """
    def __init__(self, word_dim, query_length, key_len, num_heads, emb_mod, dropout=0.0):
        """
            Parameters:
            word_dim: The dimension of word embedding
            query_length: The number of words in the query
            key_len: The number of words in the document
            num_heads: Number of attention heads
            dropout: Dropout probability (Should be non-zero only during training)
        """
        super(MultiHeadQAttention, self).__init__()
        
        # check the number of heads
        if word_dim % num_heads != 0:
            raise ValueError("embedding dim (%d) must be divisible by the number of "
               "attention heads (%d)." % (word_dim, num_heads))
          
        self.num_heads = num_heads  # Number of attention heads
        self.query_scale = (word_dim//num_heads)**-0.5

        # load the embedding
        self.emb_mod = emb_mod
        # Position embedding
        self.PositionalEncoding = PositionalEncoding(dropout, word_dim, key_len)
        #self.bias_mask = bias_mask

        self.linear_trans = nn.Linear(word_dim, word_dim, bias=False)

        self.output_linear = nn.Linear(word_dim, word_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
  
    def _split_heads(self, x):
        """
        Split x such to add an extra num_heads dimension
        Input:
          x: a Tensor with shape [batch_size, win_size, seq_length, word_dim]
        Returns:
          A Tensor with shape [batch_size, win_size, num_heads, seq_length, word_dim/num_heads]
        """
        if len(x.shape) != 4:
          raise ValueError("x must have rank 4")
        shape = x.shape
        return x.view(shape[0], shape[1], self.num_heads, shape[2], shape[3]//self.num_heads)
  
    def _merge_heads(self, x):
        """
        Merge the extra num_heads into the last dimension
        Input:
          x: a Tensor with shape [bs, 100, num_heads, query_length, dim/num_heads]
        Returns:
          A Tensor with shape [bs, 100, query, dim]
        """
        if len(x.shape) != 5:
          raise ValueError("x must have rank 4")
        shape = x.shape
        return x.contiguous().view(shape[0], shape[1], shape[3], shape[4]*self.num_heads)

    def forward(self, q, d_pos, d_neg, q_mask, d_pos_mask, d_neg_mask, pe_flag):  
        # input q Shape [BS, q_len]
        # doc Shape [BS, 1100]

        # mask out padding's variable embeddings
        q = self.emb_mod(q) * q_mask  # (BS, qlen, emb_size)

        # get the embedding
        d_pos = self.emb_mod(d_pos)
        d_neg = self.emb_mod(d_neg) # (bs, qlen, emb_size)
        # adding the pe
        
        if pe_flag == True:
            d_pos = self.PositionalEncoding(d_pos.permute(1, 0, 2))
            d_neg = self.PositionalEncoding(d_neg.permute(1, 0, 2))
        
        # add mask
        d_pos = d_pos  * d_pos_mask  # (BS, qlen, emb_size)
        d_neg = d_neg  * d_neg_mask  # (BS, qlen, emb_size)
        
        # resize
        d_pos = d_pos.view(d_pos.size(0), d_pos.size(1)//3, 3, d_pos.size(2))
        d_neg = d_neg.view(d_neg.size(0), d_neg.size(1)//3, 3, d_neg.size(2))
        q = q.unsqueeze(1)
        q = q.expand(q.size(0), d_neg.size(1), q.size(2), q.size(3))
        
        # Do a linear for each component
        queries = self.linear_trans(q)  # shape [BS, 100, q_len, emb_size]
        d_pos = self.linear_trans(d_pos) # shape [BS, 100, win_size, emb_size]
        d_neg = self.linear_trans(d_neg)

        # Split into multiple heads
        queries = self._split_heads(queries) #[BS, 100, num_heads, query_length, dim/num_heads]
        d_pos = self._split_heads(d_pos)  # [BS, 100, num_heads, doc_win, dim/num_heads]
        d_neg = self._split_heads(d_neg)
        
        # Scale queries
        queries *= self.query_scale

        # calculate two docs
        d_list = [d_pos, d_neg]
        d_output = []
        # Combine queries and keys

        for doc in d_list:
            # calculate the probability
            logits = torch.matmul(queries, doc.permute(0, 1, 2, 4, 3))  
            #SHAPE=[bs, 100, num_heads, query_length, win_size]

            # Convert to probabilites
            weights = nn.functional.softmax(logits, dim=-1)
            #SHAPE=[bs, 100, num_heads, query_length, win_size]

            # Dropout
            weights = self.dropout(weights)

            # Combine with values to get context
            contexts = torch.matmul(weights, doc)  #SHAPE=[bs, 100, num_heads, query_length, dim/num_heads]

            # Merge heads
            contexts = self._merge_heads(contexts) # SHAPE= [batch_size, 100, query_length, dim]
            # Linear to get output
            outputs = self.output_linear(contexts) # SHAPE = [batch_size, 100, query_length, dim]
            d_output.append(outputs)

        return d_output[0], d_output[1]


class GloRepQAttention(nn.Module):
    
    def __init__(self, batch_size, hidden_size, query_length, word_dim):
        """
        Parameters:
        batch_size: batch size
        hidden_size: the size of the hidden layer
        query_length: the length of the query
        word_dim: dim of the embedding
        """
        super(GloRepQAttention, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.word_dim = word_dim
        self.query_length = query_length

        self.u_w = nn.Parameter(torch.Tensor(1, 1, 1, word_dim))
        self.u_w.data.uniform_(-0.1, 0.1)
        self.mlp = MLP(word_dim, hidden_size, word_dim)
        

    def forward(self, doc_rep):
        """
        doc_rep: The output of the multi-head attention layer
        shape = (BS, 100, Q_len, dim)
        """
        # first transform layer
        u_t = self.mlp(doc_rep) # shape=[bs, 100, q_len, dim]

        # calculate the logits and weights
        #print(u_t.size())
        logits = torch.matmul(u_t, self.u_w.permute(0, 1, 3, 2))  #SHAPE=[bs, 100, query_length, query_length]
        weights = nn.functional.softmax(logits, dim=-1)  # SHAPE = [BS, 100, q_l, 1] 
        # get the summation
        gloDoc = torch.matmul(weights.permute(0, 1, 3, 2), doc_rep)
        # shape = [bs, 100, 1, dim]

        return gloDoc


class GloRepAttention(nn.Module):
    
    def __init__(self, batch_size, hidden_size, query_length, word_dim):
        """
        Parameters:
        batch_size: batch size
        hidden_size: the size of the hidden layer
        query_length: the length of the query
        word_dim: dim of the embedding
        """
        super(GloRepAttention, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.word_dim = word_dim
        self.query_length = query_length

        self.u_w = nn.Parameter(torch.Tensor(1, 1, 1, word_dim))
        self.u_w.data.uniform_(-0.1, 0.1)
        self.mlp = MLP(word_dim, hidden_size, word_dim)
        

    def forward(self, doc_rep):
        """
        doc_rep: The output of the multi-head attention layer
        shape = (BS, 100, Q_len, dim)
        """
        # first transform layer
        u_t = self.mlp(doc_rep) # shape=[bs, 100, q_len, dim]

        # calculate the logits and weights
        #print(u_t.size())
        logits = torch.matmul(u_t, self.u_w.permute(0, 1, 3, 2))  #SHAPE=[bs, 100, query_length, query_length]
        weights = nn.functional.softmax(logits, dim=-1)  # SHAPE = [BS, 100, q_l, 1] 
        # get the summation
        gloDoc = torch.matmul(weights.permute(0, 1, 3, 2), doc_rep)
        # shape = [bs, 100, 1, dim]

        return gloDoc


class GloRepWinAttention(nn.Module):
    
    def __init__(self, batch_size, hidden_size, query_length, word_dim):
        """
        Parameters:
        batch_size: batch size
        hidden_size: the size of the hidden layer
        query_length: the length of the query
        word_dim: dim of the embedding
        """
        super(GloRepWinAttention, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.word_dim = word_dim
        self.query_length = query_length

        self.u_w = nn.Parameter(torch.Tensor(1, 1, word_dim))
        self.u_w.data.uniform_(-0.1, 0.1)
        self.mlp = MLP(word_dim, hidden_size, word_dim)
        

    def forward(self, doc_rep):
        """
        doc_rep: The output of the multi-head attention layer
        shape = (BS, q_len, dim)
        """
        # first transform layer
        u_t = self.mlp(doc_rep) # shape=[bs, q_len, dim]

        # calculate the logits and weights
        #print(u_t.size())
        logits = torch.matmul(u_t, self.u_w.permute(0, 2, 1))  #SHAPE=[bs, query_length, query_length]
        weights = nn.functional.softmax(logits, dim=-1)  # PROBLEM ! dim SHAPE = [BS, 100, 1] 
        # get the summation
        gloDoc = torch.matmul(weights.permute(0, 2, 1), doc_rep)
        # shape = [bs, 1, dim]

        return gloDoc
