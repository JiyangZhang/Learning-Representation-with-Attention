import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np



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
    def __init__(self, word_dim, query_length, doc_length, num_heads, emb_mod, dropout=0.0):
        """
            Parameters:
            word_dim: The dimension of word embedding
            query_length: The number of words in the query
            doc_length: The number of words in the document
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

        #self.bias_mask = bias_mask

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

    def forward(self, q, d_pos, d_neg, q_mask, d_pos_mask, d_neg_mask):  
        # queries Shape [BS, Q_len, word_dim]
        # doc Shape [BS, D_len, word_dim]

        # mask out padding's variable embeddings
        q = self.emb_mod(q) * q_mask  # (BS, qlen, emb_size)
        d_pos = self.emb_mod(d_pos)  * d_pos_mask  # (BS, dlen, emb_size)
        d_neg = self.emb_mod(d_neg)  * d_neg_mask  # (BS, dlen, emb_size)

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
            #SHAPE=[bs, num_heads, query_length, doc_length]

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

class DocRepAttention(nn.Module):
    
    def __init__(self, batch_size, hidden_size, query_length, word_dim):
        """
        Parameters:
        batch_size: batch size
        hidden_size: the size of the hidden layer
        query_length: the length of the query
        word_dim: dim of the embedding
        """
        super(DocRepAttention, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.word_dim = word_dim
        self.query_length = query_length

        self.u_w = nn.Parameter(torch.Tensor((batch_size, 1, word_dim)))
        self.u_w.data.uniform_(-0.1, 0.1)
        self.mlp = MLP(word_dim, hidden_size, word_dim)
        

    def forward(self, doc_rep):
        """
        doc_rep: The output of the multi-head attention layer
        shape = (BS, Q_len, dim)
        """
        # first transform layer
        u_t = self.mlp(doc_rep) # shape=[bs, q_len, dim]

        # calculate the logits and weights
        logits = torch.matmul(u_t, self.u_w.permute(0, 2, 1))  #SHAPE=[bs, query_length, query_length]
        weights = nn.functional.softmax(logits, dim=-1)  # PROBLEM ! dim SHAPE = [BS, q_l, q_l] 

        # get the summation
        gloDoc = torch.matmul(weights.permute(0, 2, 1), doc_rep)
        # shape = [bs, 1, dim]

        return gloDoc


class QueryRep(nn.Module):
    
    def __init__(self, BS, q_len, kernel_size, filter_size, hidden_size, output_dim, emb_size,
                emb_mod):
        """
        Parameters:
        BS: batch size
        q_len: the query length
        kernel_size: the size of kernel
        filter_size: the number of filters
        hidden_size: the hidden layer of mlp
        output_dim: the output dim of the doc representation
        vocab_size: the size of the embedding matrix
        emb_size: the dim of the word dim
        preemb: the flag whether to train the embeddign
        """

        super(QueryRep, self).__init__()
        self.BS = BS
        self.q_len = q_len
        self.emb_mod = emb_mod
        self.hidden_size = hidden_size
        #self.preemb = preemb
        #elf.sim_type = sim_type

        
        
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


    def forward(self, q, q_mask):
        """ 
            Parameters:
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


