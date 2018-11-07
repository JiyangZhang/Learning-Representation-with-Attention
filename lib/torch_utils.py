import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

def Gaussian_ker(X, Y, k):
    """ calculate the Gaussian kernel similarity between X and Y exp(-k|X-Y|)
        X: tensor (BS, x_len, 1, hidden_size)
        Y: tensor (BS, 1,  y_len, hidden_size)
        k; scalar to control the kernel shape
        returns: (BS, x_len, y_len)
    """
    S = X.unsqueeze(2) - Y.unsqueeze(1)  # (BS, xlen, ylen, hid_size)
    S = torch.sum(S ** 2, dim=3)  # (BS, xlen, ylen)
    S = torch.exp(-k * S)  # (BS, xlen, ylen)
    return S

def Dot(X, Y):
    """ calculate the dot similarity between X and Y: dot(X, Y)
        X: tensor (BS, x_len, hidden_size)
        Y: tensor (BS, y_len, hidden_size)
        returns: (BS, x_len, y_len)
    """
    S = torch.bmm(X, Y.transpose(1, 2))  # (BS, xlen, ylen)
    return S

def cossim(X, Y):
    """ calculate the cos similarity between X and Y by dotting normalized X, Y
        X: tensor (BS, x_len, hidden_size)
        Y: tensor (BS, y_len, hidden_size)
        returns: (BS, x_len, y_len)
    """
    X = F.normalize(X, p=2, dim=2, eps=1e-12)
    Y = F.normalize(Y, p=2, dim=2, eps=1e-12)
    S = torch.bmm(X, Y.transpose(1,2))
    return S

def cossim1(X, Y):
    """ calculate the cos similarity between X and Y: cos(X, Y)
        X: tensor (BS, x_len, hidden_size)
        Y: tensor (BS, y_len, hidden_size)
        returns: (BS, x_len, y_len)
    """
    X_norm = torch.sqrt(torch.sum(X ** 2, dim=2)).unsqueeze(2) + 1e-12  # (BS, x_len, 1)
    Y_norm = torch.sqrt(torch.sum(Y ** 2, dim=2)).unsqueeze(1) + 1e-12 # (BS, 1, y_len)
    S = torch.bmm(X, Y.transpose(1,2)) / (X_norm * Y_norm + 1e-5)
    return S

class GenDot(nn.Module):
    """ calculate the generalized dot similarity between X and Y: X M Y.t
        Using nn.Bilinear
        X: tensor (BS, x_len, hidden_size)
        Y: tensor (BS, y_len, hidden_size)
        returns: (BS, x_len, y_len)
    """
    def __init__(self, BS, X_len, Y_len, hidden_size):
        super(GenDot, self).__init__()
        self.hidden_size = hidden_size
        self.BS = BS
        self.X_len = X_len
        self.Y_len = Y_len
        self.bilinear_mod = nn.Bilinear(hidden_size, hidden_size, 1, bias=False)

    def forward(self, X, Y):
        X = X.unsqueeze(2).expand(-1, -1, self.Y_len, -1).contiguous()  # (BS, xlen, y_len(copy), hs)
        Y = Y.unsqueeze(1).expand(-1, self.X_len, -1, -1).contiguous()  # (BS, x_len(copy), y_len, hs)
        X = X.view(self.BS * self.X_len * self.Y_len, -1)  # (BS*xlen*ylen, hs)
        Y = Y.view(self.BS * self.X_len * self.Y_len, -1)  # (BS*xlen*ylen, hs)
        S = self.bilinear_mod(X, Y)  # (BS*xlen*ylen, 1)
        S = S.squeeze(1).view(self.BS, self.X_len, self.Y_len)  # (BS, xlen. ylen)
        return S

class GenDotM(nn.Module):
    """ calculate the generalized dot similarity between X and Y: X M Y.t
        Using self-defined X.t M Y
        X: tensor (BS, x_len, hidden_size)
        Y: tensor (BS, y_len, hidden_size)
        returns: (BS, x_len, y_len)
    """
    def __init__(self, hidden_size):
        super(GenDotM, self).__init__()
        self.hidden_size = hidden_size
        self.M = nn.Parameter(torch.randn(hidden_size, hidden_size).
                              normal_(0, 0.01))

    def forward(self, X, Y):
        S = torch.bmm(torch.bmm(X, self.M.unsqueeze(0).expand(X.size(0), -1, -1)),
                      Y.transpose(1, 2))
        # (BS, xlen, hs).(BS, hs, hs)=(BS, xlen, hs)
        # (BS, xlen, hs). (BS, hs, ylen) = (BS xlen, ylen)
        return S

def MaxM_fromBatch(X):
    """ calculate max M for a batch of interact grid (BS, q_len, d_len)
    returns a vector M (BS, ) for the batch including each instance's M
    """
    # for each query term (row), find the max interact intensity across all doc terms
    M = torch.max(X, dim=2)  # (BS, q_len)
    # calcuate the sum of the Ms to represent the M of the whole query
    M = torch.sum(M, dim=1)  # (BS,)
    return M

def MaxM_fromBatchTopk(X, k):
    """ calculate max M for a batch of interact grid (BS, q_len, d_len)
    by considering the topk strongest interactions
    returns a vector M (BS, ) for the batch including each instance's M
    """
    # for each query term (row), find the topk interact intensity across all doc terms
    M, _ = torch.topk(X, k, dim=2)  # (BS, q_len, k)
    M = torch.mean(M, dim=2)  # (BS, q_len)
    M = torch.sum(M, dim=1)   # (BS,)
    return M

def MaxM_from4D(X):
    """ calculate max M for a batch and feature maps
     of interact grid (BS, q_len, d_len, nfeatmaps)
    returns a vector M (BS, ) for the batch including each instance's
    """
    M = torch.max(X, dim=2)  # (BS, qlen, nfeatmaps)
    M = torch.sum(M, dim=1)  #(BS, nfeatmaps)
    M = torch.mean(M, dim=1)  #(BS, )
    return M

def MaxM_from4DTopk(X, k):
    """ calculate max M for a batch and feature maps
     of interact grid (BS, q_len, d_len, nfeatmaps)
     by considering the topk strongest interactions
    returns a vector M (BS, ) for the batch including each instance's
    """
    M, _ = torch.topk(X, k, dim=2)  # (BS, qlen, k, nfeatmaps)
    M = torch.mean(M, dim=2)  # (BS, qlen, nfeatmaps)
    M = torch.sum(M, dim=1)  #(BS, nfeatmaps)
    M = torch.mean(M, dim=1)  #(BS, )
    return M

def masked_softmax(x, axis=-1, mask=None):
    mask = mask.type(torch.FloatTensor)  # cast mask to float
    mask = Variable(mask, requires_grad=False)  # wrap mask tensor into a Variable to allow mul, Variable * Variable
    if mask is not None:
        x = (mask * x) + (1 - mask) * (-10)
    x = torch.clamp(x, -10, 10)
    max_tensor, _ = torch.max(x, dim=axis, keepdim=True)
    e_x = torch.exp(x - max_tensor)
    if mask is not None:
        e_x = e_x * mask
    softmax = e_x / (torch.sum(e_x, dim=axis, keepdim=True) + 1e-6)
    return softmax

def np_softmax(x, axis=-1):
    # stable softmax for np array
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    softmax = e_x / (np.sum(e_x, axis=axis, keepdims=True) + 1e-9)
    return softmax

def non_neg_normalize(x):
    """ input: a np array vector
    output: (x - x_min)/(x_max - x_min)
    """
    x_min = np.min(x)
    x_max = np.max(x)
    return (x - x_min) / (x_max - x_min + 1e-6)

def non_neg_normalize_2Darray(x):
    """input: np 2D array
    output column-wise (x - x_min)/(x_max - x_min)
    """
    x_min = np.min(x, axis=0, keepdims=True)
    x_max = np.max(x, axis=0, keepdims=True)
    return (x - x_min) / (x_max - x_min + 1e-6)

if __name__ == '__main__':
    x = torch.FloatTensor([[[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]]])
    y = torch.FloatTensor([[[0.1, 0.2, 0.3], [0.3, 0.2, 0.0]]])
    x = torch.randn(64, 15, 300)
    y = torch.randn(64, 1000, 300)
    print(Gaussian_ker(x, y, 1).size())

    '''
    x = Variable(x, requires_grad=False)
    y = Variable(y, requires_grad=False)
    gendotM = GenDotM(3)
    print(gendotM(x, y))
    '''
