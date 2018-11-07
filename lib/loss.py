import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

def hinge_loss(S_pos, S_neg, hinge_margin):
    """ calculate the hinge loss
        S_pos: pos score Variable (BS,)
        S_neg: neg score Variable (BS,)
        hinge_margin: hinge margin
        returns: batch-averaged loss value
    """
    cost = torch.mean((hinge_margin - (S_pos - S_neg)) *
                      ((hinge_margin - (S_pos - S_neg)) > 0).float())
    return cost

if __name__ == '__main__':
    S1 = torch.randn(10)
    S2 = torch.randn(10)
    print(hinge_loss(S1, S2, 1.0))
