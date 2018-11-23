# qfdRM
The baseline model can be run with the command:

CUDA_VISIBLE_DEVICES=0 python model_train.py --path=./config --mode=train --resume=False

After adding Positional Encoding and using the Pretrained Embedding the model is much better, having similar performance to CDSSM. If we train the model on the querry-centric documents it will outperform the CDSSM.

Reserch Plan

Basic model: using Two layer attention mechnism
Adding the CNN at the aggregation layer
Designing a hierarchy attention model
Using GCN
