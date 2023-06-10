import argparse
from pyexpat import model
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
import os
import math
from torch import Tensor

from einops import rearrange
from einops.layers.torch import Rearrange
from scipy import io # to transform .npy into .mat
       
class CTNet(nn.Module):
    def __init__(self, data_num=512, drop_rate=0):
        super(CTNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(1, 32, 3, 1, 1), nn.BatchNorm1d(32), nn.LeakyReLU(0.2), 
            nn.Conv1d(32, 32, 3, 1, 1), nn.BatchNorm1d(32), nn.LeakyReLU(0.2))
        self.pool1 = nn.AvgPool1d(2, stride=2)
        
        # block2
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.LeakyReLU(0.2),
            nn.Conv1d(64, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.LeakyReLU(0.2), nn.Dropout(drop_rate))
        self.s2 = nn.Sequential(
            TransformerEncoderBlock(emb_size=32),
            nn.Conv1d(32, 64, 3, 2, 1), nn.BatchNorm1d(64), nn.LeakyReLU(0.2), nn.Dropout(drop_rate))
        self.pool2 = nn.AvgPool1d(2,stride=2)
        self.ffm2 = nn.Sequential(nn.Conv1d(128, 64, 3, 1, 1), nn.BatchNorm1d(64))
        
        # block3
        self.block3 = nn.Sequential(
            nn.Conv1d(64, 128, 3, 1, 1), nn.BatchNorm1d(128), nn.LeakyReLU(0.2), 
            nn.Conv1d(128, 128, 3, 1, 1), nn.BatchNorm1d(128), nn.LeakyReLU(0.2), nn.Dropout(drop_rate))
        self.s3 = nn.Sequential(
            TransformerEncoderBlock(emb_size=64),
            nn.Conv1d(64, 128, 3, 2, 1), nn.BatchNorm1d(128), nn.LeakyReLU(0.2), nn.Dropout(drop_rate))
        self.pool3 = nn.AvgPool1d(2,stride=2, ceil_mode=True)
        self.ffm3 = nn.Sequential(nn.Conv1d(256, 128, 3, 1, 1), nn.BatchNorm1d(128))
        
        # block4
        self.block4 = nn.Sequential(
            nn.Conv1d(128, 256, 3, 1, 1), nn.BatchNorm1d(256), nn.LeakyReLU(0.2), 
            nn.Conv1d(256, 256, 3, 1, 1), nn.BatchNorm1d(256), nn.LeakyReLU(0.2), nn.Dropout(drop_rate))
        self.s4 = nn.Sequential(
            TransformerEncoderBlock(emb_size=128),
            nn.Conv1d(128, 256, 3, 2, 1), nn.BatchNorm1d(256), nn.LeakyReLU(0.2), nn.Dropout(drop_rate))
        self.pool4 = nn.AvgPool1d(2,stride=2,ceil_mode=True)
        self.ffm4 = nn.Sequential(nn.Conv1d(512, 256, 3, 1, 1), nn.BatchNorm1d(256))
        
        # block5
        self.block5 = nn.Sequential(
            nn.Conv1d(256, 512, 3, 1, 1), nn.BatchNorm1d(512), nn.LeakyReLU(0.2), 
            nn.Conv1d(512, 512, 3, 1, 1), nn.BatchNorm1d(512), nn.LeakyReLU(0.2), nn.Dropout(drop_rate))
        self.s5 = nn.Sequential(
            TransformerEncoderBlock(emb_size=256),
            nn.Conv1d(256, 512, 3, 2, 1), nn.BatchNorm1d(512), nn.LeakyReLU(0.2), nn.Dropout(drop_rate))
        self.pool5 = nn.AvgPool1d(2,stride=2, ceil_mode=True)
        self.ffm5 = nn.Sequential(nn.Conv1d(1024, 512, 3, 1, 1), nn.BatchNorm1d(512))
        
        # block6
        self.block6 = nn.Sequential(
            nn.Conv1d(512, 1024, 3, 1, 1), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2), 
            nn.Conv1d(1024, 1024, 3, 1, 1), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2), nn.Dropout(drop_rate))
        self.s6 = nn.Sequential(
            TransformerEncoderBlock(emb_size=512),
            nn.Conv1d(512, 1024, 3, 2, 1), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2), nn.Dropout(drop_rate))
        self.pool6 = nn.AvgPool1d(2,stride=2, ceil_mode=True)
        self.ffm6 = nn.Sequential(nn.Conv1d(2048, 1024, 3, 1, 1), nn.BatchNorm1d(1024))
        
        # block7
        self.block7 = nn.Sequential(
            nn.Conv1d(1024, 1024, 3, 1, 1), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2),
            nn.Conv1d(1024, 1024, 3, 1, 1), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2), nn.Dropout(drop_rate))
        
        self.linear = nn.Linear(16 * data_num, data_num) 
    
    def forward(self, x):
        x = self.pool1(self.block1(x))
        x = self.ffm2(torch.cat((self.pool2(self.block2(x)), self.s2(x)), dim=1))
        x = self.ffm3(torch.cat((self.pool3(self.block3(x)), self.s3(x)), dim=1))
        x = self.ffm4(torch.cat((self.pool4(self.block4(x)), self.s4(x)), dim=1))
        x = self.ffm5(torch.cat((self.pool5(self.block5(x)), self.s5(x)), dim=1))
        x = self.ffm6(torch.cat((self.pool6(self.block6(x)), self.s6(x)), dim=1))
        x = self.block7(x).reshape(x.shape[0], -1)
        
        return self.linear(x)

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
    
    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * inputs.sigmoid()
    
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            Swish(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b t (h d) -> b h t d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b t (h d) -> b h d t", h=self.num_heads)
        values = rearrange(self.values(x), "b t (h d) -> b h t d", h=self.num_heads)
        energy = torch.matmul(queries, keys)

        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h t d -> b t (h d)")
        out = self.projection(out)
        return out
       
class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=8,
                 drop_p=0.1,
                 forward_expansion=1,
                 forward_drop_p=0.1):
        super().__init__(  
            nn.Sequential(
                Rearrange('n (h) (w) -> n (w) (h)'),
            ),
                     
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p),
            )),
            nn.Sequential(
                Rearrange('n (w) (h) -> n (h) (w)'),
            )    
        )
          
if __name__ == '__main__':
    x1 = torch.rand(128, 1, 512)
    model = CTNet(data_num=512)
    output = model(x1)
    
    print('output is:', output.shape)