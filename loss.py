'''
Author: Yin Jin
Date: 2022-03-08 20:17:04
LastEditTime: 2022-09-29 17:15:33
LastEditors: JinYin
Description: 定义loss函数
'''

import torch
from torch import nn
import numpy as np
from scipy import signal
import torch.nn.functional as F

def denoise_loss_mse(denoise, clean):      
  loss = torch.nn.MSELoss()
  return loss(denoise, clean)

def denoise_loss_l1(denoise, clean):      
  loss = torch.nn.L1Loss()
  return loss(denoise, clean)

def hinge_loss(pred_real, pred_fake=None):
  if pred_fake is not None:
      loss_real = F.relu(1 - pred_real).mean()
      loss_fake = F.relu(1 + pred_fake).mean()
      return loss_real + loss_fake
  else:
      loss = -pred_real.mean()
      return loss

