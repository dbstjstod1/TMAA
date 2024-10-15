import torch
from torch import nn
from torchsummary import summary as summary_
from torch.nn import functional as F
from model_REDCNN import *
model = RED_CNN3().cuda()
summary_(model,(1,256,256),batch_size=4)
