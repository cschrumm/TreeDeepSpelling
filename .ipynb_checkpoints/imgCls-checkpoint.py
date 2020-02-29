import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


# taken from here: https://github.com/pytorch/examples/blob/master/mnist/main.py

def pt_info(ot):
    print("Size: ",ot.size()," Data type: ", ot.dtype," Device: ",ot.device, " Requires grad: ", ot.requires_grad)
    return

def custom_init_weights(m):
    if type(m) == nn.Linear:
        #torch.nn.init.xavier_uniform_(m.weight)
        print('normal init..')
        nn.init.normal_(m.weight)
        nn.init.normal_(m.bias)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.bias)
    
        

class ImgNet(nn.Module):
    def __init__(self,Ctg):
        super(ImgNet, self).__init__()
        '''
        cnv2 = nn.Conv2d(1,32,2,1)
        cnv3 = nn.Conv2d(32,16,2,1,padding=1)
        '''
        self.conv1 = nn.Conv2d(1, 32, 2, 1)
        self.conv2 = nn.Conv2d(32, 16, 2, 1,padding=1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(2048, 64)
        self.fc2 = nn.Linear(64, Ctg)
        
        

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        #pt_info(x)
        x = torch.flatten(x, 1)
        #pt_info(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output