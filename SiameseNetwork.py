import torch 
import torch.nn as nn 
import torch.nn.functional as F 
 
class SiameseNetwork(nn.Module): 
    def __init__(self): 
        super(SiameseNetwork, self).__init__() 
        self.c1 = nn.Conv2d(1, 16, kernel_size=5, padding=0) 
        self.c2 = nn.Conv2d(16, 16, kernel_size=5, padding=0) 
        self.fc1 = nn.Linear(256, 64) 
        self.fc2 = nn.Linear(64, 2)  # latent output of 50 gives >98% accuracy 
 
    def forward(self, x): 
        h = F.max_pool2d(F.relu(self.c1(x)), 2) 
        h = F.max_pool2d(F.relu(self.c2(h)), 2) 
        z = h.view(h.size(0), -1) 
        h = self.fc1(z) 
        return self.fc2(h) 