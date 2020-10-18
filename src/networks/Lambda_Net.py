import torch
import torch.nn as nn
import torch.nn.functional as F

class LambdaNet(nn.Module):

    def __init__(self, n_attention_heads,hidden_size):
        super().__init__()

        self.hidden_size=hidden_size
        self.n_attention_heads=n_attention_heads
        
        self.fc1=nn.Linear(self.hidden_size*self.n_attention_heads,100)
        self.reset_param(self.fc1.weight)
        self.fc2=nn.Linear(100,1)
        self.reset_param(self.fc2.weight)

    def reset_param(self,t):
        nn.init.kaiming_normal_(t.data)

    def forward(self, x):

        out = F.relu(self.fc2(F.dropout(F.elu(self.fc1(x)),p=0.2)))

        return out
