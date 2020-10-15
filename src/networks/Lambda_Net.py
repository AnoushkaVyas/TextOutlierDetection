import torch
import torch.nn as nn
import torch.nn.functional as F

class LambdaNet(nn.Module):

    def __init__(self, n_attention_heads,hidden_size):
        super().__init__()

        self.hidden_size=hidden_size
        self.n_attention_heads=n_attention_heads
        
        self.fc=nn.Linear(self.hidden_size*self.n_attention_heads,1)
        self.reset_param(self.fc.weight)

    def reset_param(self,t):
        nn.init.kaiming_normal_(t.data)

    def forward(self, x):

        out = F.relu(self.fc(x))
        print(out)

        return out
