import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet
from networks.self_attention import SelfAttention


class CVDDNet(BaseNet):

    def __init__(self, pretrained_model, attention_size=100, n_attention_heads=1,clusters=4):
        super().__init__()

        # Load pretrained model (which provides a hidden representation per word, e.g. word vector or language model)
        self.pretrained_model = pretrained_model
        self.hidden_size = pretrained_model.embedding_size

        # Set self-attention module
        self.attention_size = attention_size
        self.n_attention_heads = n_attention_heads
        self.self_attention = SelfAttention(hidden_size=self.hidden_size,
                                            attention_size=attention_size,
                                            n_attention_heads=n_attention_heads)

        #clusters
        self.clusters=clusters

        #MLP Layer
        self.fc1=nn.Linear(self.hidden_size*self.n_attention_heads,100)
        self.reset_param(self.fc1.weight)
        self.fc2=nn.Linear(100,self.clusters)
        self.reset_param(self.fc2.weight)

    def reset_param(self,t):
        nn.init.kaiming_normal_(t.data)


    def forward(self, x):
        # x.shape = (sentence_length, batch_size)

        hidden = F.dropout(self.pretrained_model(x),p=0.2)
        # hidden.shape = (sentence_length, batch_size, hidden_size)

        M, A = self.self_attention(hidden)
        # A.shape = (batch_size, n_attention_heads, sentence_length)
        # M.shape = (batch_size, n_attention_heads, hidden_size)

        concat_M=M.flatten(1)
        membership = F.softmax(self.fc2(F.dropout(F.elu(self.fc1(concat_M)),p=0.2)), dim=1)

        return membership, concat_M, A
