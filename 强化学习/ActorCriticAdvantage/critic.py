from torch import nn
import torch
from torch.optim import Adam
import numpy as np
import torch.nn.functional as F

class CriticNet(nn.Module):
    def __init__(self, n_features, n_hidden, n_outputs):
        super().__init__()
        self.l1 = nn.Linear(n_features, n_hidden)
        self.v = nn.Linear(n_hidden, n_outputs)


    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.v(x)
        return x

class Critic:
    def __init__(self, n_features, n_hidden=30, n_output=1, lr=0.01, gamma=0.9):
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.lr = lr
        self.gamma = gamma
        self._build_net()

    def _build_net(self):
        self.critic_net = CriticNet(self.n_features, self.n_hidden, self.n_output)
        self.optimizer = Adam(self.critic_net.parameters(), lr=self.lr)
    
    def learn(self, s, r, s_):
        s, s_ = torch.FloatTensor(s[np.newaxis, :]), torch.FloatTensor(s_[np.newaxis, :])
        v, v_ = self.critic_net(s), self.critic_net(s_)
        td_error = torch.mean(r + self.gamma * v_.double() - v.double())
        loss = td_error ** 2

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

        return td_error


