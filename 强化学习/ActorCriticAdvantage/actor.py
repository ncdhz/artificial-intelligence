import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

class ActorNet(nn.Module):
    def __init__(self, n_features, n_hidden, n_outputs):
        super().__init__()
        self.l1 = nn.Linear(n_features, n_hidden)
        self.mu = nn.Linear(n_hidden, n_outputs)
        self.sigma = nn.Linear(n_hidden, n_outputs)
    
    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        mu = self.mu(x)
        mu = torch.tanh(mu) # [-1, 1]
        sigma = self.sigma(x)
        sigma = F.softplus(sigma) # [0, ∞]
        return mu, sigma

class Actor(object):
    def __init__(self, n_features, action_bound, n_hidden=30, lr=0.0001):
        self.n_features = n_features
        self.action_bound = action_bound
        self.n_hidden = n_hidden
        self.lr = lr

        self._build_net()

    def _build_net(self):
        self.actor_net = ActorNet(self.n_features, self.n_hidden, 1)
        self.optimizer = Adam(self.actor_net.parameters(), lr=self.lr)

    def normal_dist(self, s):
        s = torch.FloatTensor(s[np.newaxis, :])
        mu, sigma = self.actor_net(s)
        mu, sigma = (mu * 2).squeeze(),  (sigma + 0.1).squeeze()
        # get the normal distribution of average=mu and std=sigma
        normal_dist = torch.distributions.Normal(mu, sigma)
        return normal_dist
    
    def choose_action(self, s):
        self.actor_net.eval()
        normal_dist = self.normal_dist(s)
        # sample action accroding to the distribution
        with torch.no_grad():
            action = torch.clamp(normal_dist.sample(), self.action_bound[0], self.action_bound[1])
        return action.item()

    def learn(self, s, a, td):
        self.actor_net.train()
        normal_dist = self.normal_dist(s)
        # log_prob get the probability of action a under the distribution of normal_dist
        log_prob = normal_dist.log_prob(torch.tensor(a))
        # advantage (TD_error) guided loss
        exp_v = log_prob * torch.tensor(td.item())
        # Add cross entropy cost to encourage exploration
        exp_v += 0.01 * normal_dist.entropy()
        # max(v) = min(-v)
        loss = - exp_v   
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return exp_v
        