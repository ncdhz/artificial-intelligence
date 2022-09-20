import numpy as np
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super().__init__()
        self.layer = nn.Linear(n_feature, n_hidden)
        self.tanh = nn.Tanh()
        self.act = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        return self.act(self.tanh(self.layer(x)))

class PolicyGradient:
    def __init__(self, n_actions, n_features, n_hidden=10, learning_rate=0.01, reward_decay=0.95) -> None:
        self.n_actions = n_actions
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_net()
    
    def _build_net(self):
        self.net = Net(self.n_features, self.n_hidden, self.n_actions)
        self.optimizer = Adam(self.net.parameters(), lr=self.lr)
    
    def choose_action(self, observation):
        observation = torch.FloatTensor(observation[np.newaxis, :])
        self.net.eval()
        with torch.no_grad():
            prob_weights = self.net(observation)
        prob = F.softmax(prob_weights, dim=1)
        return np.random.choice(range(prob.shape[1]), p=prob.view(-1).numpy())

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        self.net.train()
        discounted_ep_rs_norm = self.__discount_and_norm_rewards()
        obs = torch.FloatTensor(self.ep_obs)
        acts = torch.LongTensor(self.ep_as)
        vt = torch.FloatTensor(discounted_ep_rs_norm)

        all_act = self.net(obs)

        neg_log_prob = F.cross_entropy(all_act, acts, reduction='none')
        loss = torch.mean(neg_log_prob * vt)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        return discounted_ep_rs_norm

    def __discount_and_norm_rewards(self):
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add
        
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)

        return discounted_ep_rs
    