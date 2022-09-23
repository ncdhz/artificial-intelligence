import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np

class ActorNet(nn.Module):
    def __init__(self, n_features, n_hidden, n_outputs, bound):
        super().__init__()
        self.bound = torch.tensor(bound)
        self.l1 = nn.Linear(n_features, n_hidden)
        self.output = nn.Linear(n_hidden, n_outputs)
    
    def forward(self, x):
        x = F.tanh(self.output(F.relu(self.l1(x))) ) 
        return x * self.bound

class CriticNet(nn.Module):
    def __init__(self, a_features, s_features, n_hidden=30, n_outputs=1):
        super().__init__()
        self.la = nn.Linear(a_features, n_hidden)
        self.ls = nn.Linear(s_features, n_hidden)
        self.output = nn.Linear(n_hidden, n_outputs)

    def forward(self, s, a):
        x1 = self.la(a)
        x2 = self.ls(s)
        x = F.relu(x1 + x2)
        return self.output(x)

class DDPG:
    def __init__(self, a_features, s_features, a_bound, lra=0.001, lrc=0.002, memory_capacity=10000, batch_size=32, tau=0.01, gamma=0.9):
        self.a_features = a_features
        self.s_features = s_features
        self.a_bound = a_bound
        self.lra = lra
        self.lrc = lrc
        self.bacth_size = batch_size
        self.pointer = 0
        self.memory_capacity = memory_capacity
        self.gamma = gamma
        self.tau = tau
        self.memory = np.zeros((memory_capacity, s_features * 2 + a_features + 1), dtype=np.float32)
        self.__build_net()
    
    def __build_net(self):
        self.actor_eval = ActorNet(self.s_features, 30, self.a_features, self.a_bound)
        self.actor_target = ActorNet(self.s_features, 30, self.a_features, self.a_bound)
        self.actor_target.eval()
        self.actor_optimizer = Adam(self.actor_eval.parameters(), lr=self.lra)

        self.critic_eval = CriticNet(self.a_features, self.s_features)
        self.critic_target = CriticNet(self.a_features, self.s_features)
        self.critic_target.eval()
        self.critic_optimizer = Adam(self.critic_eval.parameters(), lr=self.lra)

        self.loss_func = nn.MSELoss()
        

    def choose_action(self, s):
        x = torch.FloatTensor(s[np.newaxis, :])
        return self.actor_eval(x)[0].item()

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
         # replace the old memory with new memory
        index = self.pointer % self.memory_capacity 
        self.memory[index, :] = transition
        self.pointer += 1

    def update_parameter(self, net1, net2):
        net1_state = net1.state_dict()
        net2_state = net2.state_dict()

        for key in net1_state.keys():
            net1_state[key] = net1_state[key] * (1 - self.tau) + self.tau * net2_state[key]
        
        net1.load_state_dict(net1_state)

    def learn(self):
        self.update_parameter(self.actor_target, self.actor_eval)
        self.update_parameter(self.critic_target, self.critic_eval)
        # sample from buffer a batch data
        indices = np.random.choice(self.memory_capacity, size=self.bacth_size)
        bt = torch.FloatTensor(self.memory[indices, :])

        bs = bt[:, :self.s_features]
        ba = bt[:, self.s_features: self.s_features + self.a_features]
        br = bt[:, -self.s_features - 1: - self.s_features]
        bs_ = bt[:, -self.s_features:]

        a = self.actor_eval(bs)
        q = self.critic_eval(bs, a)

        action_loss = -torch.mean(q)
        # optimize the loss of actor network
        self.actor_optimizer.zero_grad()
        action_loss.backward()
        self.actor_optimizer.step()

        # compute the target Q value using the information of next state
        with torch.no_grad():
            at = self.actor_target(bs_)
            qtmp = self.critic_target(bs_, at)
            qt = br + self.gamma * qtmp
        # compute the current q value and the loss
        qe = self.critic_eval(bs, ba)
        tde = self.loss_func(qe, qt)
        # optimize the loss of critic network
        self.critic_optimizer.zero_grad()
        tde.backward()
        self.critic_optimizer.step()



    

        
