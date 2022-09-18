import numpy as np
import torch
from torch.optim import RMSprop
from torch import nn

class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super().__init__()
        self.el = nn.Linear(n_feature, n_hidden)
        self.values = nn.Linear(n_hidden, 1)
        self.advantages = nn.Linear(n_hidden, n_output)
        self.relu = nn.ReLU()

    def forward(self, x):
        el = self.relu(self.el(x))
        value = self.values(el)
        advantages = self.advantages(el)
        return value + (advantages - advantages.mean(dim=1, keepdim=True))

# Deep Q Network off-policy
class DuelingDQN:
    def __init__(self, n_actions, n_features, n_hidden=20, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, replace_target_iter=300, memory_size=500, batch_size=32, e_greedy_increment=None):
        self.n_actions = n_actions
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.memory_counter = 0

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()

        self.cost_his = []

    def _build_net(self):
        self.q_eval = Net(self.n_features, self.n_hidden, self.n_actions)
        self.q_eval.train()
        self.q_target = Net(self.n_features, self.n_hidden, self.n_actions)
        self.q_target.eval()
        self.optimizer = RMSprop(self.q_eval.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = torch.FloatTensor(observation[np.newaxis, :])

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.q_eval(observation)
            action = actions_value.argmax().item()
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.q_target.load_state_dict(self.q_eval.state_dict())

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = torch.FloatTensor(self.memory[sample_index, :])

        with torch.no_grad():
            q_next = self.q_target(batch_memory[:, -self.n_features:])
        q_eval = self.q_eval(batch_memory[:, :self.n_features])

        # change q_target w.r.t q_eval's action
        q_target = q_eval.detach().clone()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].type(torch.long)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * q_next.max(dim=1).values

        # train eval network
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.cost_his.append(loss.item())

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
