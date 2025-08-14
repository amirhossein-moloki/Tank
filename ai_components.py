import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math
import numpy as np
from collections import deque
from constants import *

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            return F.linear(x, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(x, self.weight_mu, self.bias_mu)

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            NoisyLinear(128, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            NoisyLinear(128, output_size)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum(A(s,a')))
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals

    def reset_noise(self):
        for name, module in self.named_children():
            if isinstance(module, nn.Sequential):
                for sub_name, sub_module in module.named_children():
                    if hasattr(sub_module, 'reset_noise'):
                        sub_module.reset_noise()

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[data_idx])

class PrioritizedReplayMemory:
    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.beta = PER_BETA_START
        self.beta_increment_per_sampling = (1.0 - PER_BETA_START) / PER_BETA_FRAMES

    def _get_priority(self, error):
        return (np.abs(error) + PER_EPSILON) ** PER_ALPHA

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, torch.from_numpy(is_weight).float().to(device)

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def __len__(self):
        return self.tree.n_entries

class DQNAgent:
    def __init__(self, agent_id, state_size, action_size, n_step, gamma):
        self.agent_id = agent_id
        self.state_size = state_size
        self.action_size = action_size
        self.n_step = n_step
        self.gamma = gamma

        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = PrioritizedReplayMemory(REPLAY_MEMORY_SIZE)

    def select_action(self, state):
        self.policy_net.reset_noise()
        with torch.no_grad():
            return self.policy_net(state).max(1)[1].view(1, 1)

    def store_experience(self, state, action, next_state, reward):
        # Calculate initial error to give new experiences max priority
        with torch.no_grad():
            state_action_value = self.policy_net(state).gather(1, action)

            next_state_value = 0.0
            if next_state is not None:
                best_next_action = self.policy_net(next_state).max(1)[1].unsqueeze(1)
                next_state_value = self.target_net(next_state).gather(1, best_next_action).squeeze().item()

            # For N-step, the future reward is discounted by gamma^N
            expected_state_action_value = (next_state_value * (self.gamma ** self.n_step)) + reward.item()
            error = abs(state_action_value.item() - expected_state_action_value)

        self.memory.add(error, Transition(state, action, next_state, reward))

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return

        mini_batch, idxs, is_weights = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*mini_batch))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        self.target_net.reset_noise()
        best_next_actions = self.policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, best_next_actions).squeeze(1).detach()

        # For N-step, the future reward is discounted by gamma^N
        expected_state_action_values = (next_state_values * (self.gamma ** self.n_step)) + reward_batch

        # Compute Huber loss with no reduction
        criterion = nn.SmoothL1Loss(reduction='none')
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Apply importance sampling weights
        loss = (is_weights * loss).mean()

        # Update priorities in memory
        errors = torch.abs(state_action_values.squeeze() - expected_state_action_values).detach().cpu().numpy()
        for i in range(len(idxs)):
            self.memory.update(idxs[i], errors[i])

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
