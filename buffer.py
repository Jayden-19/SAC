import torch
import torch.nn as nn
import numpy as np

class SumTree:
  def __init__(self, capacity):
    self.capacity = capacity
    self.tree = np.zeros(2*capacity-1)
    self.data = np.zeros(capacity, dtype=object)
    self.write = 0
    self.n_entries = 0

  def _propagate(self,idx,change):
    parent = (idx-1)//2
    self.tree[parent] += change
    if parent != 0:
      self._propagate(parent, change)

  def _retrieve(self,idx,s):
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

  def add(self, priority, data):
    idx = self.write + self.capacity - 1
    self.data[self.write] = data
    self.update(idx, priority)
    self.write = (self.write + 1) % self.capacity
    self.n_entries = min(self.n_entries + 1, self.capacity)

  def update(self, idx, priority):
    change = priority - self.tree[idx]
    self.tree[idx] = priority
    self._propagate(idx, change)

  def get(self, s):
    idx = self._retrieve(0, s)
    data_idx = min(idx - self.capacity + 1, self.n_entries - 1)
    idx = data_idx + self.capacity - 1
    return idx, self.tree[idx], self.data[data_idx]



class PrioritizedReplayBuffer:
  def __init__(self, capacity, seed, alpha=0.6, eps=1e-5):
    self.capacity = capacity
    self.tree = SumTree(capacity)
    self.alpha = alpha
    self.eps = eps
    self.max_priority = 1.0
    self.rand_generator = np.random.RandomState(seed)

  def append(self, state, action, reward, next_state, done):
    data = (state, action, reward, next_state, done)
    priority = self.max_priority** self.alpha
    self.tree.add(priority, data)

  def sample(self, batch_size, beta=0.4):
    batch, idxs, priorities = [], [], []
    segment = self.tree.total() / batch_size

    for i in range(batch_size):
      a, b = segment * i, segment * (i+1)
      s = self.rand_generator.uniform(a, b)
      idx, priority, data = self.tree.get(s)
      batch.append(data)
      idxs.append(idx)
      priorities.append(priority)

    states, actions, rewards, next_states, dones = zip(*batch)

    sampling_probs = np.array(priorities) / self.tree.total()
    weights = (self.tree.n_entries * sampling_probs) ** (-beta)
    weights /= weights.max()

    return (
        torch.FloatTensor(np.array(states)),
        torch.FloatTensor(actions),
        torch.FloatTensor(rewards).unsqueeze(1),
        torch.FloatTensor(np.array(next_states)),
        torch.FloatTensor(dones).unsqueeze(1),
        idxs,
        torch.FloatTensor(weights).unsqueeze(1)
    )

  def update_priorities(self, idxs, td_errors):
    for idx, td_error in zip(idxs, td_errors):
      priority = (abs(td_error) + self.eps) ** self.alpha
      self.tree.update(idx, priority)
      self.max_priority = max(self.max_priority, priority)

  def __len__(self):
    return self.tree.n_entries