import torch.nn as nn
import torch
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_low, action_high, log_std_min=-20, log_std_max=2):
        super().__init__()

        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.action_scale = torch.FloatTensor((action_high - action_low) / 2)
        self.action_bias = torch.FloatTensor((action_high + action_low) / 2)

    def forward(self, state):
        features = self.shared_layers(state)
        mean = self.mean_head(features)
        log_std = self.log_std(features)

        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        std = torch.exp(log_std)
        return mean, std

    def sample(self, state, epsilon=1e-6):
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        u = dist.rsample()  # reparameterization trick
        tanh_u = torch.tanh(u) # squash into [-1, 1]

        # This samples exact log-probability density of the chosen action
        # torch.log(1 - tanh_u.pow(2)) corrects the log_prob from the distribution graph
        # P(a) = P(u) * (du/da)
        # da/du = 1-tan^2(u)
        # du/da = 1/(1-tan^2(u))
        # log(P(a)) = log(P(u) * (du/da))
        # log(P(a)) = log(P(u)) + log(du/da)
        # log(P(a)) = log(P(u)) + log(1/(1-tan^2(u)))
        # log(P(a)) = log(P(u)) - log(1-tan^2(u))
        # High std outputs high log to show confidence
        log_prob = dist.log_prob(u) - torch.log(1 - tanh_u.pow(2) + epsilon)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        action = tanh_u * self.action_scale + self.action_bias  # rescale to env action range

        # Log-prob is used for entropy to validate the confidence
        # High log prob means higher confidence and lower log prob means lower confidence
        return action, log_prob

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.sequence(x)