from network import Actor, Critic
import torch.optim as optim
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class SACAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, env, alpha_lr, actor_lr, critic_lr):
        self.actor = Actor(state_dim, hidden_dim, action_dim, env.action_space.low, env.action_space.high)
        self.critic1 = Critic(state_dim, hidden_dim, action_dim)
        self.critic2 = Critic(state_dim, hidden_dim, action_dim)

        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)
        
        self.target_entropy = -action_dim
        self.log_alpha = nn.Parameter(torch.zeros(1))

        self.actor_optimizer = optim.Adam(self.actor.parameters(), actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), critic_lr)

        self.alpha_optimizer = optim.Adam([self.log_alpha], alpha_lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def update_actor(self, states, alpha):
        action, log_prob = self.actor.sample(states)
        # log_prob = πθ(a|s)
        # log_prob shape [batch_size, 1]
        # action shape [batch_size, action_dim]

        q1 = self.critic1(states, action)
        q2 = self.critic2(states, action)
        # For pessimistic Q-value estimation, we take the minimum of the two Q-values
        q_min = torch.min(q1, q2)

        # L(θ) = E[ α log πθ(a|s) − min Q(s,a) ]
        # Goal to maximise - q_min (reward) and reduce log_prob to increase exploration
        # J(theta) = q_min - alpha * log_prob
        # To minimise log_prob (Reduce loss)
        # Loss = -1 * J(theta)
        # Loss = alpha * log_prob - q_min
        actor_loss = torch.mean(alpha * log_prob - q_min)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        return actor_loss.item(), log_prob.detach()

    def update_critics(
            self, states, actions, rewards,
            next_states, dones, alpha, gamma, weights
    ):
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_states)
            # next_log_prob = πθ(a'|s_{t+1})  — computed inside actor.sample

            target_q1 = self.target_critic1(next_states, next_action)
            target_q2 = self.target_critic2(next_states, next_action)
            # For pessimistic Q-value estimation, we take the minimum of the two target Q-values
            target_q_min = torch.min(target_q1, target_q2)

            # entropy-augmented soft Bellman target
            # next_log_prob will determine exploration or exploitation
            # Higher next_log_prob means higher confidence thus reducing exploration
            y = rewards + gamma * (1 - dones) * (target_q_min - alpha * next_log_prob)

        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)

        td_error1 = current_q1 - y
        td_error2 = current_q2 - y

        # Massive prediction error will have higher weights
        # Thus, increasing loss
        critic1_loss = (weights * td_error1.pow(2)).mean()
        critic2_loss = (weights * td_error2.pow(2)).mean()

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        td_errors = torch.max(td_error1.abs(), td_error2.abs()).detach()

        return critic1_loss.item(), critic2_loss.item(), td_errors

    def update_alpha(self, log_prob):
        # Lower entropy means more confidence thus increasing alpha to increase entropy to increase exploration is needed
        # High entropy means less confidence thus lowering alpha to not exaggerate or lose sight of other actions
        alpha_loss = (-self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        alpha = self.alpha

        return alpha, alpha_loss.item()

    def target_soft_update(self, tau=0.005):
        critic = [self.critic1, self.critic2]
        target_critic = [self.target_critic1, self.target_critic2]

        for c, t in zip(critic, target_critic):
            for c_param, t_param in zip(c.parameters(), t.parameters()):
                t_param.data.copy_(tau * c_param.data + (1 - tau) * t_param.data)

    def train(self, batch, gamma):
        states, actions, rewards, next_states, dones, idxs, weights = batch

        critic1_loss, critic2_loss, td_errors = self.update_critics(states, actions, rewards, next_states, dones, self.alpha, gamma, weights)

        actor_loss, log_prob = self.update_actor(states, self.alpha)

        alpha, alpha_loss = self.update_alpha(log_prob)

        self.target_soft_update()

        return critic1_loss, critic2_loss, actor_loss, alpha_loss, alpha, idxs, td_errors