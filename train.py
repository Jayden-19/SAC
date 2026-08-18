import torch
import copy
from tqdm import tqdm
import numpy as np

class OUNoise:
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        # self.mu is the mean of the distribution
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        # This formula allows sampling of random states and also being pulled back to the mean of the environment/states
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.rand(self.action_dim)
        self.state += dx
        return self.state
    
def train_sac(env, agent, replay_buffer, total_steps, batch_size, gamma, warmup_steps=1000, potential_fn=None, exploration_noise=None):
    state, _ = env.reset()
    episode_reward = 0 
    episode_rewards = []
    actor_losses = []
    critic1_losses = []
    critic2_losses = []
    alpha_losses = []
    alphas = []

    best_reward = -float('inf')
    best_state_dict = None

    pbar = tqdm(range(total_steps))
    for step in pbar:
        if step < warmup_steps:
            if exploration_noise is not None:
                action = np.clip(exploration_noise.sample(), env.action_space.low, env.action_space.high)
            else:
                action = env.action_space.sample() # pure random exploration
        else:
            with torch.no_grad():
                action, _ = agent.actor.sample(torch.FloatTensor(state).unsqueeze(0))
                action = action.squeeze(0).numpy()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Reward shaping - allows the agent to not be overly conservative
        if potential_fn is not None:
            # F(s, a, s') = γ·Φ(s') − Φ(s)
            next_potential = 0.0 if terminated else potential_fn(next_state)
            f_s_a_s_prime = gamma * next_potential - potential_fn(state)
            # shaped_reward = original_reward + F(s, a, s')
            shaped_reward = reward + f_s_a_s_prime
        else:
            shaped_reward = reward

        replay_buffer.append(state, action, shaped_reward, next_state, done)
        episode_reward += reward
        state = next_state

        if done:
            episode_rewards.append(episode_reward)
            if episode_reward > best_reward:
                best_reward = episode_reward
                best_state_dict = copy.deepcopy(agent.actor.state_dict())
            avg_reward = sum(episode_rewards[-10:]) / len(episode_rewards[-10:])
            print(f"Episode = {len(episode_rewards)}, Reward = {episode_reward:.1f}, Avg10 = {avg_reward:.1f}, Best Reward={best_reward:.1f}")
            episode_reward = 0
            state, _ = env.reset()
            if exploration_noise is not None:
                exploration_noise.reset()

        if len(replay_buffer) > batch_size and step >= warmup_steps:
            batch = replay_buffer.sample(batch_size)
            critic1_loss, critic2_loss_, actor_loss, alpha_loss, alpha, idxs, td_errors = agent.train(batch, gamma)
            replay_buffer.update_priorities(idxs, td_errors)
            critic1_losses.append(critic1_loss)
            critic2_losses.append(critic2_loss_)
            actor_losses.append(actor_loss)
            alpha_losses.append(alpha_loss) 
            alphas.append(alpha)

    return episode_rewards, critic1_losses, critic2_losses, actor_losses, alpha_losses, alphas, best_state_dict