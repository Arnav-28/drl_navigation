import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# A standard, simple replay buffer for storing and sampling experiences
class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        """Adds a new transition to the buffer."""
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """Samples a batch of transitions from the buffer."""
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

# The Actor network, which maps state to action
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

# The Critic network, which learns the Q-value (state-action value)
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

# The main Agent class that ties everything together
class Agent(object):
    def __init__(self, state_dim, action_dim, max_action, hp):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_action = max_action

        # --- Hyperparameters ---
        self.discount = hp.get("discount", 0.99)
        self.tau = hp.get("tau", 0.005)
        self.policy_noise = hp.get("policy_noise", 0.2)
        self.noise_clip = hp.get("noise_clip", 0.5)
        self.policy_freq = hp.get("policy_freq", 2)
        self.exploration_noise = hp.get("exploration_noise", 0.1)
        self.lr = hp.get("lr", 3e-4)
        self.batch_size = hp.get("batch_size", 256) # <-- CHECKPOINTING ADDITION: Need batch_size for train_and_reset

        # --- Networks ---
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        # --- Replay Buffer ---
        buffer_size = hp.get("buffer_size", int(1e6))
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, buffer_size)
        
        # --- Training Counter ---
        self.total_it = 0

        # --- CHECKPOINTING ADDITIONS ---
        self.checkpoint_actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.checkpoint_actor.load_state_dict(self.actor.state_dict())
        
        # Hyperparameters for checkpointing
        self.reset_weight = hp.get("reset_weight", 0.9)
        self.steps_before_checkpointing = hp.get("steps_before_checkpointing", 500000)
        self.max_eps_when_checkpointing = hp.get("max_eps_when_checkpointing", 10)

        # Tracking variables
        self.eps_since_update = 0
        self.timesteps_since_update = 0
        self.max_eps_before_update = 1
        self.min_return = 1e8
        self.best_min_return = -1e8
        # --- END CHECKPOINTING ADDITIONS ---

    # <-- CHECKPOINTING ADDITION: `use_checkpoint` argument added back -->
    def select_action(self, state, use_checkpoint=False, use_exploration=True):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        
        # <-- CHECKPOINTING ADDITION: Logic to select which actor to use -->
        if use_checkpoint:
            action = self.checkpoint_actor(state).cpu().data.numpy().flatten()
        else:
            action = self.actor(state).cpu().data.numpy().flatten()

        if use_exploration:
            noise = np.random.normal(0, self.max_action * self.exploration_noise, size=action.shape)
            action = (action + noise).clip(-self.max_action, self.max_action)
        return action

    def train(self, batch_size=256):
        self.total_it += 1
        state, action, next_state, reward, not_done = self.replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    # <-- CHECKPOINTING ADDITION: All methods below are new -->
    def train_and_checkpoint(self, ep_timesteps, ep_return):
        """If using checkpoints: run when each episode terminates."""
        self.eps_since_update += 1
        self.timesteps_since_update += ep_timesteps
        self.min_return = min(self.min_return, ep_return)

        # End evaluation of current policy early if it's already worse than the best
        if self.min_return < self.best_min_return:
            self.train_and_reset()
        # Update checkpoint if we have evaluated for enough episodes
        elif self.eps_since_update == self.max_eps_before_update:
            self.best_min_return = self.min_return
            self.checkpoint_actor.load_state_dict(self.actor.state_dict())
            self.train_and_reset()

    def train_and_reset(self):
        """Batch training and reset tracking variables."""
        for _ in range(self.timesteps_since_update):
            if self.total_it == self.steps_before_checkpointing:
                self.best_min_return *= self.reset_weight
                self.max_eps_before_update = self.max_eps_when_checkpointing
            self.train(batch_size=self.batch_size)

        self.eps_since_update = 0
        self.timesteps_since_update = 0
        self.min_return = 1e8

    def save(self, directory, filename):
        """Save model parameters"""
        torch.save(self.critic.state_dict(), f"{directory}/{filename}_critic.pth")
        torch.save(self.actor.state_dict(), f"{directory}/{filename}_actor.pth")
        # Save the checkpoint actor
        torch.save(self.checkpoint_actor.state_dict(), f"{directory}/{filename}_checkpoint_actor.pth")

    def load(self, directory, filename):
        """Load model parameters"""
        self.critic.load_state_dict(torch.load(f"{directory}/{filename}_critic.pth"))
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor.load_state_dict(torch.load(f"{directory}/{filename}_actor.pth"))
        self.actor_target.load_state_dict(self.actor.state_dict())
        # Load the checkpoint actor
        self.checkpoint_actor.load_state_dict(torch.load(f"{directory}/{filename}_checkpoint_actor.pth"))