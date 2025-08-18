import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal  # We need this for PPO
from torch.utils.tensorboard import SummaryWriter


def AvgL1Norm(x, eps=1e-8):
    return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)

# The Encoder can remain the same as it's a great way to process state information
class Encoder(nn.Module):
    def __init__(self, state_dim, zs_dim=256, hdim=256, activ=F.elu):
        super(Encoder, self).__init__()
        self.activ = activ
        self.zs1 = nn.Linear(state_dim, hdim)
        self.zs2 = nn.Linear(hdim, hdim)
        self.zs3 = nn.Linear(hdim, zs_dim)

    def zs(self, state):
        zs = self.activ(self.zs1(state))
        zs = self.activ(self.zs2(zs))
        # The normalization is still a good idea
        # In Encoder.zs()
        zs = self.zs3(zs)
        zs = AvgL1Norm(zs)
        # zs = x / x.abs().mean(-1, keepdim=True).clamp(min=1e-8)
        return zs

# MODIFIED: The PPO Actor outputs a distribution, not a single action
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256, activ=F.relu):
        super(Actor, self).__init__()
        self.activ = activ
        self.l0 = nn.Linear(state_dim, hdim)
        self.l1 = nn.Linear(zs_dim + hdim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        # This layer now outputs the MEAN of the action distribution
        self.actor_mean = nn.Linear(hdim, action_dim)
        # We also need a learnable standard deviation for the distribution
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, state, zs):
        # In Actor.forward()
        a = self.l0(state)
        a = AvgL1Norm(a)
        # a = x / x.abs().mean(-1, keepdim=True).clamp(min=1e-8)(self.l0(state))
        a = torch.cat([a, zs], 1)
        a = self.activ(self.l1(a))
        a = self.activ(self.l2(a))
        # Get the mean from the network
        action_mean = self.actor_mean(a)
        # Exponentiate the log_std to get the actual standard deviation
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        # Return the distribution parameters
        return action_mean, action_std

# MODIFIED: The PPO Critic outputs a single state-value V(s)
class Critic(nn.Module):
    def __init__(self, state_dim, zs_dim=256, hdim=256, activ=F.elu):
        super(Critic, self).__init__()
        self.activ = activ
        # The critic only needs to see the state, not the action
        self.l0 = nn.Linear(state_dim, hdim)
        self.l1 = nn.Linear(zs_dim + hdim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.critic_v = nn.Linear(hdim, 1) # Outputs a single value

    def forward(self, state, zs):
        # In Critic.forward()
        v = self.l0(state)
        v = AvgL1Norm(v)
        # v = x / x.abs().mean(-1, keepdim=True).clamp(min=1e-8)(self.l0(state))
        v = torch.cat([v, zs], 1)
        v = self.activ(self.l1(v))
        v = self.activ(self.l2(v))
        # Return the state-value V(s)
        return self.critic_v(v)

# REWRITTEN: The Agent class is now structured for PPO's on-policy, rollout-based learning
class Agent(object):
    def __init__(self, state_dim, action_dim, max_action, hp, log_dir=None):
        # PPO Hyperparameters
        self.lr = hp.get("learning_rate", 3e-4)
        self.n_steps = hp.get("n_steps", 2048) # Steps per rollout
        self.gamma = hp.get("gamma", 0.99) # Discount factor
        self.gae_lambda = hp.get("gae_lambda", 0.95) # GAE parameter
        self.n_epochs = hp.get("n_epochs", 10) # Update epochs per rollout
        self.clip_coef = hp.get("clip_coef", 0.2) # PPO clipping coefficient
        self.ent_coef = hp.get("ent_coef", 0.0) # Entropy coefficient
        self.vf_coef = hp.get("vf_coef", 0.5) # Value function coefficient

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Keeping the Encoder is a good idea
        self.encoder = Encoder(state_dim, zs_dim=256).to(self.device)
        self.actor = Actor(state_dim, action_dim, zs_dim=256).to(self.device)
        self.critic = Critic(state_dim, zs_dim=256).to(self.device)

        # PPO uses a single optimizer for all networks
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.lr
        )

        self.max_action = max_action
        self.training_steps = 0
        self.writer = SummaryWriter(log_dir=log_dir)

        # PPO needs temporary storage for one rollout, not a big replay buffer
        self.memory = {
            "states": torch.zeros((self.n_steps, state_dim)).to(self.device),
            "actions": torch.zeros((self.n_steps, action_dim)).to(self.device),
            "logprobs": torch.zeros((self.n_steps,)).to(self.device),
            "rewards": torch.zeros((self.n_steps,)).to(self.device),
            "dones": torch.zeros((self.n_steps,)).to(self.device),
            "values": torch.zeros((self.n_steps,)).to(self.device),
        }
        self.mem_idx = 0

    def get_action_and_value(self, state, action=None):
        """ Get action, log probability, entropy, and state-value """
        with torch.no_grad():
             # The encoder is used here
            zs = self.encoder.zs(state)
        action_mean, action_std = self.actor(state, zs)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        log_prob = probs.log_prob(action).sum(1)
        entropy = probs.entropy().sum(1)
        value = self.critic(state, zs)
        return action, log_prob, entropy, value

    def store_transition(self, state, action, logprob, reward, done, value):
        """ Store a single transition in the rollout memory """
        if self.mem_idx < self.n_steps:
            self.memory["states"][self.mem_idx] = state
            self.memory["actions"][self.mem_idx] = action
            self.memory["logprobs"][self.mem_idx] = logprob.detach()
            self.memory["rewards"][self.mem_idx] = torch.tensor(reward).to(self.device)
            self.memory["dones"][self.mem_idx] = torch.tensor(done).to(self.device)
            self.memory["values"][self.mem_idx] = value.detach().flatten()
            self.mem_idx += 1

    def learn(self, next_state, next_done):
        """ Perform the PPO update after a rollout is complete """
        # 1. Calculate advantages using GAE
        with torch.no_grad():
            next_value = self.get_action_and_value(next_state)[3].reshape(1, -1)
            advantages = torch.zeros_like(self.memory["rewards"]).to(self.device)
            last_gae_lam = 0
            for t in reversed(range(self.n_steps)):
                if t == self.n_steps - 1:
                    next_nonterminal = 1.0 - next_done
                    next_values = next_value
                else:
                    next_nonterminal = 1.0 - self.memory["dones"][t + 1]
                    next_values = self.memory["values"][t + 1]
                delta = self.memory["rewards"][t] + self.gamma * next_values * next_nonterminal - self.memory["values"][t]
                advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_nonterminal * last_gae_lam
            returns = advantages + self.memory["values"]

        # Flatten the batch
        b_states = self.memory["states"]
        b_actions = self.memory["actions"]
        b_logprobs = self.memory["logprobs"]
        b_advantages = advantages
        b_returns = returns
        b_values = self.memory["values"]

        # 2. Optimize policy and value network for K epochs
        for epoch in range(self.n_epochs):
            _, new_logprob, entropy, new_value = self.get_action_and_value(b_states, b_actions)
            logratio = new_logprob - b_logprobs
            ratio = logratio.exp()

            # Policy loss (PPO-Clip)
            pg_loss1 = -b_advantages * ratio
            pg_loss2 = -b_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            v_loss = F.mse_loss(new_value.flatten(), b_returns)

            # Entropy loss
            entropy_loss = entropy.mean()

            # Total loss
            loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Reset memory for the next rollout
        self.mem_idx = 0
        self.training_steps += self.n_epochs * self.n_steps

        # Logging
        self.writer.add_scalar("loss", loss.item(), self.training_steps)
        self.writer.add_scalar("value_loss", v_loss.item(), self.training_steps)

    # The save and load functions would need to be adapted for the new network names
    def save(self, directory, filename):
        torch.save(self.actor.state_dict(), f"{directory}/{filename}_actor.pth")
        torch.save(self.critic.state_dict(), f"{directory}/{filename}_critic.pth")
        torch.save(self.encoder.state_dict(), f"{directory}/{filename}_encoder.pth")

    def load(self, directory, filename):
        self.actor.load_state_dict(torch.load(f"{directory}/{filename}_actor.pth"))
        self.critic.load_state_dict(torch.load(f"{directory}/{filename}_critic.pth"))
        self.encoder.load_state_dict(torch.load(f"{directory}/{filename}_encoder.pth"))