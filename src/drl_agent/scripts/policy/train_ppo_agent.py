#!/usr/bin/env python3

import os
import sys
import rclpy
import time
from datetime import date
import torch
import numpy as np
from ppo_agent import Agent
import csv
from environment_interface import EnvInterface
from file_manager import DirectoryManager, load_yaml

class TrainPPO(EnvInterface):
    def __init__(self):
        super().__init__("train_ppo_node")

        # Load training config parameters
        drl_agent_src_path_env = "DRL_AGENT_SRC_PATH"
        drl_agent_src_path = os.getenv(drl_agent_src_path_env)
        if drl_agent_src_path is None:
            self.get_logger().error(
                f"Environment variable: {drl_agent_src_path_env}, is not set"
            )
        drl_agent_pkg_path = os.path.join(drl_agent_src_path, "drl_agent")

        self.hyperparameters_path = os.path.join(
            drl_agent_pkg_path, "config", "hyperparameters_ppo.yaml"
        )
        self.train_config_file_path = os.path.join(
            drl_agent_pkg_path, "config", "train_config.yaml"
        )

        # Load config file
        try:
            training_settings = load_yaml(self.train_config_file_path)["train_settings"]
        except Exception as e:
            self.get_logger().error(f"Unable to load config file: {e}")
            sys.exit(-1)
        # Extract training settings
        self.seed = training_settings["seed"]
        self.max_episode_steps = training_settings["max_episode_steps"]
        self.load_model = training_settings["load_model"]
        self.max_timesteps = training_settings["max_timesteps"]
        self.use_checkpoints = training_settings["use_checkpoints"]
        self.eval_freq = training_settings["eval_freq"]
        self.timesteps_before_training = training_settings["timesteps_before_training"]
        self.eval_eps = training_settings["eval_eps"]
        self.base_file_name = "ppo_agent"
        self.file_name = (
            f"{self.base_file_name}_seed_{self.seed}_{date.today().strftime('%Y%m%d')}"
        )

        # Setup directories for saving models, results and logs
        temp_dir_path = os.path.join(drl_agent_src_path, "drl_agent", "temp_ppo")
        self.pytorch_models_dir = os.path.join(temp_dir_path, "pytorch_models")
        self.final_models_dir = os.path.join(temp_dir_path, "final_models")
        self.results_dir = os.path.join(temp_dir_path, "results")
        self.log_dir = os.path.join(temp_dir_path, "logs")
        self.log_csv_path = os.path.join(self.results_dir, f"{self.base_file_name}.csv")
        self.initialize_log_csv()

        # Make directories
        self.create_directories()

        # Set seed
        self.set_env_seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Initialize the agent
        try:
            hyperparameters = load_yaml(self.hyperparameters_path)["hyperparameters"]
        except Exception as e:
            self.get_logger().error(f"Unable to load config file: {e}")
            sys.exit(-1)
        self.state_dim, self.action_dim, self.max_action = self.get_dimensions()
        self.rl_agent = Agent(
            self.state_dim,
            self.action_dim,
            self.max_action,
            hyperparameters,
            self.log_dir,
        )

        # Try to load the model
        if self.load_model:
            try:
                self.rl_agent.load(self.pytorch_models_dir, self.file_name)
                self.get_logger().info("Model loaded")
            except Exception as e:
                self.get_logger().warning(f"Failed to load the model: {e}")

        # Flag to indicate that the training is done
        self.done_training = False

        self.log_training_setting_data()

    def initialize_log_csv(self):
        with open(self.log_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Episode number", "Episode timesteps", "Total time steps", "Rewards"])

    def log_training_to_csv(self, ep_num, ep_timesteps, total_timesteps, ep_total_reward):
        with open(self.log_csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([ep_num, ep_timesteps, total_timesteps, ep_total_reward])

    def log_training_setting_data(self):
        """Log general info at the start of training"""
        self.border = "+" + "-" * 80 + "+"
        self.get_logger().info(self.border)
        self.get_logger().info(f"| File name: {self.file_name}: Seed: {self.seed}")
        self.get_logger().info(self.border)
        self.get_logger().info("| Results will be saved in:")
        self.get_logger().info(f"|  {self.pytorch_models_dir}")
        self.get_logger().info(f"|  {self.final_models_dir}")
        self.get_logger().info(f"|  {self.results_dir}")
        self.get_logger().info(f"|  {self.log_dir}")
        self.get_logger().info(self.border)
        self.get_logger().info("| Environment")
        self.get_logger().info(self.border)
        self.get_logger().info(f"| State Dim: {self.state_dim}")
        self.get_logger().info(f"| Action Dim: {self.action_dim}")
        self.get_logger().info(f"| Max Action: {self.max_action}")
        self.get_logger().info(self.border)

    def create_directories(self):
        """Create directories for saving models"""
        directories = [
            self.pytorch_models_dir,
            self.final_models_dir,
            self.results_dir,
            self.log_dir,
        ]
        for dir in directories:
            dir_manager = DirectoryManager(dir)
            dir_manager.remove_if_present()
            dir_manager.create()

    def save_models(self, directory, file_name):
        """Save the models at the given step"""
        self.rl_agent.save(directory, file_name)
        self.get_logger().info("Models updated")

        # In your TrainPPO class, which you would rename to TrainPPO
    def train_online(self):
        """ Train the PPO agent online """
        # Create an instance of our new PPOAgent
        # self.rl_agent = Agent(self.state_dim, self.action_dim, self.max_action, hyperparameters)

        state = self.reset()
        state = torch.tensor(state, dtype=torch.float32, device=self.rl_agent.device)
        ep_total_reward, ep_timesteps, ep_num = 0, 0, 1

        for t in range(1, self.max_timesteps + 1):
            # --- ROLLOUT PHASE ---
            # Get action and value from the PPO agent
            action, log_prob, _, value = self.rl_agent.get_action_and_value(state.reshape(1,-1))

            # Act in the environment
            # Note: PPO's actions are sampled, so we pass the numpy version to the env
            next_state, reward, done, _ = self.step(action.cpu().numpy().flatten())

            # We'll use a new variable for the final reward
            final_reward = reward
            ep_timesteps += 1
            
            # Check if the episode is ending ONLY because of the time limit
            is_timeout = (ep_timesteps >= self.max_episode_steps)
            
            if is_timeout and not done:
                self.get_logger().info(f"{'Max episode steps reached, applying penalty.':-^50}")
                final_reward = -100.0  # Override the last reward with a penalty

            # The 'done' flag passed here should be the original one from the environment
            # This is important for PPO's value calculation (GAE).
            self.rl_agent.store_transition(state, action.flatten(), log_prob.flatten(), final_reward, done, value)

            # Update the total reward for logging with the final reward
            ep_total_reward += final_reward

            # Now, determine if the episode should be reset
            episode_is_done = done or is_timeout

            state = torch.tensor(next_state, dtype=torch.float32, device=self.rl_agent.device)
            if episode_is_done:
                self.get_logger().info(f"Total T: {t+1} Episode Num: {ep_num} Episode T: {ep_timesteps} Reward: {ep_total_reward:.3f}")
                
                # Add this line to log to csv:
                self.log_training_to_csv(ep_num, ep_timesteps, t+1, ep_total_reward)

                state = self.reset()
                state = torch.tensor(state, dtype=torch.float32, device=self.rl_agent.device)
                ep_total_reward, ep_timesteps = 0, 0
                ep_num += 1

            # --- LEARNING PHASE ---
            # If the rollout buffer is full, trigger the learning update
            if self.rl_agent.mem_idx == self.rl_agent.n_steps:
                self.rl_agent.learn(state.reshape(1,-1), done)
                self.get_logger().info(f"--- Learning Update at Timestep {t} ---")

                # Periodically save the model
                if (t // self.rl_agent.n_steps) % 10 == 0: # Save every 10 updates
                     self.save_models(self.pytorch_models_dir, self.file_name)


    def evaluate_and_print(self, evals, epoch, start_time):
        """Evaluate the agent and print the results"""

        self.get_logger().info(self.border)
        self.get_logger().info(f"| Evaluation at epoch: {epoch}")
        self.get_logger().info(
            f"| Total time passed: {round((time.time()-start_time)/60.,2)} min(s)"
        )

        total_reward = np.zeros(self.eval_eps)
        for ep in range(self.eval_eps):
            state, done = self.reset(), False
            ep_timesteps = 0
            while not done and ep_timesteps < self.max_episode_steps:
                action = self.rl_agent.select_action(
                    np.array(state), self.use_checkpoints, use_exploration=True
                )
                # Act
                state, reward, done, _ = self.step(action)
                total_reward[ep] += reward
                ep_timesteps += 1

        self.get_logger().info(
            f"| Average reward over {self.eval_eps} episodes: {total_reward.mean():.3f}"
        )
        self.get_logger().info(self.border)
        evals.append(total_reward.mean())
        np.save(f"{self.results_dir}/{self.file_name}", evals)

def main(args=None):
    # Initialize the ROS2 communication
    rclpy.init(args=args)
    # Initialize the node
    train_ppo_node = TrainPPO()
    # Start training
    train_ppo_node.train_online()
    try:
        while rclpy.ok() and not train_ppo_node.done_training:
            rclpy.spin_once(train_ppo_node)
    except KeyboardInterrupt as e:
        train_ppo_node.get_logger().warning(f"KeyboardInterrupt: {e}")
    finally:
        train_ppo_node.get_logger().info("rclpy, shutting down...")
        train_ppo_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
