#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import math
import threading
import random
import time
import numpy as np
from collections import deque
from squaternion import Quaternion

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from gazebo_msgs.msg import EntityState

from std_srvs.srv import Empty
from gazebo_msgs.srv import SetEntityState
from drl_agent_interfaces.srv import Step, Reset, Seed, GetDimensions, SampleActionSpace

from sensor_msgs.msg import LaserScan  # Add at the top

# import point_cloud2 as pc2
from file_manager import load_yaml


class Environment(Node):
    """Environment Node for providing services required for DRL.

    This class provides functionalities to interact with an environment through ROS2 services.
    The services include:
    - step: Take an action and get the resulting situation from the environment.
    - reset: Reset the environment and get initial observation.
    - get_dimensions: Get the dimensions of the state, action, and maximum action value.
    """

    def __init__(self):
        super().__init__("gym_node")

        # Determine if the environment is to be run in training or testing mode
        self.declare_parameter("environment_mode", "train")
        self.environment_mode = (
            self.get_parameter("environment_mode")
            .get_parameter_value()
            .string_value.lower()
        )
        if not self.environment_mode in ["train", "test", "random_test"]:
            raise NotImplementedError
        # Environment run mode
        self.train_mode = (
            self.environment_mode == "train" or self.environment_mode == "random_test"
        )
        self.get_logger().info(f"Environment run mode: {self.environment_mode}")

        # Load environment config file
        drl_agent_src_path_env = "DRL_AGENT_SRC_PATH"
        drl_agent_src_path = os.getenv(drl_agent_src_path_env)
        if drl_agent_src_path is None:
            self.get_logger().error(
                f"Environment variable: {drl_agent_src_path_env} is not set"
            )
            sys.exit(-1)
        env_config_file_name = "environment.yaml"
        start_goal_pairs_file = "test_config.yaml"
        env_config_file_path = os.path.join(
            drl_agent_src_path, "drl_agent", "config", env_config_file_name
        )
        start_goal_pairs_file_path = os.path.join(
            drl_agent_src_path, "drl_agent", "config", start_goal_pairs_file
        )
        # Define the dimensions of the state, action, and maximum action value
        try:
            self.config = load_yaml(env_config_file_path)
        except Exception as e:
            self.get_logger().info(f"Unable to load config file: {e}")
            sys.exit(-1)
        self.environment_config = self.config["environment"]
        self.lower = self.environment_config["lower"]
        self.upper = self.environment_config["upper"]
        self.environment_dim = self.environment_config["environment_state_dim"]
        self.agent_dim = self.environment_config["agent_state_dim"]
        self.agent_name = self.environment_config["agent_name"]
        self.num_of_obstacles = self.environment_config["num_of_obstacles"]

        self.action_dim = self.environment_config["action_dim"]
        self.max_action = self.environment_config["max_action"]
        self.actions_low = self.environment_config["actions_low"]
        self.actions_high = self.environment_config["actions_high"]

        self.threshold_params_config = self.config["threshold_parameters"]
        self.goal_threshold = self.threshold_params_config["goal_threshold"]
        self.collision_threshold = self.threshold_params_config["collision_threshold"]
        self.time_delta = self.threshold_params_config["time_delta"]
        self.inter_entity_distance = self.threshold_params_config[
            "inter_entity_distance"
        ]

        self.lidar_max_range = self.threshold_params_config["lidar_max_range"]

        # Callback groups for handling sensors and services in parallel
        self.odom_callback_group = MutuallyExclusiveCallbackGroup()
        self.velodyne_callback_group = MutuallyExclusiveCallbackGroup()
        self.clients_callback_group = MutuallyExclusiveCallbackGroup()

        # Initialize publishers
        self.velocity_publisher = self.create_publisher(Twist, "/cmd_vel", 10)
        self.goal_point_marker_pub = self.create_publisher(
            MarkerArray, "goal_point", 10
        )
        self.linear_vel_marker_pub = self.create_publisher(
            MarkerArray, "linear_velocity", 10
        )
        self.angular_vel_marker_pub = self.create_publisher(
            MarkerArray, "angular_velocity", 10
        )

        # Create services
        self.srv_seed = self.create_service(Seed, "seed", self.seed_callback)
        self.srv_step = self.create_service(Step, "step", self.step_callback)
        self.srv_reset = self.create_service(Reset, "reset", self.reset_callback)
        self.srv_dimentions = self.create_service(
            GetDimensions, "get_dimensions", self.get_dimensions_callback
        )
        self.srv_action_space_sample = self.create_service(
            SampleActionSpace, "action_space_sample", self.sample_action_callback
        )
        # Initialize clients
        self.unpause = self.create_client(
            Empty, "/unpause_physics", callback_group=self.clients_callback_group
        )
        self.pause = self.create_client(
            Empty, "/pause_physics", callback_group=self.clients_callback_group
        )
        self.reset_proxy = self.create_client(
            Empty, "/reset_world", callback_group=self.clients_callback_group
        )
        self.set_model_state = self.create_client(
            SetEntityState,
            "gazebo/set_entity_state",
            callback_group=self.clients_callback_group,
        )
        # Sensor subscriptions QoS
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        # Odometry subscription
        self.odom = self.create_subscription(
            Odometry,
            "/odom",
            self.update_agent_state,
            qos_profile,
            callback_group=self.odom_callback_group,
        )
        self.odom
        # Velodyne subscription
        # self.velodyne = self.create_subscription(
        #     PointCloud2,
        #     "/velodyne_points",
        #     self.update_environment_state,
        #     qos_profile,
        #     callback_group=self.velodyne_callback_group,
        # )
        # self.velodyne
        
        #For 2D Lidar
        self.laser_sub = self.create_subscription(
            LaserScan,
            "/scan",  # Change this if your LiDAR publishes on a different topic
            self.update_environment_state_laserscan,
            qos_profile,
            callback_group=self.velodyne_callback_group,
        )
        self.laser_sub
        # Define bins for grouping the velodyne_points
        self.bins = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / self.environment_dim]]
        for m in range(self.environment_dim - 1):
            self.bins.append(
                [self.bins[m][1], self.bins[m][1] + np.pi / self.environment_dim]
            )
        self.bins[-1][-1] += 0.03

        # Initialize commands
        self.velocity_command = Twist()
        self.set_agent_state = EntityState()
        self.set_agent_state.name = self.agent_name
        self.set_obstacle_state = EntityState()
        # Command requests
        self.set_agent_state_req = SetEntityState.Request()
        self.set_static_obs_state_req = SetEntityState.Request()

        # Initialize environment and agent state
        self.environment_state = None
        self.agent_state = None
        # Initialize lock to protect environment_state and agent sate from race condition
        self.environment_state_lock = threading.Lock()
        self.agent_state_lock = threading.Lock()

        # Load start-goal pairs
        if not self.train_mode:
            try:
                self.start_goal_pairs = deque(
                    load_yaml(start_goal_pairs_file_path)["start_goal_pairs"]
                )
            except Exception as e:
                self.get_logger().error(f"Unable to load start-goal pairs: {e}")
                sys.exit(-1)
            self.current_pairs = None

        # Define initial goal pos
        self.goal_x = 0.0
        self.goal_y = 0.0

    def terminate_session(self):
        """Destroy the node and shut down rclpy when done"""
        self.get_logger().info("gym_node shutting down...")
        self.destroy_node()

    def seed_callback(self, request, response):
        """Sets environment seed for reproducibility of the training process."""
        np.random.seed(request.seed)
        response.success = True
        return response

    def sample_action_callback(self, _, response):
        """Samples an action from the action space."""
        action = np.random.uniform(self.actions_low, self.actions_high)
        response.action = np.array(action, dtype=np.float32).tolist()
        return response

    def get_dimensions_callback(self, _, response):
        """Returns the dimensions of the state, action, and maximum action value"""
        response.state_dim = self.environment_dim + self.agent_dim
        response.action_dim = self.action_dim
        response.max_action = self.max_action
        return response

    # def update_environment_state(self, velodyne_data):
    #     """Updates environment state using pointcloud data from velodyne sensor

    #     Reads velodyne point cloud data, converts it into distance data, and
    #     selects the minimum value for each angle range as a state representation.
    #     """
    #     with self.environment_state_lock:
    #         self.environment_state = (
    #             np.ones(self.environment_dim) * self.lidar_max_range
    #         )
    #         data = list(
    #             pc2.read_points(
    #                 velodyne_data, skip_nans=False, field_names=("x", "y", "z")
    #             )
    #         )
    #         for i in range(len(data)):
    #             if data[i][2] > -0.2:
    #                 dot = data[i][0] * 1 + data[i][1] * 0
    #                 mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
    #                 mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
    #                 beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])
    #                 dist = math.sqrt(
    #                     data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2
    #                 )

    #                 for j in range(len(self.bins)):
    #                     if self.bins[j][0] <= beta < self.bins[j][1]:
    #                         self.environment_state[j] = min(
    #                             self.environment_state[j], dist
    #                         )
    #                         break
    
    def update_environment_state_laserscan(self, scan_msg):
        """Updates environment state using 2D LiDAR scan ranges."""
        with self.environment_state_lock:
            self.environment_state = np.ones(self.environment_dim) * self.lidar_max_range

            # Extract valid ranges
            valid_ranges = np.array(scan_msg.ranges)
            valid_ranges = np.clip(valid_ranges, 0, self.lidar_max_range)  # Sanitize

            # Split into bins
            num_ranges = len(valid_ranges)
            step = num_ranges // self.environment_dim
            for i in range(self.environment_dim):
                start = i * step
                end = min(start + step, num_ranges)
                if end > start:
                    self.environment_state[i] = np.min(valid_ranges[start:end])


    def get_environment_state(self):
        """Returns a copy of the environment state"""
        with self.environment_state_lock:
            return self.environment_state.copy()

    # ADDITIONAL HELPER METHODS FOR ENHANCED TRACKING
    def update_agent_state(self, odom):
        """Enhanced agent state update with better tracking"""
        with self.agent_state_lock:
            # Calculate robot heading from odometry data
            odom_x = odom.pose.pose.position.x
            odom_y = odom.pose.pose.position.y
            quaternion = Quaternion(
                odom.pose.pose.orientation.w,
                odom.pose.pose.orientation.x,
                odom.pose.pose.orientation.y,
                odom.pose.pose.orientation.z,
            )
            euler = quaternion.to_euler(degrees=False)
            angle = round(euler[2], 4)

            # Calculate distance to the goal from the robot
            distance = np.linalg.norm([odom_x - self.goal_x, odom_y - self.goal_y])

            # Calculate the relative angle between robot's heading and goal direction
            skew_x = self.goal_x - odom_x
            skew_y = self.goal_y - odom_y
            dot = skew_x * 1 + skew_y * 0
            mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
            mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
            beta = math.acos(dot / (mag1 * mag2))
            if skew_y < 0:
                if skew_x < 0:
                    beta = - beta
                else:
                    beta = 0 - beta
            theta = beta - angle

            # Normalize angle to [-pi, pi]
            if theta > np.pi:
                theta = theta - 2 * np.pi
            if theta < -np.pi:
                theta = theta + 2 * np.pi

            # Store enhanced state: [distance, angle_to_goal, linear_vel, angular_vel]
            self.agent_state = np.array([distance, theta, 0, 0])

            # Optional: Track position history for debugging
            if not hasattr(self, 'position_history'):
                self.position_history = []
            self.position_history.append((odom_x, odom_y, distance))
            if len(self.position_history) > 100:  # Keep last 100 positions
                self.position_history.pop(0)
    
    # DEBUGGING FUNCTION TO MONITOR TRAINING
    def log_reward_breakdown(self, reward_components):
        """Log individual reward components for debugging"""
        if hasattr(self, 'reward_log_counter'):
            self.reward_log_counter += 1
        else:
            self.reward_log_counter = 1

        # Log every 100 steps
        if self.reward_log_counter % 100 == 0:
            self.get_logger().info(f"Reward Breakdown: {reward_components}")

    def get_agent_state(self):
        """Return a copy of the agent state"""
        with self.agent_state_lock:
            return self.agent_state.copy()

    def set_gazebo_model_state(self, model_state):
        """Chage the position of gazebo model"""
        while not self.set_model_state.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                "Service /gazebo/set_entity_state not available, waiting again..."
            )
        try:
            self.set_model_state.call_async(model_state)
        except Exception as e:
            self.get_logger().error(
                "/gazebo/set_entity_state service call failed: %s" % str(e)
            )
            sys.exit(-1)

    def propagate_state(self, time_delta):
        """Propagate the state of the environment for time_delata secons"""
        while not self.unpause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                "Service /unpause_physics not available, waiting again..."
            )
        try:
            self.unpause.call_async(Empty.Request())
        except Exception as e:
            self.get_logger().error("/unpause_physics service call failed: %s" % str(e))
            sys.exit(-1)
        # propagate state for time_delta seconds
        time.sleep(time_delta)
        while not self.pause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                "Service /pause_physics not available, waiting again..."
            )
        try:
            self.pause.call_async(Empty.Request())
        except Exception as e:
            self.get_logger().error("/pause_physics service call failed: %s" % str(e))
            sys.exit(-1)

    def step_callback(self, request, response):
        """Enhanced step callback that tracks progress for reward calculation"""
        target = False
        action = request.action

        # Store previous distance for progress calculation
        prev_distance = self.get_agent_state()[0] if self.agent_state is not None else None

        # Send velocity command
        self.velocity_command.linear.x = action[0]
        self.velocity_command.angular.z = action[1]
        self.velocity_publisher.publish(self.velocity_command)
        self.publish_markers(action)

        # Propagate the state for time_delta secs
        self.propagate_state(self.time_delta)

        # Compute state
        environment_state = self.get_environment_state()
        agent_state = self.get_agent_state()
        agent_state[2], agent_state[3] = action[0], action[1]
        state = np.append(environment_state, agent_state)

        # Get current distance and angle to goal
        current_distance = agent_state[0]
        goal_angle = agent_state[1]

        # Compute reward with progress tracking
        done, collision, min_laser = self.check_collision(environment_state)
        if current_distance < self.goal_threshold:
            self.get_logger().info(f"{'GOAL REACHED':-^50}")
            target = True
            done = True

        # Calculate reward with enhanced information
        reward = self.get_reward(
            target=target,
            collision=collision,
            action=action,
            min_laser=min_laser,
            distance_to_goal=current_distance,
            prev_distance_to_goal=prev_distance,
            robot_heading=None,  # Could add if needed
            goal_angle=goal_angle
        )

        # Formulate response
        response.state = state.tolist()
        response.reward = reward
        response.done = done
        response.target = target
        return response


    def reset_callback(self, _, response):
        """Resets the state of the environment and returns an initial observation, state"""

        """*****************************************************
		** Start by reseting the world
		*****************************************************"""
        while not self.reset_proxy.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                "Service /reset_world not available, waiting again..."
            )
        try:
            self.reset_proxy.call_async(Empty.Request())
        except Exception as e:
            self.get_logger().error("/reset_world service call failed: %s" % str(e))
            sys.exit(-1)
        time.sleep(self.time_delta)

        """*****************************************************
		** Determine start positions for the agent
		*****************************************************"""
        if self.train_mode:
            position_ok = False
            angle = np.random.uniform(-np.pi, np.pi)
            while not position_ok:
                start_x = np.random.uniform(self.lower, self.upper)
                start_y = np.random.uniform(self.lower, self.upper)
                position_ok = not self.check_dead_zone(start_x, start_y)
        else:
            if not self.start_goal_pairs:
                self.get_logger().info(f"{'All start-goal pairs are visited':-^50}")
                self.terminate_session()
            self.current_pairs = self.start_goal_pairs.popleft()
            start_x = self.current_pairs["start"]["x"]
            start_y = self.current_pairs["start"]["y"]
            angle = self.current_pairs["start"]["theta"]

        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        self.set_agent_state.pose.position.x = start_x
        self.set_agent_state.pose.position.y = start_y
        self.set_agent_state.pose.position.z = 0.0
        self.set_agent_state.pose.orientation.x = quaternion.x
        self.set_agent_state.pose.orientation.y = quaternion.y
        self.set_agent_state.pose.orientation.z = quaternion.z
        self.set_agent_state.pose.orientation.w = quaternion.w

        self.set_agent_state_req.state = self.set_agent_state
        # Set agent state
        self.set_gazebo_model_state(self.set_agent_state_req)

        """*****************************************************
		** Change goal and randomize obstacles
		*****************************************************"""
        self.change_goal()
        if self.train_mode:
            self.shuffle_obstacles(start_x, start_y)
        # Publish markers for rviz
        self.publish_markers([0.0, 0.0])
        # Propagate state for 2*time_delta seconds
        self.propagate_state(2 * self.time_delta)

        """*****************************************************
		** Compute state after reset
		*****************************************************"""
        environment_state = self.get_environment_state()
        agent_state = self.get_agent_state()
        response.state = np.append(environment_state, agent_state).tolist()
        return response

    def change_goal(self):
        """Places a new goal and ensures its location is not on one of the obstacles"""
        if self.train_mode:
            goal_ok = False
            while not goal_ok:
                self.goal_x = random.uniform(self.upper, self.lower)
                self.goal_y = random.uniform(self.upper, self.lower)
                goal_ok = not self.check_dead_zone(self.goal_x, self.goal_y)
        else:
            self.goal_x = self.current_pairs["goal"]["x"]
            self.goal_y = self.current_pairs["goal"]["y"]

    def check_collision(self, laser_data):
        """Detect a collision from laser data"""
        done, collision = False, False
        min_laser = min(laser_data)
        if min_laser < self.collision_threshold:
            done, collision = True, True
        return done, collision, min_laser

    def shuffle_obstacles(self, start_x, start_y):
        """Randomly changes the location of the obstacles upon reset"""
        prev_obstacle_positions = []
        for i in range(1, self.num_of_obstacles + 1):
            position_ok = False
            self.set_obstacle_state.name = "obstacle_" + str(i)
            while not position_ok:
                x = np.random.uniform(self.lower, self.upper)
                y = np.random.uniform(self.lower, self.upper)

                position_ok = not self.check_dead_zone(x, y)
                distance_to_robot = np.linalg.norm([x - start_x, y - start_y])
                distance_to_goal = np.linalg.norm([x - self.goal_x, y - self.goal_y])
                if (
                    distance_to_robot < self.inter_entity_distance
                    or distance_to_goal < self.inter_entity_distance
                ):
                    position_ok = False
                    continue

                for prev_x, prev_y in prev_obstacle_positions:
                    distance_to_other_obstacles = np.linalg.norm(
                        [x - prev_x, y - prev_y]
                    )
                    if distance_to_other_obstacles < self.inter_entity_distance:
                        position_ok = False

            self.set_obstacle_state.pose.position.x = x
            self.set_obstacle_state.pose.position.y = y
            self.set_obstacle_state.pose.position.z = 0.0
            self.set_obstacle_state.pose.orientation.x = 0.0
            self.set_obstacle_state.pose.orientation.y = 0.0
            self.set_obstacle_state.pose.orientation.z = 0.0
            self.set_obstacle_state.pose.orientation.w = 1.0
            # Set obstacle state
            self.set_static_obs_state_req.state = self.set_obstacle_state
            self.set_gazebo_model_state(self.set_static_obs_state_req)
            prev_obstacle_positions.append((x, y))

    def check_dead_zone(self, x, y):
        """Check if (x, y) is in occupied space"""
        dead_zone = False
        if abs(x) > self.upper or abs(y) > self.upper:
            dead_zone = True
        elif 2.0 < abs(x) < self.upper and abs(y) < 1.0:
            dead_zone = True
        elif abs(x) < 1.0 and 2.0 < abs(y) < self.upper:
            dead_zone = True
        return dead_zone

    def publish_markers(self, action):
        """Publishes visual data for Rviz to visualize the goal and the robot's actions"""
        marker_specs = [
            {
                "frame_id": "odom",
                "marker_type": Marker.CYLINDER,
                "scale": (0.1, 0.1, 0.01),
                "color": (1.0, 0.0, 1.0, 0.0),
                "position": (self.goal_x, self.goal_y, 0.0),
                "orientation": (0.0, 0.0, 0.0, 1.0),
                "action": Marker.ADD,
                "ns": "",
                "marker_id": 0,
                "publisher": self.goal_point_marker_pub,
            },
            {
                "frame_id": "odom",
                "marker_type": Marker.CUBE,
                "scale": (abs(action[0]), 0.1, 0.01),
                "color": (1.0, 1.0, 0.0, 0.0),
                "position": (5.0, 0.0, 0.0),
                "orientation": (0.0, 0.0, 0.0, 1.0),
                "action": Marker.ADD,
                "ns": "",
                "marker_id": 1,
                "publisher": self.linear_vel_marker_pub,
            },
            {
                "frame_id": "odom",
                "marker_type": Marker.CUBE,
                "scale": (abs(action[1]), 0.1, 0.01),
                "color": (1.0, 1.0, 0.0, 0.0),
                "position": (5.0, 0.2, 0.0),
                "orientation": (0.0, 0.0, 0.0, 1.0),
                "action": Marker.ADD,
                "ns": "",
                "marker_id": 2,
                "publisher": self.angular_vel_marker_pub,
            },
        ]
        for spec in marker_specs:
            marker = self.create_marker(**spec)
            marker_array = MarkerArray()
            marker_array.markers.append(marker)
            spec["publisher"].publish(marker_array)

    @staticmethod
    def create_marker(**kwargs):
        """Create marker to be published for visualization"""
        marker = Marker()
        marker.ns = kwargs.get("ns", "")
        marker.id = kwargs.get("marker_id", 0)
        marker.header.frame_id = kwargs.get("frame_id", "odom")
        marker.type = kwargs.get("marker_type", Marker.CYLINDER)
        marker.action = kwargs.get("action", Marker.ADD)
        marker.scale.x, marker.scale.y, marker.scale.z = kwargs.get(
            "scale", (0.1, 0.1, 0.01)
        )
        marker.color.a, marker.color.r, marker.color.g, marker.color.b = kwargs.get(
            "color", (1.0, 0.0, 1.0, 0.0)
        )
        (
            marker.pose.position.x,
            marker.pose.position.y,
            marker.pose.position.z,
        ) = kwargs.get("position", (0.0, 0.0, 0.0))
        (
            marker.pose.orientation.x,
            marker.pose.orientation.y,
            marker.pose.orientation.z,
            marker.pose.orientation.w,
        ) = kwargs.get("orientation", (0.0, 0.0, 0.0, 1.0))
        return marker

    @staticmethod
    def get_reward(target, collision, action, min_laser, distance_to_goal, prev_distance_to_goal, robot_heading, goal_angle):
        """
        Calculate reward based on goal progress, obstacle avoidance, and movement efficiency
    
        Args:
            target: Boolean, whether goal is reached
            collision: Boolean, whether collision occurred
            action: [linear_vel, angular_vel]
            min_laser: Minimum distance from LiDAR readings
            distance_to_goal: Current distance to goal
            prev_distance_to_goal: Previous distance to goal
            robot_heading: Current robot heading (radians)
            goal_angle: Angle to goal relative to robot heading (radians)
        """
    
        # 1. TERMINAL REWARDS (highest priority)
        if target:
            return 100.0  
        if collision:
            return -100.0 
    
        # 2. PROGRESS REWARD (encourage moving toward goal)
        progress_reward = 0.0
        if prev_distance_to_goal is not None:
            progress = prev_distance_to_goal - distance_to_goal
            progress_reward = progress * 10.0  # Scale factor
    
        # 3. DISTANCE REWARD (encourage being closer to goal)
        max_distance = 10.0  
        distance_reward = (max_distance - distance_to_goal) / max_distance
    
        # 4. HEADING REWARD (encourage pointing toward goal)
        heading_reward = 0.0
        if goal_angle is not None:
            alignment = np.cos(goal_angle)
            heading_reward = alignment * 0.5 
    
        # 5. OBSTACLE AVOIDANCE REWARD
        obstacle_reward = 0.0
        if min_laser < 1.0:
            obstacle_reward = (min_laser - 1.0) * 2.0  
    
        # 6. ACTION EFFICIENCY REWARD Encourage forward movement, discourage excessive turning
        forward_reward = action[0] * 0.5  
        turning_penalty = -abs(action[1]) * 0.3 
    
        # 7. STILLNESS PENALTY (prevent getting stuck)
        stillness_penalty = 0.0
        if abs(action[0]) < 0.05 and abs(action[1]) < 0.05:
            stillness_penalty = -0.5  
    
        # 8. WALL HUGGING PENALTY (prevent following walls instead of going to goal)
        wall_following_penalty = 0.0
        if min_laser < 0.5 and abs(action[1]) > 0.1: 
            if np.sign(action[1]) == np.sign(goal_angle): 
                wall_following_penalty = -0.3
    
        # COMBINE ALL REWARDS
        total_reward = (
            progress_reward +           # Most important: progress toward goal
            distance_reward * 0.1 +     # Small bonus for being close
            heading_reward +            # Bonus for facing goal
            obstacle_reward +           # Penalty for being close to obstacles
            forward_reward +            # Encourage forward movement
            turning_penalty +           # Discourage excessive turning
            stillness_penalty +         # Prevent getting stuck
            wall_following_penalty -    # Prevent wall following
            0.01                        # Small time penalty (encourage efficiency)
        )
    
        return total_reward


def main(args=None):
    # Initialize the ROS2 communication
    rclpy.init(args=args)
    # Create the environment node
    environment = Environment()
    # Use MultiThreadedExecutor to handle the two sensor callbacks in parallel.
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(environment)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        environment.get_logger().info("gym_node, shutting down...")
        environment.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
