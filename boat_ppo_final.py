import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import random
from collections import deque
from datetime import datetime, timedelta
from scipy.interpolate import LinearNDInterpolator

# Global variables
F_velocity = None
F_SWH = None
F_MWP = None
F_Desal = None

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points on Earth"""
    earth_radius = 6371000  # meters
    
    lat1_rad = np.deg2rad(lat1)
    lon1_rad = np.deg2rad(lon1)
    lat2_rad = np.deg2rad(lat2)
    lon2_rad = np.deg2rad(lon2)
    
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return earth_radius * c

# Environment definition
class WaterPurificationShipEnv:
    def __init__(self, map_bounds=None, start_position=None, 
                 lookup_speed=None, lookup_power=None, white_regions=None, max_steps=8000):
        # Geographic boundaries [lat_min, lat_max, lon_min, lon_max]
        self.map_bounds = map_bounds if map_bounds else [5.0, 8.0, 132.0, 136.0]
        
        # Island regions (white regions)
        self.white_regions = white_regions if white_regions else np.array([
            [7.324, 7.822, 134.324, 134.822]
        ])
        
        # Record the position with the highest power found
        self.highest_power_position = None
        self.highest_power_value = 0
    
        # Starting position
        self.start_position = start_position if start_position else (7.324, 134.739)
        self.position = self.start_position
        
        # Tank fill percentage (0-50%), target is to fill to 50%
        self.tank_level = 0.0
        
        # Speed and power lookup functions
        self.lookup_speed = lookup_speed if lookup_speed else self._default_lookup_speed
        self.lookup_power = lookup_power if lookup_power else self._default_lookup_power
        
        # Conversion factor from watts to liters/hour
        self.conv = 0.055

        # Time step (seconds)
        self.time_step = 200
        
        # Maximum steps and current step count
        self.max_steps = max_steps
        self.steps = 0
        
        # Accumulated simulation time and current time index
        self.simulation_time = 0
        
        # Path record
        self.path = [self.start_position]
        
        # 8 discrete directions (North, Northeast, East, Southeast, South, Southwest, West, Northwest)
        self.directions = [
            (0, -1), (1, -1), (1, 0), (1, 1), 
            (0, 1), (-1, 1), (-1, 0), (-1, -1)
        ]
        
        # Mapping from direction to bearing
        self.direction_to_bearing = {
            (0, -1): 0,    # North
            (1, -1): 45,   # Northeast
            (1, 0): 90,    # East
            (1, 1): 135,   # Southeast
            (0, 1): 180,   # South
            (-1, 1): 225,  # Southwest
            (-1, 0): 270,  # West
            (-1, -1): 315  # Northwest
        }

    def _default_lookup_speed(self, pos):
        """Query speed using interpolation function"""
        lat, lon = pos
        if F_velocity is not None:
            velocity_result = F_velocity(lat, lon)
            velocity = velocity_result
            # Handle NaN values
            if np.isnan(velocity):
                velocity = 0
            return velocity
        else:
            # Simple simulation: closer to map center, faster speed
            center_lat = (self.map_bounds[0] + self.map_bounds[1]) / 2
            center_lon = (self.map_bounds[2] + self.map_bounds[3]) / 2
            dist_to_center = np.sqrt((lat - center_lat)**2 + (lon - center_lon)**2)
            max_dist = np.sqrt((self.map_bounds[1] - self.map_bounds[0])**2/4 + 
                              (self.map_bounds[3] - self.map_bounds[2])**2/4)
            return 1.0 + 2.0 * (1 - dist_to_center / max_dist)

    def _default_lookup_power(self, pos):
        """Query purification power using interpolation function"""
        lat, lon = pos
        if F_Desal is not None:
            desal_result = F_Desal(lat, lon)
            # Apply conversion factor
            desal = desal_result
            power = desal * self.conv
            # Handle NaN values
            if np.isnan(power):
                power = 0
            return power
        else:
            # Simple simulation: different regions have different purification powers
            power = 0.5 + 0.5 * np.sin(lat * 0.1) * np.cos(lon * 0.1)
            return max(0.1, power)  # Ensure at least some power

    def reset(self):
        """Reset environment state"""
        self.position = self.start_position
        self.tank_level = 0.0
        self.highest_power_position = None
        self.highest_power_value = 0
        self.steps = 0
        self.simulation_time = 0
        self.path = [self.start_position]
        return self._get_state()

    def _get_state(self):
        """Get current state representation"""
        lat, lon = self.position
        # Normalize latitude and longitude within map bounds
        norm_lat = (lat - self.map_bounds[0]) / (self.map_bounds[1] - self.map_bounds[0])
        norm_lon = (lon - self.map_bounds[2]) / (self.map_bounds[3] - self.map_bounds[2])
        
        return np.array([
            norm_lat,  # Normalized latitude
            norm_lon,  # Normalized longitude
            self.tank_level / 50.0  # Normalized tank level
        ])

    def _is_collision(self, position):
        """Check if the given position collides with islands"""
        lat, lon = position
        
        # Boundary check
        if lat < self.map_bounds[0] or lat > self.map_bounds[1] or lon < self.map_bounds[2] or lon > self.map_bounds[3]:
            return True
        
        # Island region check (white regions)
        for region in self.white_regions:
            lat_min, lat_max, lon_min, lon_max = region
            if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                return True
        
        return False

    def _calculate_new_position(self, lat1, lon1, bearing, distance_m):
        """Calculate new position given bearing and distance"""
        earth_radius = 6371000  # meters
        angular_distance = distance_m / earth_radius
        
        lat1_rad = np.deg2rad(lat1)
        lon1_rad = np.deg2rad(lon1)
        bearing_rad = np.deg2rad(bearing)
        
        new_lat_rad = np.arcsin(np.sin(lat1_rad) * np.cos(angular_distance) + 
                               np.cos(lat1_rad) * np.sin(angular_distance) * np.cos(bearing_rad))
        new_lon_rad = lon1_rad + np.arctan2(np.sin(bearing_rad) * np.sin(angular_distance) * np.cos(lat1_rad),
                                           np.cos(angular_distance) - np.sin(lat1_rad) * np.sin(new_lat_rad))
        
        new_lat = np.rad2deg(new_lat_rad)
        new_lon = np.rad2deg(new_lon_rad)
        
        return new_lat, new_lon

    def step(self, action):
        self.steps += 1
        old_tank_level = self.tank_level  # Record tank level before update
        old_position = self.position  # Record previous position
        
        # Get direction based on action
        direction = self.directions[action]
        
        # Get speed and power at current position
        old_speed = self.lookup_speed(old_position)
        old_power = self.lookup_power(old_position)
        
        # Calculate new position
        segment_distance = old_speed * self.time_step
        bearing = self.direction_to_bearing[direction]
        new_lat, new_lon = self._calculate_new_position(
            self.position[0], self.position[1], 
            bearing, segment_distance
        )
        new_position = (new_lat, new_lon)
        
        # Check for collision
        if self._is_collision(new_position):
            return self._get_state(), -50000, True, {"collision": True}
        
        # Update position and time index
        self.position = new_position
        self.path.append(new_position)
        
        # Get speed and power at new position
        new_speed = self.lookup_speed(self.position)
        new_power = self.lookup_power(self.position)
        
        # Accumulate time and update time index
        self.simulation_time += self.time_step
        
        # Update tank
        water_increment = new_power * self.time_step / 3600  # Convert to hours
        self.tank_level = min(50.0, self.tank_level + water_increment)
        
        # ----- Optimized reward calculation -----
        
        # 1. Tank fill reward - primary objective
        fill_reward = (self.tank_level - old_tank_level) * 2000
        
        # 2. Power change reward - encourage moving to higher power areas
        power_delta = new_power - old_power
        power_delta_reward = power_delta * 50  # Positive reward for moving to higher power areas
        
        # 3. Power level reward - reward staying in high power areas
        power_level_reward = new_power * 10  # Proportional to current position's power
        
        # 4. Effective movement reward - reduce movement reward if power is already high
        effective_movement_reward = 0
        power_threshold = 30.0  # Assume power > 3.0 is high power area, adjust based on actual data
        
        if new_power < power_threshold:
            # Low power area, encourage fast movement
            effective_movement_reward = new_speed * 5.0
        else:
            # High power area, reduce movement reward, encourage staying
            stay_reward = 100  # Fixed reward for staying in high power area
            effective_movement_reward = stay_reward if power_delta <= 0 else 0
        
        # 5. Time penalty - becomes stricter as steps increase
        time_penalty = -0.5 * (1 + self.steps / 500)
        
        # 6. Progress reward - tank fill milestones
        progress_milestones = [10, 20, 30, 40]
        progress_reward = 0
        for milestone in progress_milestones:
            if old_tank_level < milestone and self.tank_level >= milestone:
                progress_reward += 200
        
        # Calculate total reward
        reward = fill_reward + power_delta_reward + power_level_reward + effective_movement_reward + time_penalty + progress_reward
        
        # Check if task is complete (tank filled to 50%)
        done = False
        if self.tank_level >= 50.0:
            reward += 1000  # Extra reward for task completion
            done = True
        elif self.steps >= self.max_steps:
            # If max steps reached but task not complete, give partial reward
            reward += self.tank_level * 10  # Partial reward based on completion level
            done = True
        
        return self._get_state(), reward, done, {
            "position": self.position, 
            "tank_level": self.tank_level,
            "speed": new_speed,
            "power": new_power
        }

    def render(self):
        """Visualize current path"""
        plt.figure(figsize=(50, 50))
        
        # Draw island regions
        for region in self.white_regions:
            lat_min, lat_max, lon_min, lon_max = region
            rect = plt.Rectangle(
                (lon_min, lat_min), 
                lon_max - lon_min, 
                lat_max - lat_min, 
                color='green', 
                alpha=0.5
            )
            plt.gca().add_patch(rect)
        
        # Draw path
        path_lat = [pos[0] for pos in self.path]
        path_lon = [pos[1] for pos in self.path]
        plt.plot(path_lon, path_lat, 'b-', linewidth=2)
        
        # Mark start and end points
        plt.plot(self.start_position[1], self.start_position[0], 'ro', markersize=10)
        plt.plot(self.position[1], self.position[0], 'go', markersize=8)
        
        # Set plot limits and title
        plt.xlim(self.map_bounds[2], self.map_bounds[3])
        plt.ylim(self.map_bounds[0], self.map_bounds[1])
        plt.title(f'Water Purification Ship Path (Tank: {self.tank_level:.1f}%)')
        plt.grid(True)
        plt.show()

# Define policy network
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.network(x)

# Define value network
class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.network(x)

# PPO algorithm implementation
class PPO:
    def __init__(self, state_dim, action_dim, lr=0.0003, gamma=0.99, epsilon=0.2, 
                 value_coef=0.5, entropy_coef=0.01):
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.value = ValueNetwork(state_dim)
        self.optimizer = optim.Adam(list(self.policy.parameters()) + 
                                    list(self.value.parameters()), lr=lr)
        
        self.memory = []

    def save(self, path):
        """Save model to specified path"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    def load(self, path):
        """Load model from specified path"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value.load_state_dict(checkpoint['value_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def select_action(self, state, training=True):
        state = torch.FloatTensor(state)
        probs = self.policy(state)
        
        if training:
            distribution = Categorical(probs)
            action = distribution.sample()
            return action.item(), distribution.log_prob(action)
        else:
            # In evaluation mode, select action with highest probability
            return torch.argmax(probs).item(), None

    def store_transition(self, state, action, action_log_prob, reward, next_state, done):
        self.memory.append((state, action, action_log_prob, reward, next_state, done))

    def update(self, batch_size=32, epochs=10):
        if len(self.memory) < batch_size:
            return
        
        # Sample from memory
        batch = random.sample(self.memory, batch_size)
        states, actions, old_log_probs, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        # Calculate returns
        returns = []
        for i in range(len(rewards)):
            R = 0
            for j in range(i, len(rewards)):
                R += self.gamma ** (j - i) * rewards[j] * (1 - dones[j])
            returns.append(R)
        returns = torch.FloatTensor(returns)
        
        # Update over multiple epochs
        for _ in range(epochs):
            # Calculate action probabilities under current policy
            probs = self.policy(states)
            dist = Categorical(probs)
            current_log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            
            # Calculate value predictions
            state_values = self.value(states).squeeze()
            
            # Calculate ratio and clipped objective function
            ratios = torch.exp(current_log_probs - old_log_probs)
            advantages = returns - state_values.detach()
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.epsilon, 1+self.epsilon) * advantages
            
            # Calculate total loss
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(state_values, returns)
            entropy_loss = -entropy.mean()
            
            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            # Update network
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # Clear memory
        self.memory = []

# Create instance and train
def load_data_and_create_interpolators():
    """Load data and create interpolation functions"""
    global F_velocity, F_SWH, F_MWP, F_Desal
    
    try:
        data = pd.read_excel('Palau інтерpolated_b7000.xlsx')
        print("Excel data loaded successfully.")
        
        # Extract required data
        latitude = np.round(data['lat'].values, 9)
        longitude = np.round(data['long'].values, 9)
        velocity = np.round(data['Interpolated_TS'].values, 9)
        SWH = np.round(data['SWH1'].values, 9)
        MWP = np.round(data['MWP1'].values, 9)
        desalination = np.round(data['Interpolated_Power'].values, 9)
        time_index = data['index'].values
        
        # Create interpolation functions
        points = np.column_stack((latitude, longitude, time_index))
        F_velocity = LinearNDInterpolator(points, velocity, fill_value=np.nan)
        F_SWH = LinearNDInterpolator(points, SWH, fill_value=np.nan)
        F_MWP = LinearNDInterpolator(points, MWP, fill_value=np.nan)
        F_Desal = LinearNDInterpolator(points, desalination, fill_value=np.nan)
        
        return True
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Using default lookup functions instead")
        return False

def visualize_speed_power(env, resolution=50):
    """Visualize speed and power distribution"""
    lat_min, lat_max, lon_min, lon_max = env.map_bounds
    
    lat = np.linspace(lat_min, lat_max, resolution)
    lon = np.linspace(lon_min, lon_max, resolution)
    LAT, LON = np.meshgrid(lat, lon)
    
    speed = np.zeros((resolution, resolution))
    power = np.zeros((resolution, resolution))
    
    for i in range(resolution):
        for j in range(resolution):
            pos = (LAT[i, j], LON[i, j])
            if not env._is_collision(pos):
                speed[i, j] = env.lookup_speed(pos)
                power[i, j] = env.lookup_power(pos)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.contourf(LON, LAT, speed, cmap='viridis')
    plt.colorbar(label='Speed (m/s)')
    plt.title('Speed Distribution')
    
    plt.subplot(1, 2, 2)
    plt.contourf(LON, LAT, power, cmap='plasma')
    plt.colorbar(label='Desalination Power (L/hr)')
    plt.title('Desalination Power Distribution')
    
    # Add islands
    for ax in plt.gcf().get_axes()[:2]:  # Only get first two subplots
        plt.sca(ax)
        for region in env.white_regions:
            lat_min, lat_max, lon_min, lon_max = region
            rect = plt.Rectangle(
                (lon_min, lat_min), 
                lon_max - lon_min, 
                lat_max - lat_min, 
                color='green', 
                alpha=0.3,
                label='Island'
            )
            plt.gca().add_patch(rect)
    
    plt.tight_layout()
    plt.show()

def train_agent(env, num_episodes=100000, batch_size=32, render_interval=1000):
    """Train the agent"""
    state_dim = 3  # lat, lon, tank_level
    action_dim = 8  # 8 directions
    
    agent = PPO(state_dim, action_dim)
    
    total_rewards = []
    best_reward = -float('inf')
    best_path = None
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.store_transition(state, action, log_prob, reward, next_state, done)
            
            state = next_state
            total_reward += reward
        
        # Update every batch_size episodes
        if (episode + 1) % batch_size == 0:
            agent.update(batch_size)
        
        total_rewards.append(total_reward)
        
        # Save best path
        if total_reward > best_reward:
            best_reward = total_reward
            best_path = env.path.copy()
        
        # Print training progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(total_rewards[-10:])
            print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}, Best Reward: {best_reward:.2f}")
        
        # Render path periodically
        if (episode + 1) % render_interval == 0:
            env.path = best_path  # Show best path
            # env.render()
            save_path = '/home/zihanyu/BO/'
            save_name = save_path + str(episode) + '.pth'
            agent.save(save_name)
    
    # Visualize reward trend after training
    # plt.figure(figsize=(10, 5))
    # plt.plot(total_rewards)
    # plt.title('Training Rewards')
    # plt.xlabel('Episode')
    # plt.ylabel('Total Reward')
    # plt.grid(True)
    # plt.show()
    
    print("Saving model...")
    agent.save('ppo_water_purification_model.pth')
    
    return agent, best_path

def evaluate_agent(env, agent, num_episodes=1):
    """Evaluate agent performance"""
    all_rewards = []
    all_steps = []
    all_tank_levels = []
    
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _ = agent.select_action(state, training=False)
            next_state, reward, done, info = env.step(action)
            
            state = next_state
            total_reward += reward
        
        all_rewards.append(total_reward)
        all_steps.append(env.steps)
        all_tank_levels.append(env.tank_level)
        
        # Render path of the last evaluation
        env.render()
    
    print(f"Evaluation over {num_episodes} episodes:")
    print(f"Average Reward: {np.mean(all_rewards):.2f}")
    print(f"Average Steps: {np.mean(all_steps):.2f}")
    print(f"Average Final Tank Level: {np.mean(all_tank_levels):.2f}%")
    
    return env.path

def calculate_full_path_metrics(test_env, outbound_path):
    """Calculate metrics for complete path (outbound + return)"""
    # Copy environment to avoid interfering with current environment
    # test_env = WaterPurificationShipEnv(
    #     map_bounds=env.map_bounds,
    #     start_position=env.start_position,
    #     white_regions=env.white_regions,
    #     lookup_speed=env.lookup_speed,
    #     lookup_power=env.lookup_power
    # )
    
    # Calculate outbound metrics
    outbound_distance = 0
    outbound_time = 0
    outbound_desalination = 0
    
    # Move along outbound path
    test_env.position = outbound_path[0]
    for i in range(1, len(outbound_path)):
        prev_pos = outbound_path[i-1]
        curr_pos = outbound_path[i]
        
        # Calculate distance between points
        segment_distance = calculate_distance(prev_pos[0], prev_pos[1], curr_pos[0], curr_pos[1])
        
        # Get speed and power at current point
        speed = test_env.lookup_speed(prev_pos)
        power = test_env.lookup_power(prev_pos)
        
        # Calculate time required
        segment_time = segment_distance / speed if speed > 0 else 0
        
        # Accumulate metrics
        outbound_distance += segment_distance
        outbound_time += segment_time
        outbound_desalination += power * segment_time / 3600  # Convert to hours
    
    # Calculate return metrics (return along same path)
    return_path = outbound_path[::-1]
    return_distance = outbound_distance
    return_time = 0
    
    # Move along return path
    for i in range(1, len(return_path)):
        prev_pos = return_path[i-1]
        curr_pos = return_path[i]
        
        # Calculate distance between points
        segment_distance = calculate_distance(prev_pos[0], prev_pos[1], curr_pos[0], curr_pos[1])
        
        # Get speed at current point
        speed = test_env.lookup_speed(prev_pos)
        
        # Calculate time required
        segment_time = segment_distance / speed if speed > 0 else 0
        
        # Accumulate metrics
        return_time += segment_time
    
    # Total metrics
    total_distance = outbound_distance + return_distance
    total_time = outbound_time + return_time
    
    print("\nComplete Path Metrics (Outbound + Return):")
    print(f"Total Distance: {total_distance/1000:.2f} km")
    print(f"Total Time: {total_time/3600:.2f} hours")
    print(f"Outbound Desalination: {outbound_desalination:.2f} L")
    print(f"Average Speed: {total_distance/total_time:.2f} m/s")
    
    return {
        'total_distance': total_distance,
        'total_time': total_time,
        'outbound_desalination': outbound_desalination
    }

def visualize_complete_path(env, outbound_path):
    """Visualize complete path (outbound + return)"""
    plt.figure(figsize=(10, 10))
    
    # Draw island regions
    for region in env.white_regions:
        lat_min, lat_max, lon_min, lon_max = region
        rect = plt.Rectangle(
            (lon_min, lat_min), 
            lon_max - lon_min, 
            lat_max - lat_min, 
            color='green', 
            alpha=0.5,
            label='Island'
        )
        plt.gca().add_patch(rect)
    
    # Draw outbound path
    outbound_lat = [pos[0] for pos in outbound_path]
    outbound_lon = [pos[1] for pos in outbound_path]
    plt.plot(outbound_lon, outbound_lat, 'b-', linewidth=2, label='Outbound')
    
    # Draw return path
    return_lat = [pos[0] for pos in outbound_path[::-1]]
    return_lon = [pos[1] for pos in outbound_path[::-1]]
    plt.plot(return_lon, return_lat, 'r--', linewidth=2, label='Return')
    
    # Mark start and end points
    plt.plot(env.start_position[1], env.start_position[0], 'ko', markersize=10, label='Start/End')
    plt.plot(outbound_path[-1][1], outbound_path[-1][0], 'go', markersize=8, label='Half-Full Point')
    
    # Set plot limits and title
    plt.xlim(env.map_bounds[2], env.map_bounds[3])
    plt.ylim(env.map_bounds[0], env.map_bounds[1])
    plt.title('Complete Path: Outbound and Return')
    plt.grid(True)
    plt.legend()
    plt.savefig('result_2.jpg')
    plt.show()
    

def create_map_with_islands(map_size=(50, 50), num_islands=5, min_radius=2, max_radius=5):
    """Create map with islands"""
    islands = []
    for _ in range(num_islands):
        x = random.uniform(max_radius, map_size[0] - max_radius)
        y = random.uniform(max_radius, map_size[1] - max_radius)
        radius = random.uniform(min_radius, max_radius)
        islands.append((x, y, radius))
    return islands

# Main function: set environment parameters and run training
def main(mode='train'):
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Load data and create interpolation functions
    data = pd.read_excel('Palau_interpolated_b7000.xlsx')
    # Extract required data
    latitude = np.round(data['lat'].values, 9)
    longitude = np.round(data['long'].values, 9)
    velocity = np.round(data['Interpolated_TS'].values, 9)
    desalination = np.round(data['Interpolated_Power'].values, 9)
    
    # Create interpolation functions
    points = np.column_stack((latitude, longitude))
    global F_velocity, F_Desal
    F_velocity = LinearNDInterpolator(points, velocity, fill_value=np.nan)
    F_Desal = LinearNDInterpolator(points, desalination, fill_value=np.nan)
    
    env = WaterPurificationShipEnv()
    
    if mode == 'train':
        # Visualize speed and power distribution
        visualize_speed_power(env)
        
        # Train agent
        print("Starting training...")
        agent, best_path = train_agent(env, num_episodes=300000, render_interval=3000)
    elif mode == 'evaluate':
        # Evaluate agent
        evaluate()
    
def evaluate():
    print("Evaluating agent...")
    # Load data and create interpolation functions
    data = pd.read_excel('/home/zihanyu/BO/Palau_interpolated_b7000.xlsx')
    # Extract required data
    latitude = np.round(data['lat'].values, 9)
    longitude = np.round(data['long'].values, 9)
    velocity = np.round(data['Interpolated_TS'].values, 9)
    desalination = np.round(data['Interpolated_Power'].values, 9)
    
    # Create interpolation functions
    points = np.column_stack((latitude, longitude))
    global F_velocity, F_Desal
    F_velocity = LinearNDInterpolator(points, velocity, fill_value=np.nan)
    F_Desal = LinearNDInterpolator(points, desalination, fill_value=np.nan)
    
    env = WaterPurificationShipEnv()
    state_dim = 3  # lat, lon, tank_level
    action_dim = 8  # 8 directions
    
    agent = PPO(state_dim, action_dim)
    agent.load('194999.pth')
    
    final_path = evaluate_agent(env, agent)
    
    # Calculate and visualize complete path metrics
    metrics = calculate_full_path_metrics(env, final_path)
    visualize_complete_path(env, final_path)
    
    return final_path, metrics

if __name__ == "__main__":
    main(mode='train')