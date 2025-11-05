import numpy as np
from typing import Tuple, Dict, List
import random

class GridWorld:
    """
    A grid world environment with stochastic transitions and terminal states.
    """
    
    def __init__(self):
        self.grid = np.array([
            [0, -1, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0],  
            [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10]   
        ])
        
        self.rows, self.cols = self.grid.shape
        self.terminal_states = self._find_terminal_states()
        
        # Constants
        self.GAMMA = 0.9
        self.LIVING_REWARD = -0.01
        self.TRANSITION_PROBS = {"success": 0.8, "right": 0.1, "left": 0.1}
        
        # Action definitions
        self.ACTIONS = {
            0: "UP",     # Up
            1: "RIGHT",  # Right  
            2: "DOWN",   # Down
            3: "LEFT"    # Left
        }
        self.ACTION_SYMBOLS = ['↑', '→', '↓', '←']
    
    def _find_terminal_states(self) -> List[Tuple[int, int]]:
        """Find all terminal states (non-zero values in grid)."""
        terminal_states = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r, c] != 0:
                    terminal_states.append((r, c))
        return terminal_states
    
    def step(self, state: Tuple[int, int], action: int) -> Tuple[int, int]:
        """
        Take a step in the environment.
        
        Args:
            state: Current (row, col) position
            action: Action to take (0-3)
            
        Returns:
            New state (row, col)
        """
        r, c = state
        
        if action == 0:  # Up
            return max(0, r - 1), c
        elif action == 1:  # Right
            return r, min(c + 1, self.cols - 1)
        elif action == 2:  # Down
            return min(r + 1, self.rows - 1), c
        elif action == 3:  # Left
            return r, max(0, c - 1)
        else:
            raise ValueError(f"Invalid action: {action}")
    
    def is_terminal(self, state: Tuple[int, int]) -> bool:
        """Check if a state is terminal."""
        r, c = state
        return self.grid[r, c] != 0


class MDP_Solver:
    """Solves MDP problems using Value Iteration and Policy Iteration."""
    
    def __init__(self, env: GridWorld):
        self.env = env
    
    def value_iteration(self, iterations: int = 1000, tolerance: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform Value Iteration to find optimal policy.
        
        Args:
            iterations: Maximum number of iterations
            tolerance: Convergence threshold
            
        Returns:
            Value function and policy
        """
        V = np.zeros_like(self.env.grid, dtype=float)
        policy = np.zeros_like(self.env.grid, dtype=int)
        
        for iteration in range(iterations):
            new_V = np.copy(V)
            max_delta = 0
            
            for r in range(self.env.rows):
                for c in range(self.env.cols):
                    state = (r, c)
                    
                    # Skip terminal states
                    if self.env.is_terminal(state):
                        new_V[r, c] = self.env.grid[r, c]
                        continue
                    
                    # Calculate Q-values for all actions
                    q_values = []
                    for action in range(4):
                        total_value = 0
                        for prob, offset in zip(self.env.TRANSITION_PROBS.values(), [-1, 0, 1]):
                            next_action = (action + offset) % 4
                            next_state = self.env.step(state, next_action)
                            reward = self.env.grid[next_state] + self.env.LIVING_REWARD
                            total_value += prob * (reward + self.env.GAMMA * V[next_state])
                        q_values.append(total_value)
                    
                    # Update value and policy
                    new_V[r, c] = max(q_values)
                    policy[r, c] = np.argmax(q_values)
                    max_delta = max(max_delta, abs(new_V[r, c] - V[r, c]))
            
            V = new_V
            
            # Check for convergence
            if max_delta < tolerance:
                print(f"Value Iteration converged after {iteration + 1} iterations")
                break
        
        return V, policy
    
    def policy_iteration(self, iterations: int = 100, eval_iterations: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform Policy Iteration to find optimal policy.
        
        Args:
            iterations: Maximum number of policy iterations
            eval_iterations: Maximum number of policy evaluation iterations
            
        Returns:
            Value function and policy
        """
        V = np.zeros_like(self.env.grid, dtype=float)
        policy = np.zeros_like(self.env.grid, dtype=int)
        
        for policy_iter in range(iterations):
            # Policy Evaluation
            for eval_iter in range(eval_iterations):
                new_V = np.copy(V)
                max_delta = 0
                
                for r in range(self.env.rows):
                    for c in range(self.env.cols):
                        state = (r, c)
                        
                        if self.env.is_terminal(state):
                            new_V[r, c] = self.env.grid[r, c]
                            continue
                        
                        action = policy[r, c]
                        total_value = 0
                        for prob, offset in zip(self.env.TRANSITION_PROBS.values(), [-1, 0, 1]):
                            next_action = (action + offset) % 4
                            next_state = self.env.step(state, next_action)
                            reward = self.env.grid[next_state] + self.env.LIVING_REWARD
                            total_value += prob * (reward + self.env.GAMMA * V[next_state])
                        
                        new_V[r, c] = total_value
                        max_delta = max(max_delta, abs(new_V[r, c] - V[r, c]))
                
                V = new_V
                
                if max_delta < 1e-6:
                    break
            
            # Policy Improvement
            policy_stable = True
            for r in range(self.env.rows):
                for c in range(self.env.cols):
                    state = (r, c)
                    
                    if self.env.is_terminal(state):
                        continue
                    
                    q_values = []
                    for action in range(4):
                        total_value = 0
                        for prob, offset in zip(self.env.TRANSITION_PROBS.values(), [-1, 0, 1]):
                            next_action = (action + offset) % 4
                            next_state = self.env.step(state, next_action)
                            reward = self.env.grid[next_state] + self.env.LIVING_REWARD
                            total_value += prob * (reward + self.env.GAMMA * V[next_state])
                        q_values.append(total_value)
                    
                    best_action = np.argmax(q_values)
                    if policy[r, c] != best_action:
                        policy_stable = False
                    policy[r, c] = best_action
            
            if policy_stable:
                print(f"Policy Iteration converged after {policy_iter + 1} iterations")
                break
        
        return V, policy


class Simulator:
    """Simulates agent behavior in the environment."""
    
    def __init__(self, env: GridWorld):
        self.env = env
    
    def simulate(self, policy: np.ndarray, episodes: int = 100, max_steps: int = 100) -> float:
        """
        Simulate policy execution.
        
        Args:
            policy: Policy to simulate
            episodes: Number of episodes to run
            max_steps: Maximum steps per episode
            
        Returns:
            Average reward per episode
        """
        total_reward = 0
        
        for episode in range(episodes):
            # Start from random non-terminal state
            while True:
                state = (np.random.randint(self.env.rows), np.random.randint(self.env.cols))
                if not self.env.is_terminal(state):
                    break
            
            episode_reward = 0
            
            for step in range(max_steps):
                r, c = state
                action = policy[r, c]
                
                # Stochastic transition
                rand_val = random.random()
                if rand_val < self.env.TRANSITION_PROBS["success"]:
                    actual_action = action
                elif rand_val < self.env.TRANSITION_PROBS["success"] + self.env.TRANSITION_PROBS["right"]:
                    actual_action = (action + 1) % 4
                else:
                    actual_action = (action - 1) % 4
                
                next_state = self.env.step(state, actual_action)
                
                # Calculate reward (environment reward + living reward)
                reward = self.env.grid[next_state] + self.env.LIVING_REWARD
                episode_reward += reward
                
                state = next_state
                
                # Check if terminal state reached
                if self.env.is_terminal(state):
                    break
            
            total_reward += episode_reward
        
        return total_reward / episodes
    
    def print_policy(self, policy: np.ndarray):
        """Print policy in a readable format."""
        for row in policy:
            print(" ".join(self.env.ACTION_SYMBOLS[a] for a in row))


def main():
    """Main function to run the MDP solution comparison."""
    # Create environment and solvers
    env = GridWorld()
    solver = MDP_Solver(env)
    simulator = Simulator(env)
    
    print("=" * 50)
    print("MDP SOLVER COMPARISON")
    print("=" * 50)
    
    # Value Iteration
    print("\n1. VALUE ITERATION")
    print("-" * 30)
    V_vi, policy_vi = solver.value_iteration()
    mean_reward_vi = simulator.simulate(policy_vi)
    
    print("Value Function:")
    print(np.round(V_vi, 2))
    print("\nOptimal Policy:")
    simulator.print_policy(policy_vi)
    print(f"\nAverage Reward: {mean_reward_vi:.3f}")
    
    # Policy Iteration  
    print("\n2. POLICY ITERATION")
    print("-" * 30)
    V_pi, policy_pi = solver.policy_iteration()
    mean_reward_pi = simulator.simulate(policy_pi)
    
    print("Value Function:")
    print(np.round(V_pi, 2))
    print("\nOptimal Policy:")
    simulator.print_policy(policy_pi)
    print(f"\nAverage Reward: {mean_reward_pi:.3f}")
    
    # Comparison
    print("\n3. COMPARISON")
    print("-" * 30)
    print(f"Value Iteration Average Reward:  {mean_reward_vi:.3f}")
    print(f"Policy Iteration Average Reward: {mean_reward_pi:.3f}")
    
    # Check if policies are identical
    policies_identical = np.array_equal(policy_vi, policy_pi)
    print(f"Policies Identical: {policies_identical}")


if __name__ == "__main__":
    main()
