import numpy as np

# Define the Q-learning function
def q_learning(env, num_episodes, alpha, gamma, epsilon):
    # Initialize Q-table with zeros
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    
    # Loop over episodes
    for i in range(num_episodes):
        # Reset the environment
        state = env.reset()
        done = False
        
        # Loop over time steps within the episode
        while not done:
            # Choose an action using epsilon-greedy policy
            if np.random.uniform() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])
            
            # Take the chosen action and observe the next state and reward
            next_state, reward, done, _ = env.step(action)
            
            # Update the Q-value for the current state-action pair
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state]))
            
            # Update the state
            state = next_state
    
    # Return the learned Q-table
    return Q

def main():
    # Import the gym module
    import gymansium as gym
    # Create a FrozenLake environment
    env = gym.make('FrozenLake-v0')

    # Run the Q-learning algorithm
    Q = q_learning(env, 10000, 0.8, 0.95, 0.1)

    # Print the learned Q-table
    print(Q)
