import numpy as np
import random
from tqdm import tqdm

class SARSA_Agent():
    def __init__(self, environment):
        self.env = environment


    def reduce_epsilon_over_epochs(self, epoch, max_episode=10000, initial_epsilon=1.0):
        return initial_epsilon * np.power(0.1, (epoch+1)/max_episode)

    
    def train_agent(self, alpha = 0.01, gamma = 0.6, epsilon = 0.8, max_episode = 100000, policy=None):
        q_table = np.ones([self.env.observation_space.n, self.env.action_space.n])*[1e-5]
        if policy == None:
            policy = np.random.randint(low=0, high=3, size=([self.env.observation_space.n]), dtype=np.uint8)

        # For plotting metrics
        all_epochs = []
        accumulative_return = []
        accumulative_return_step = 0

        for e in tqdm(range(max_episode)):
            state = self.env.reset()
            
            episode_reward = 0
            step = 0
            done = False

            while not done:
                if random.uniform(0, 1) < self.reduce_epsilon_over_epochs(e, max_episode= max_episode, initial_epsilon=epsilon):
                    action = self.env.action_space.sample()  # Explore action space
                else:
                    action = policy[state]  # Exploit learned values
                
                (next_state, reward, done, info) = self.env.step(action)
                next_action = policy[next_state]

                q_old = q_table[state, action]
                q_next = q_table[next_state, next_action]

                new_value = (1 - alpha) * q_old + alpha * (reward + gamma * q_next)
                q_table[state, action] = new_value

                state = next_state

                episode_reward = episode_reward + gamma**step * reward
                step += 1 

            all_epochs.append(step)

            policy = self.update_policy(q_table= q_table)
            
            #Accumulated reward
            accumulative_return_step = (0.99*accumulative_return_step) + (0.01*episode_reward)
            # accumulative_return_step = accumulated_reward
            accumulative_return.append(accumulative_return_step)

        return q_table, policy, all_epochs, accumulative_return

    def update_policy(self, q_table: np.array):
        policy = np.zeros([self.env.observation_space.n])
        for i in range(self.env.observation_space.n):
            policy[i] = np.argmax(q_table[i])
        return policy.astype(np.uint8)