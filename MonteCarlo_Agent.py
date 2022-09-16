import numpy as np
import random
from tqdm import tqdm

class MonteCarlo_Agent():
    def __init__(self, environment):
        self.env = environment
    
    def reduce_epsilon_over_epochs(self, epoch, max_episode=10000, initial_epsilon=1.0):
        return initial_epsilon * np.power(0.1, epoch/max_episode)

    

    def train_agent(self, gamma = 0.6, epsilon = 0.8, max_episode = 100000, policy=None):
        q_table = np.zeros([self.env.observation_space.n, self.env.action_space.n]) # Empty dictionary for storing rewards for each state-action pair
        if policy == None:
            policy = np.ones([self.env.observation_space.n, self.env.action_space.n])*[1/self.env.action_space.n]
        returns = {}

        all_epochs = []
        epoch_step = 0
        accumulative_return = []
        accumulative_return_step = 0

        for e in tqdm(range(max_episode)):
            G = 0 # Store cumulative reward in G (initialized at 0)
            done = False
            state_action = []
            rewards = []

            state = self.env.reset()
            while not done:
                p = random.uniform(0, sum(policy[state]))
                top_range = 0
                for i in range(self.env.action_space.n):
                    top_range += policy[state, i]
                    if p < top_range:
                        action = i
                        break
                # if random.uniform(0, 1) < self.reduce_epsilon_over_epochs(e, max_episode= max_episode, initial_epsilon=epsilon):
                #     action = self.env.action_space.sample()  # Explore action space
                # else:
                #     action = policy[state]  # Exploit learned values
                
                state_action.append((state, action))
                
                (state, reward, done, info) = self.env.step(action)

                rewards.append(reward)


            all_epochs.append(len(state_action))

            

            for i in reversed(range(0, len(state_action))):
                (s, a) = state_action[i]
                r = rewards[i]

                G = r +  gamma * G  # Increment total reward by reward on current timestep

                if (s, a) not in state_action[:i]:
                    if returns.get((s, a)):
                        returns[(s, a)].append(G)
                    else:
                        returns[(s, a)] = [G]

                    q_table[s, a] = sum(returns[(s, a)]) / len(returns[(s, a)]) # Average reward across episodes

            # Accumulated reward
            accumulative_return_step = (0.99*accumulative_return_step) + (0.01*G)
            accumulative_return.append(accumulative_return_step)

            eps = self.reduce_epsilon_over_epochs(e, max_episode= max_episode, initial_epsilon=epsilon)
            for (s, a) in state_action:
                a_star = np.argmax(q_table[s])

                for a in range(self.env.action_space.n): # Update action probability for s_t in policy
                    if a == a_star:
                        policy[s, a] = 1 - eps + (eps / abs(sum(policy[s])))
                    else:
                        policy[s, a] = (eps / abs(sum(policy[s])))
        
        policy = self.final_policy(q_table=q_table)
        return q_table, policy, all_epochs, accumulative_return

    def final_policy(self, q_table: np.array):
        policy = np.zeros(self.env.observation_space.n)
        for i in range(self.env.observation_space.n):
            policy[i] = np.argmax(q_table[i])
        return policy.astype(np.uint8)