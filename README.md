# Comparison of classical RL algorithms
This project aims to implement and compare three reinforcement learning algorithms Q-Learning, SARSA, and Monte Carlo. For this, the Taxi-v3 environment in the Gym library has been used. The agent of each of these algorithms starts exploring and learning in this environment. Also, to better understand how these algorithms work, the parameters in them and the environment have been changed, and a comparison has been made on the results obtained. In the following gifs, you can see the movement of each agent for 5 episodes.

Q-Learning           |  SARSA       |   Monte Carlo
:-------------------------:|:-------------------------:|:-------------------------:
<img src="https://github.com/alireza-montazeri/RL-Algorithms/blob/master/figures/Taxi_Q.gif" />|<img src="https://github.com/alireza-montazeri/RL-Algorithms/blob/master/figures/Taxi_SARSA.gif" />|<img src="https://github.com/alireza-montazeri/RL-Algorithms/blob/master/figures/Taxi_MC.gif" />

The implementation of Q-Learning, SARSA, and Monte Carlo algorithms is in the Q_Agent.py, SARSA_Agent.py, and MonteCarlo_Agent.py files, respectively. The comparison of the effect of changing the parameters in Taxi_QAgent.ipynb, Taxi_SARSA.ipynb, and Taxi_MCAgent.ipynb and Comparison of three algorithms in Taxi_CompareAgents.ipynb file.

## Result
### Affect of Changing Q-Learning parameters
Learning Rate           |  Discount Factor
:-------------------------:|:-------------------------:
<img src="https://github.com/alireza-montazeri/RL-Algorithms/blob/master/figures/Q-alpha-reward.png" />|<img src="https://github.com/alireza-montazeri/RL-Algorithms/blob/master/figures/Q-gamma-reward.png" />
<img src="https://github.com/alireza-montazeri/RL-Algorithms/blob/master/figures/Q-alpha-steps.png" />|<img src="https://github.com/alireza-montazeri/RL-Algorithms/blob/master/figures/Q-gamma-steps.png" />

### Affect of Changing SARSA parameters
Learning Rate           |  Discount Factor
:-------------------------:|:-------------------------:
<img src="https://github.com/alireza-montazeri/RL-Algorithms/blob/master/figures/SARSA-alpha-reward.png" />|<img src="https://github.com/alireza-montazeri/RL-Algorithms/blob/master/figures/SARSA-gamma-reward.png" />
<img src="https://github.com/alireza-montazeri/RL-Algorithms/blob/master/figures/SARSA-alpha-steps.png" />|<img src="https://github.com/alireza-montazeri/RL-Algorithms/blob/master/figures/SARSA-gamma-steps.png" />

### Affect of Changing Monte Carlo parameters
Discount Factor |
:-------------------------:
<img src="https://github.com/alireza-montazeri/RL-Algorithms/blob/master/figures/MC-gamma-reward.png" />
<img src="https://github.com/alireza-montazeri/RL-Algorithms/blob/master/figures/MC-gamma-steps.png" />

### Compare Q-Learning, SARSA, Monte Carlo
Comparison |
:-------------------------:
<img src="https://github.com/alireza-montazeri/RL-Algorithms/blob/master/figures/All-reward.png" />
<img src="https://github.com/alireza-montazeri/RL-Algorithms/blob/master/figures/All-steps.png" />

