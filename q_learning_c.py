# Adapted from: https://github.com/lazyprogrammer/machine_learning_examples/tree/master/rl
# the Q-Learning method to find the optimal policy and value function
import numpy as np
import matplotlib.pyplot as plt
from gridWorldGame_c import standard_grid, negative_grid,print_values, print_policy, print_moves_from_start

SMALL_ENOUGH = 1e-3
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('UL', 'UR', 'DL', 'DR', 'LU', 'LD', 'RU', 'RD')
start_point = (0, 0)
soldiers = {(2, 4), (5, 5), (4, 3), (5, 2)}
king_point = (6, 6)
king_cost = 2
soldier_cost = 1
soldier_aim_cost = -1
step_cost = -0.1
width = 8
hight = 8
ALPHA = 0.1

def max_dict(d):
  # returns the argmax (key) and max (value) from a dictionary
  max_key = None
  max_val = float('-inf')
  for k, v in d.items():
    if v > max_val:
      max_val = v
      max_key = k
  return max_key, max_val

def random_action(a, eps=0.1):
  # epsilon-soft to ensure all states are visited
  p = np.random.random()
  if p < (1 - eps):
    return a
  else:
    return np.random.choice(ALL_POSSIBLE_ACTIONS)

#grid = negative_grid(step_cost=-0.1)
grid = negative_grid(width, hight, start_point, soldiers, king_point, soldier_aim_cost, soldier_cost, king_cost, step_cost)


# print rewards
print("rewards:")
print_values(grid.rewards, grid)

# no policy initialization, policy is derived from most recent Q like SARSA

# initialize Q(s,a)
Q = {}
states = grid.all_states()
for s in states:
  Q[s] = {}
  for a in ALL_POSSIBLE_ACTIONS:
    Q[s][a] = 0

# initial Q values for all states in grid
print(Q)

update_counts = {}
update_counts_sa = {}
for s in states:
  update_counts_sa[s] = {}
  for a in ALL_POSSIBLE_ACTIONS:
    update_counts_sa[s][a] = 1.0

# repeat until convergence
t = 1.0
deltas = []
for it in range(10000):
  if it % 100 == 0:
    t += 1e-2
  if it % 2000 == 0:
    print("iteration:", it)

  # instead of 'generating' an epsiode, we will PLAY
  # an episode within this loop
  s = (2, 0) # start state
  grid.set_state(s)

  # the first (s, r) tuple is the state we start in and 0
  # (since we don't get a reward) for simply starting the game
  # the last (s, r) tuple is the terminal state and the final reward
  # the value for the terminal state is by definition 0, so we don't
  # care about updating it.
  a, _ = max_dict(Q[s])
  biggest_change = 0
  while not grid.game_over():
    a = random_action(a, eps=0.5/t) # epsilon-greedy
    # random action also works, but slower since you can bump into walls
    # a = np.random.choice(ALL_POSSIBLE_ACTIONS)
    r = grid.move(a)
    s2 = grid.current_state()

    # adaptive learning rate
    alpha = ALPHA / update_counts_sa[s][a]
    update_counts_sa[s][a] += 0.005

    # we will update Q(s,a) AS we experience the episode
    old_qsa = Q[s][a]
    # the difference between SARSA and Q-Learning is with Q-Learning
    # we will use this max[a']{ Q(s',a')} in our update
    # even if we do not end up taking this action in the next step
    a2, max_q_s2a2 = max_dict(Q[s2])
    Q[s][a] = Q[s][a] + alpha*(r + GAMMA*max_q_s2a2 - Q[s][a])
    biggest_change = max(biggest_change, np.abs(old_qsa - Q[s][a]))

    # we would like to know how often Q(s) has been updated too
    update_counts[s] = update_counts.get(s,0) + 1

    # next state becomes current state
    s = s2
    a = a2
   
  deltas.append(biggest_change)

#print("final values:")
#print_values(Q, grid)
plt.plot(deltas)
plt.show()