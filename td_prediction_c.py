# the temporal difference 0 method to find the optimal policy
# only policy evaluation, not optimization
import numpy as np
import matplotlib.pyplot as plt
from gridWorldGame_c import standard_grid, negative_grid,print_values, print_policy, print_moves_from_start

SMALL_ENOUGH = 1e-3
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('UL', 'UR', 'DL', 'DR', 'LU', 'LD', 'RU', 'RD')
start_point = (0, 0)
soldiers = {(2, 4), (5, 5), (4, 3), (5, 2)}
king_point = (6, 6)
king_cost = 3
soldier_cost = 1
soldier_aim_cost = -1
step_cost = 0
width = 8
hight = 8
ALPHA = 0.1

def random_action(a, eps=0.1):
  # epsilon-soft to ensure all states are visited
  p = np.random.random()
  if p < (1 - eps):
    return a
  else:
    return np.random.choice(ALL_POSSIBLE_ACTIONS)

def play_game(grid, policy):
  # returns a list of states and corresponding rewards (not returns as in MC)
  # start at the designated start state
  s = (0, 0)
  grid.set_state(s)
  states_and_rewards = [(s, 0)] # list of tuples of (state, reward)
  while not grid.game_over():
    a = policy[s]
    a = random_action(a)
    r = grid.move(a)
    s = grid.current_state()
    states_and_rewards.append((s, r))
  return states_and_rewards

#grid = standard_grid()
grid = negative_grid(width, hight, start_point, soldiers, king_point, soldier_aim_cost, soldier_cost, king_cost, step_cost)

# print rewards
print("rewards:")
print_values(grid.rewards, grid)

"""
# state -> action
policy = {
  (2, 0): 'U',
  (1, 0): 'U',
  (0, 0): 'R',
  (0, 1): 'R',
  (0, 2): 'R',
  (1, 2): 'R',
  (2, 1): 'R',
  (2, 2): 'R',
  (2, 3): 'U',
}
"""
policy = {}
for s in grid.actions.keys():
  policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

# initial policy
print("initial policy:")
print_policy(policy, grid)

# initialize V(s) and returns
"""
V = {}
states = grid.all_states()
for s in states:
  V[s] = 0
"""
V = {}
states = grid.all_states()
for s in states:
  # V[s] = 0
  if s in grid.actions:
    V[s] = np.random.random()
  else:
    # terminal state
    V[s] = 0
  
# initial value for all states in grid
print_values(V, grid)

# repeat until convergence
for it in range(10000):
  # generate an episode using pi
  states_and_rewards = play_game(grid, policy)
  # the first (s, r) tuple is the state we start in and 0
  # (since we don't get a reward) for simply starting the game
  # the last (s, r) tuple is the terminal state and the final reward
  # the value for the terminal state is by definition 0, so we don't
  # care about updating it.
  for t in range(len(states_and_rewards) - 1):
    s, _ = states_and_rewards[t]
    s2, r = states_and_rewards[t+1]
    # we will update V(s) AS we experience the episode
    V[s] = V[s] + ALPHA*(r + GAMMA*V[s2] - V[s])

print("final values:")
print_values(V, grid)
#print("final policy:")
#print_policy(policy, grid)
#print_moves_from_start(policy, grid, start_point, soldiers, king_point)