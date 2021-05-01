# Adapted from: https://github.com/lazyprogrammer/machine_learning_examples/tree/master/rl
# Value iteration
import numpy as np
from gridWorldGame_c import standard_grid, negative_grid,print_values, print_policy, print_moves_from_start

SMALL_ENOUGH = 1e-3
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('UL', 'UR', 'DL', 'DR', 'LU', 'LD', 'RU', 'RD')
start_point = (0, 0)
soldiers = {(2, 4), (5, 5), (4, 3), (5, 2)}
king_point = (6, 6)
king_cost = 10
soldier_cost = 1
soldier_aim_cost = -1
step_cost = 0
width = 8
hight = 8
# this grid gives you a reward of -0.1
# to find a shorter path to the goal, use negative grid
grid = negative_grid(width, hight, start_point, soldiers, king_point, soldier_aim_cost, soldier_cost, king_cost, step_cost)
print("rewards:")
print_values(grid.rewards, grid)

# state -> action
# choose an action and update randomly 
policy = {}
for s in grid.actions.keys():
  policy[s] = grid.actions.get(s)#np.random.choice(ALL_POSSIBLE_ACTIONS)

# initialize V(s) - value function
V = {}
states = grid.all_states()
for s in states:
  V[s] = 0

# initial value for all states in grid
print(V)
print_values(V, grid)

# this section is different from policy iteration
# repeat until convergence
# V[s] = max[a]{ sum[s',r] { p(s',r|s,a)[r + gamma*V[s']] } }
iteration=0
while True:
  iteration+=1
  
  print("values %d: " % iteration)
  print_values(V, grid)

  biggest_change = 0
  for s in states:
    old_v = V[s]

    # V(s) only has value if it's not a terminal state
    if s in policy:
      new_v = 0
      for a in grid.actions.get(s):
        grid.set_state(s)
        r = grid.move(a)
        policy_prop = 1/len(grid.actions.get(s))
        new_v += policy_prop * (r + GAMMA * V[grid.current_state()])
      V[s] = new_v
      biggest_change = max(biggest_change, np.abs(old_v - V[s]))

  if biggest_change < SMALL_ENOUGH:
    break

# our goal here is to verify that we get the same answer as with policy iteration
print("values:")
print_values(V, grid)
