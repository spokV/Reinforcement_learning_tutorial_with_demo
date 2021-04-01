# Adapted from: https://github.com/lazyprogrammer/machine_learning_examples/tree/master/rl
# Value iteration
import numpy as np
from gridWorldGame_c import standard_grid, negative_grid,print_values, print_policy, print_moves_from_start

SMALL_ENOUGH = 1e-7
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('UL', 'UR', 'DL', 'DR', 'LU', 'LD', 'RU', 'RD')
start_point = (0, 0)
soldiers = {(2, 4), (5, 5), (4, 3), (5, 2)}
king_point = (8, 6)
king_cost = 10
soldier_cost = 7
soldier_aim_cost = -1
step_cost = 0
width = 10
hight = 10
# this grid gives you a reward of -0.1
# to find a shorter path to the goal, use negative grid
grid = negative_grid(width, hight, start_point, soldiers, king_point, soldier_aim_cost, soldier_cost, king_cost, step_cost)
print("rewards:")
print_values(grid.rewards, grid)

# state -> action
# choose an action and update randomly 
policy = {}
for s in grid.actions.keys():
  policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

# initial policy
print("initial policy:")
print_policy(policy, grid)
print(grid.all_states())
print(grid.actions)

# initialize V(s) - value function
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
print(V)
print_values(V, grid)

# this section is different from policy iteration
# repeat until convergence
# V[s] = max[a]{ sum[s',r] { p(s',r|s,a)[r + gamma*V[s']] } }
iteration=0
while True:
  iteration+=1
  """
  print("values %d: " % iteration)
  print_values(V, grid)
  print("policy %d: " % iteration)
  print_policy(policy, grid)
  """
  biggest_change = 0
  for s in states:
    old_v = V[s]

    # V(s) only has value if it's not a terminal state
    if s in policy:
      new_v = float('-inf')
      for a in ALL_POSSIBLE_ACTIONS:
        grid.set_state(s)
        r = grid.move(a)
        v = r + GAMMA * V[grid.current_state()]
        if v > new_v:
          new_v = v
      V[s] = new_v
      biggest_change = max(biggest_change, np.abs(old_v - V[s]))

  if biggest_change < SMALL_ENOUGH:
    break

# find a policy that leads to optimal value function
for s in policy.keys():
  best_a = None
  best_value = float('-inf')
  # loop through all possible actions to find the best current action
  for a in ALL_POSSIBLE_ACTIONS:
    grid.set_state(s)
    r = grid.move(a)
    v = r + GAMMA * V[grid.current_state()]
    if v > best_value:
      best_value = v
      best_a = a
  policy[s] = best_a

# our goal here is to verify that we get the same answer as with policy iteration
print("values:")
print_values(V, grid)
print("policy:")
print_policy(policy, grid)
print_moves_from_start(policy, grid, start_point, soldiers, king_point)