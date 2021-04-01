# Adapted from: https://github.com/lazyprogrammer/machine_learning_examples/tree/master/rl
import numpy as np

class Grid: # Environment
  def __init__(self, width, height, start):
    self.width = width
    self.height = height
    self.i = start[0]
    self.j = start[1]

  def set(self, rewards, actions, draw):
    # rewards should be a dict of: (i, j): r (row, col): reward
    # actions should be a dict of: (i, j): A (row, col): list of possible actions
    self.rewards = rewards
    self.actions = actions
    self.draw = draw

  def set_state(self, s):
    self.i = s[0]
    self.j = s[1]

  def current_state(self):
    return (self.i, self.j)

  def is_terminal(self, s):
    return s not in self.actions

  def move(self, action):
    # check if legal move first
    if action in self.actions[(self.i, self.j)]:
      if action == 'UL':
        self.i -= 2
        self.j -= 1
      elif action == 'UR':
        self.i -= 2
        self.j += 1
      elif action == 'RU':
        self.i -= 1
        self.j += 2
      elif action == 'RD':
        self.i += 1
        self.j += 2
      elif action == 'DR':
        self.i += 2
        self.j += 1
      elif action == 'DL':
        self.i += 2
        self.j -= 1
      elif action == 'LU':
        self.i -= 1
        self.j -= 2
      elif action == 'LD':
        self.i += 1
        self.j -= 2
    # return a reward (if any)
    return self.rewards.get((self.i, self.j), 0)

  def undo_move(self, action):
    # these are the opposite of what U/D/L/R should normally do
    if action == 'UL':
      self.i += 2
      self.j += 1
    elif action == 'UR':
      self.i += 2
      self.j -= 1
    elif action == 'RU':
      self.i += 1
      self.j -= 2
    elif action == 'RD':
      self.i -= 1
      self.j -= 2
    elif action == 'DR':
      self.i -= 2
      self.j -= 1
    elif action == 'DL':
      self.i -= 2
      self.j += 1
    elif action == 'LU':
      self.i += 1
      self.j += 2
    elif action == 'LD':
      self.i -= 1
      self.j += 2
    # raise an exception if we arrive somewhere we shouldn't be
    # should never happen
    assert(self.current_state() in self.all_states())

  def game_over(self):
    # returns true if game is over, else false
    # true if we are in a state where no actions are possible
    return (self.i, self.j) not in self.actions

  def all_states(self):
    # possibly buggy but simple way to get all states
    # either a position that has possible next actions
    # or a position that yields a reward
    return set(self.actions.keys()) | set(self.rewards.keys())


def standard_grid(width, hight, start_point, soldiers, king_point, soldier_aim_cost, soldier_cost, king_cost, step_cost):
  # define a grid that describes the reward for arriving at each state
  # and possible actions at each state
  # the grid looks like this
  # x means you can't go there
  # s means start position
  # number means reward at that state
  # .  .  .  1
  # .  x  . -1
  # s  .  .  .
  #width = 8
  #hight = 8
  step_cost = 0#-1
  g = Grid(width, hight, start_point)
  #soldier1_row = soldier_point[0]
  #soldier1_col = soldier_point[1]
  #king_row = king_point[0]
  #king_col = king_point[1]
  #soldier1_loc = soldier_point
  king_loc = king_point
  actions = {}
  rewards = {}
  draw = {}

  for i in range(hight):
    for j in range(width):
      possible_moves = ()
      if i < hight - 2:
        if j == 0:
          possible_moves = possible_moves + ('DR', )
        elif j == hight - 1:
          possible_moves = possible_moves + ('DL', )
        else:
          possible_moves = possible_moves + ('DR', 'DL', )
      if i > 1:
        if j == 0:
          possible_moves = possible_moves + ('UR', )
        elif j == hight - 1:
          possible_moves = possible_moves + ('UL', )
        else:
          possible_moves = possible_moves + ('UR', 'UL', )
      if j > 1:
        if i == 0:
          possible_moves = possible_moves + ('LD', )
        elif i == hight - 1:
          possible_moves = possible_moves + ('LU', )
        else:
          possible_moves = possible_moves + ('LD', 'LU', )
      if j < width - 2:
        if i == 0:
          possible_moves = possible_moves + ('RD', )
        elif i == hight - 1:
          possible_moves = possible_moves + ('RU', )
        else:
          possible_moves = possible_moves + ('RD', 'RU', )
      if possible_moves.count != 0:
        actions.update({(i, j): possible_moves})
      rewards.update({(i, j): step_cost})
      if (i, j) in soldiers: # i == soldier1_row and j == soldier1_col:
        rewards.update({
          (i, j): soldier_cost,
          (i-1, j-1): soldier_aim_cost,
          (i-1, j+1): soldier_aim_cost,
        })
        #draw.update({soldier1_loc: 's'})
      if (i, j) == king_loc: #i == king_row and j == king_col:
        rewards.update({king_loc: king_cost})
        #draw.update({king_loc: 'K'})
      #draw.update({(i, j): ' '})
  """
  actions = {
    (0, 0): ('D', 'R'),
    (0, 1): ('L', 'R'),
    (0, 2): ('L', 'D', 'R'),
    (1, 0): ('U', 'D'),
    (1, 2): ('U', 'D', 'R'),
    (2, 0): ('U', 'R'),
    (2, 1): ('L', 'R'),
    (2, 2): ('L', 'R', 'U'),
    (2, 3): ('L', 'U'),
  }
  """
  g.set(rewards, actions, draw)
  return g


def negative_grid(width, hight, start_point, soldiers, king_point, soldier_aim_cost, soldier_cost, king_cost, step_cost):
  # in this game we want to try to minimize the number of moves
  # so we will penalize every move
  g = standard_grid(width, hight, start_point, soldiers, king_point, soldier_aim_cost, soldier_cost, king_cost, step_cost)
  """
  g.rewards.update({
    (0, 0): step_cost,
    (0, 1): step_cost,
    (0, 2): step_cost,
    (1, 0): step_cost,
    (1, 2): step_cost,
    (2, 0): step_cost,
    (2, 1): step_cost,
    (2, 2): step_cost,
    (2, 3): step_cost,
  })
  """
  return g


def print_values(V, g):
  for i in range(g.width):
    print("------------------------------------------------------")
    for j in range(g.height):
      v = V.get((i,j), 0)
      if v >= 0:
        print(" %.2f|" % v, end="")
      else:
        print("%.2f|" % v, end="") # -ve sign takes up an extra space
    print("")


def print_policy(P, g):
  for i in range(g.width):
    print("------------------------------------------------------")
    for j in range(g.height):
      a = P.get((i,j), ' ')
      print("  %s  |" % a, end="")
    print("")

def draw_next_and_move(P, g, point):
  i = point[0]
  j = point[1]
  next_point = ()
  if P.get(point, ' ') == 'UL':
    g.draw.update({(i - 2, j - 1): '*'})
    g.draw.update({(i - 1, j): '*'})
    g.draw.update({(i - 2, j): '*'})
    next_point = (i-2, j-1)
  elif P.get(point, ' ') == 'UR':
    g.draw.update({(i - 2, j + 1): '*'})
    g.draw.update({(i - 1, j): '*'})
    g.draw.update({(i - 2, j): '*'})
    next_point = (i+2, j+1)
  elif P.get(point, ' ') == 'RU':
    g.draw.update({(i - 1, j + 2): '*'})
    g.draw.update({(i, j + 2): '*'})
    g.draw.update({(i, j + 1): '*'})
    next_point = (i-1, j+2)
  elif P.get(point, ' ') == 'RD':
    g.draw.update({(i + 1, j + 2): '*'})
    g.draw.update({(i, j + 2): '*'})
    g.draw.update({(i, j + 1): '*'})
    next_point = (i+1, j+2)
  elif P.get(point, ' ') == 'DR':
    g.draw.update({(i + 2, j + 1): '*'})
    g.draw.update({(i + 1, j): '*'})
    g.draw.update({(i + 2, j): '*'})
    next_point = (i+2, j+1)
  elif P.get(point, ' ') == 'DL':
    g.draw.update({(i + 2, j - 1): '*'})
    g.draw.update({(i + 1, j): '*'})
    g.draw.update({(i + 2, j): '*'})
    next_point = (i+2, j-1)
  elif P.get(point, ' ') == 'LU':
    g.draw.update({(i - 1, j - 2): '*'})
    g.draw.update({(i, j - 2): '*'})
    g.draw.update({(i, j - 1): '*'})
    next_point = (i-1, j-2)
  elif P.get(point, ' ') == 'LD':
    g.draw.update({(i + 1, j - 2): '*'})
    g.draw.update({(i, j - 2): '*'})
    g.draw.update({(i, j - 1): '*'})
    next_point = (i+1, j-2)
  return next_point

def print_moves_from_start(P, g, start, soliders, king):
  g.draw.update({start: 'S'})
  point = start
  step = 1
  while point != king:
    next_point = draw_next_and_move(P, g, point)
    point = next_point
    g.draw.update({point: '{}'.format(step)})
    step += 1
      
  g.draw.update({king: 'K'})
  for sol in soliders: 
    g.draw.update({sol: 's'})

  for i in range(g.width):
    print("------------------------------------------------------------")
    for j in range(g.height):
      a = g.draw.get((i,j), ' ')
      print("  %s  |" % a, end="")
    print("")
  print("")