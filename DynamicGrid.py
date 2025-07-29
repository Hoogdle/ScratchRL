import numpy as np

class GridWorld:
    def __init__(self):
        self.action_space = [0,1,2,3]
        self.action_mapping = {
            0: 'UP',
            1: 'DOWN',
            2: 'LEFT',
            3: 'RIGHT'
        }
        self.reward_map = np.array(
            [[0,0,0,1.0],
             [0,None,0,-1.0],
             [0,0,0,0]]
        )
        self.goal_state = (0,3)
        self.wall_state = (1,1)
        self.start_state = (2,0)
        self.agent_state = self.start_state

    @property
    def height(self):
        return len(self.reward_map)
    
    @property
    def width(self):
        return len(self.reward_map[0])
    
    @property
    def shape(self):
        return self.reward_map.shape
    
    @property
    def actions(self):
        return self.action_space
    
    @property
    def states(self):
        for h in range(self.height):
            for w in range(self.width):
                yield (h,w)

    def next_state(self, state, action):
        action_move_map = [(-1,0), (1,0), (0,-1), (0,1)]
        move = action_move_map[action]
        next_state = (state[0] + move[0], state[1] + move[1])
        next_x, next_y = next_state

        if next_x < 0 or next_x >= self.width or next_y < 0 or  next_y >= self.height:
            next_state = state
        elif next_state == self.wall_state:
            next_state = state
        
        return next_state
    
    def reward(self, new_state):
        return self.reward_map[new_state]

# Setting
from collections import defaultdict

env = GridWorld()
V = defaultdict(lambda: 0)
pi = defaultdict(lambda: {0:0.25, 1:0.25, 2:0.25, 3:0.25})

# update value map
def eval_onestep(pi: defaultdict, V: defaultdict, env: GridWorld, gamma=0.9):
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue
        
        action_prob = pi[state]
        new_state_value = 0

        for action, prob in pi.items():
            new_state = env.next_state(state = state, action = action)
            reward = env.reward(new_state = new_state)
            new_state_value += prob * (reward + gamma*V[state])

        V[state] = new_state_value
    return V

# Iterate Updating Value
def policy_eval(pi: defaultdict, V: defaultdict, env: GridWorld, gamma, threshold=0.001):
    while True:
        old_V = V.copy()
        V = eval_onestep(pi=pi, V=V, env=env, gamma=gamma)

        max = 0
        
        for state in V.keys():
           delta = abs(old_V[state] - V[state])

           if delta > max:
               max = delta
        
        if max < threshold:
            break
    
    return V
