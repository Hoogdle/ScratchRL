from collections import defaultdict
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


def greedy_probs(Q, state, epsilon=0, action_size=4):
    qs = [Q[(state,action)] for action in range(action_size)]
    best_action = np.argmax(qs)

    base_probs = epsilon / action_size
    action_probs = {action : base_probs for action in range(action_size)}
    action_probs[best_action] += (1-epsilon)
    return action_probs

class McAgent:
    def __init__(self):
        self.gamma = 0.9
        self.epsilon = 0.1
        self.alpha = 0.1
        self.action_size = 4

        random_actions = {0:0.25, 1:0.25, 2:0.25, 3:0.25}
        self.pi = defaultdict(lambda : random_actions)
        self.Q = defaultdict(lambda: 0)
        self.memory = []

    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)
    
    def add(self, state, action, reward):
        data = (state, action, reward)
        self.memory.append(data)

    def reset(self):
        self.memory.clear()

    def update(self):
        G = 0
        for data in reversed(self.memory):
            state, action, reward = data
            G = self.gamma*G + reward
            key = (state, action)
            self.Q[key] += (G - self.Q[key]) * self.alpha
            self.pi[state] = greedy_probs(self.Q, state, self.epsilon)


env = GridWorld()
agent = McAgent()

iter_num = 10000

for episode in range(iter_num):
    state = env.reset()
    agent.reset()

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)

        agent.add(state, action, reward)

        if done:
            agent.update()
            break

        state = next_state


env.render(agent.Q)
        
        
        
