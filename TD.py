import numpy as np
from collections import defaultdict
from DynamicGrid import GridWorld
from MonteCarlo import greedy_probs

class TdAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.01
        self.action_size = 4

        random_actions = {0:.25, 1:.25, 2:.25, 3:.25}
        self.pi = defaultdict(lambda: random_actions)
        self.V = defaultdict(lambda: 0)

    def get_action(self, state):
        action_probs = self.pi
        action, prob = list(action_probs.keys()), list(action_probs.values())
        return np.random.choice(action, p=prob)
    
    def eval(self, state, reward, next_state, done):
        next_V = 0 if done else self.V[next_state]
        target = reward + self.gamma * next_V

        self.V[state] += self.alpha * (target - self.V[state])

env = GridWorld()
agent = TdAgent()

episodes = 1000

for episode in range(episodes):
    state = env.reset()

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)

        agent.eval(state, reward, next_state, done)

        if done:
            break
        state = next_state

env.render_v(agent.V)