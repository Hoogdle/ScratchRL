import numpy as np

np.random.seed(0)

class Bandit:
    def __init__(self, arms=10):
        self.rates = np.random.rand(arms)

    def play(self, arm):
        rate = self.rates[arm]
        trial = np.random.rand()

        if rate > trial:
            return 1
        else:
            return 0
    
class Agent:
    def __init__(self, e, arms_num=10):
        self.e = e
        self.Qs = np.zeros(arms_num)
        self.ns = np.zeros(arms_num)
    
    def update(self, index, reward):
        self.ns[index] += 1
        self.Qs[index] += (reward - self.Qs[index]) / self.ns[index]
    
    def get_action(self):
        random_value = np.random.rand()
        
        if self.e > random_value:
            return np.random.randint(0, len(self.Qs))
        else:
            return np.argmax(self.Qs)
        

### Do Bendit ###
import matplotlib.pyplot as plt

steps = 1000
e = 0.1

bandit = Bandit()
agent = Agent(e = e, arms_num=10)
total_reward = 0
total_rewards = []
rates = []

for step in range(steps):
    action = agent.get_action()
    reward = bandit.play(action)
    agent.update(index = action, reward=reward)
    total_reward += reward

    total_rewards.append(total_reward)
    rates.append(total_reward / (step+1))

print(total_reward)
print(rates[-1])

# draw graph
plt.ylabel('Total Reward')
plt.xlabel('Steps')
plt.plot(total_rewards)
plt.show()

plt.ylabel('Rates')
plt.xlabel('Steps')
plt.plot(rates)
plt.show()
