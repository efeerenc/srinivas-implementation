import GPy
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm

class TargetPrior():
  
  def __init__(self, d_cardinal=1000):
    self.d_cardinal = d_cardinal
    self.X = np.linspace(0, 1, self.d_cardinal)[:, None]

    self.mu = np.zeros(self.d_cardinal)
    self.k = GPy.kern.RBF(1)
    self.k.lengthscale = 0.2

    self.C = self.k.K(self.X, self.X)

    self.Y = np.random.multivariate_normal(self.mu, self.C, 1)[0]
  
  def sample(self, x):
    return self.Y[int(np.unique(x))]
    #return np.sin(x)
  
  def sample_noisy(self, x):
    return self.sample(x) + np.random.normal(0, 0)
  
  def best_action(self):
    return np.argmax(self.Y)
  
  def best_reward(self):

    best_action_idx = self.best_action()
    return self.sample_noisy(best_action_idx)


class GPUCB():

  def __init__(self, target, rounds=50, delta=0.1, d_cardinal=1000):

    self.target = target
    self.d_cardinal = d_cardinal
    self.interval = np.linspace(0, 1, self.d_cardinal)[:, None]

    self.X = np.zeros(rounds)[:, None]
    self.Y = np.zeros(rounds)[:, None]

    self.regrets = np.zeros(rounds)

    self.beta = None
    self.delta = delta
    
    self.kernel = GPy.kern.RBF(1)
    self.kernel.lengthscale = 0.2
    self.kernel.fix()

    self.mu = np.zeros(self.interval.size)
    self.sigma = np.ones(self.interval.size)/2

    self.round = 1
    self.rounds = rounds
  
  def update(self, plotting=False):
    
    # Choose optimal exploration parameter (Theorem 1)
    self.beta = 2*np.log(self.d_cardinal*(self.round**2)*(np.pi**2)/(6*self.delta))

    # Choose an action according to the exploration parameter
    action_idx = np.argmax(self.mu + np.sqrt(self.beta)*self.sigma)
    action = self.interval[action_idx]

    # Get noisy feedback
    reward = self.target.sample_noisy(action_idx)

    # Store for Bayesian update
    self.X[self.round-1] = action
    self.Y[self.round-1] = reward

    # GP Regression
    gp_model = GPy.models.GPRegression(self.X[:self.round], self.Y[:self.round], self.kernel)
    gp_model.optimize()

    # Update parameters
    self.mu, var = gp_model.predict(self.interval)
    self.sigma = np.sqrt(var)

    # Compute regret for this round
    best_reward = self.target.best_reward()
    regret = best_reward - reward

    # Store regret
    self.regrets[self.round-1] = regret

    if plotting:
      self.plot(self.X[:self.round], self.Y[:self.round])

    # Proceed to the next round
    self.round += 1
  
  def train(self, update_plot=False, train_plot=True, no_print=True):

    for i in tqdm(range(self.rounds), disable=no_print):
      self.update(update_plot)
    
    if train_plot:
      self.plot(self.X, self.Y)

  
  def plot(self, X, Y):

    plt.figure()
    plt.plot(self.target.X, self.target.Y)
    plt.plot(X, Y, "ro")
    plt.show()


def reproducibility(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == "__main__":
    rounds = 100
    experiments = 50

    regrets = np.zeros(rounds)

    reproducibility(1)

    for i in tqdm(range(experiments), disable=False):
        target = TargetPrior()
        gp_agent = GPUCB(target, rounds=100)
        gp_agent.train()
        regrets += gp_agent.regrets

        regrets /= experiments

        plt.figure()
        plt.plot(np.arange(rounds), regrets)