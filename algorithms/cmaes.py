import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class FCModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FCModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        return x.size()[1]


class CMAES(object):
    def __init__(self,
                 env,
                 sigma=0.2,
                 population_size=50,
                 evaluate_episode=1,
                 elite_ratio=0.2,
                 noise=0.01):
        '''
        Cross entropy solver. Assume that the full dynamics of the environment is known
        :param env:
        :param sigma: initial std for the weights
        :param optimization_iter:
        :param population_size:
        :param elite_ratio:
        :param horizon:
        '''
        self.env = env
        if not hasattr(env, 'horizon'):
            env.horizon = 50
        self.population_size = population_size
        self.evaluate_episode = evaluate_episode
        self.elite_ratio = elite_ratio
        self.elite_cutoff = int(self.population_size * elite_ratio) + 1
        self.noise = noise
        self.action_dim = len(env.action_space.sample())
        self.observation_dim = len(env.observation_space.sample())
        self.model = FCModel(self.observation_dim, self.action_dim)
        self.weight_dim = sum(param.numel() for param in list(self.model.parameters()))
        self.mu = np.zeros(self.weight_dim)
        self.S = sigma ** 2 * np.eye(self.weight_dim)

    def populate(self):
        '''
        Populate a generation using the current estimates of mu and
        :return:
        '''
        self.population = np.random.multivariate_normal(self.mu, self.S, self.population_size)

    def set_weight(self, member):
        member_idx = 0
        for f in self.model.parameters():
            f.data = torch.Tensor(member[member_idx:member_idx + f.numel()]).reshape(f.data.shape)
            member_idx += f.numel()

    def evaluate(self, member, num_episode):
        '''
        Evaluate a set of weights by interacting with the environment and return the average return over all episodes
        '''
        self.set_weight(member)
        return self.evaluate_policy(self.model, num_episode)

    def evaluate_policy(self, policy, num_episode, render=False):
        rets = []
        for i in range(num_episode):
            ret = 0
            obs = env.reset()
            for _ in range(env.horizon):
                action = policy(torch.Tensor(obs)).detach().numpy()
                obs, reward, _, _ = env.step(action)
                if render:
                    env.render()
                ret += reward
            rets.append(ret)
        return np.mean(rets)

    def train_step(self):
        self.populate()
        scores = [self.evaluate(member, self.evaluate_episode) for member in self.population]
        sorted_idx = np.argsort(-1. * np.array(scores))
        self.best_member = self.population[sorted_idx[0]]
        elite_members = self.population[sorted_idx[:self.elite_cutoff]]
        self.mu = np.mean(elite_members, axis=0)
        self.S = np.cov(self.population, rowvar=False) + np.eye(self.weight_dim) * self.noise


if __name__ == '__main__':
    import gym
    import softgym

    softgym.register_flex_envs()
    env = gym.make('ClothFoldPointControl-v0')
    # env = gym.make('Pendulum-v0')
    model_path = './data/cmaes_cloth_fold_{}.pt'
    training = True
    if training:
        agent = CMAES(env)
        for i in range(1000):
            agent.train_step()
            model_return = agent.evaluate_policy(agent.model, 1)
            print('step {}, return {}'.format(i, model_return))
            if i % 50 == 0:
                torch.save(agent.model.state_dict(), model_path.format(i))
    else:
        # TODO: Load model and visualize
        agent = CMAES(env)
        agent.model.load_state_dict(torch.load(model_path))
        agent.evaluate_policy(agent.model, 100, render=True)
