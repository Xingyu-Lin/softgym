import copy
import pickle
import numpy as np
import scipy.stats as stats


class CEMOptimizer(object):
    def __init__(self, cost_function, solution_dim, max_iters, population_size, num_elites,
                 upper_bound=None, lower_bound=None, epsilon=0.05):
        """
        :param cost_function: Takes input one or multiple data points in R^{sol_dim}
        :param solution_dim: The dimensionality of the problem space
        :param max_iters: The maximum number of iterations to perform during optimization
        :param population_size: The number of candidate solutions to be sampled at every iteration
        :param num_elites: The number of top solutions that will be used to obtain the distribution
                            at the next iteration.
        :param upper_bound: An array of upper bounds for the sampled data points
        :param lower_bound: An array of lower bounds for the sampled data points
        :param epsilon: A minimum variance. If the maximum variance drops below epsilon, optimization is stopped.
        """
        super().__init__()
        self.solution_dim, self.max_iters, self.population_size, self.num_elites = \
            solution_dim, max_iters, population_size, num_elites

        self.ub, self.lb = upper_bound.reshape([1, solution_dim]), lower_bound.reshape([1, solution_dim])
        self.epsilon = epsilon

        if num_elites > population_size:
            raise ValueError("Number of elites must be at most the population size.")

        self.cost_function = cost_function

    def obtain_solution(self, cur_state, init_mean=None, init_var=None):
        """ Optimizes the cost function using the provided initial candidate distribution
        :param cur_state: Full state of the current environment such that the environment can always be reset to this state
        :param init_mean: (np.ndarray) The mean of the initial candidate distribution.
        :param init_var: (np.ndarray) The variance of the initial candidate distribution.
        :return:
        """
        if init_mean is None or init_var is None:
            mean = (self.ub + self.lb) / 2.
            var = (self.ub - self.lb) / 4.
        else:
            mean, var = init_mean, init_var
        t = 0
        X = stats.norm(loc=np.zeros_like(mean), scale=np.ones_like(mean))

        while (t < self.max_iters) and np.max(var) > self.epsilon:
            samples = X.rvs(size=[self.population_size, self.solution_dim]) * np.sqrt(var) + mean
            samples = np.clip(samples, self.lb, self.ub)
            costs = self.cost_function(cur_state, samples)
            sort_costs = np.argsort(costs)

            elites = samples[sort_costs][:self.num_elites]

            mean = np.mean(elites, axis=0)
            var = np.var(elites, axis=0)

            t += 1
        sol, solvar = mean, var
        return sol


class CEMPolicy(object):
    """ Use the ground truth dynamics to optimize a trajectory of actions. """

    def __init__(self, env, plan_horizon, max_iters, population_size, num_elites):
        self.env = env
        self.plan_horizon, self.action_dim = plan_horizon, len(env.action_space.sample())
        self.action_buffer = []

        lower_bound = np.tile(env.action_space.low[None], [self.plan_horizon, 1]).flatten()
        upper_bound = np.tile(env.action_space.high[None], [self.plan_horizon, 1]).flatten()
        self.optimizer = CEMOptimizer(self.cost_function,
                                      self.plan_horizon * self.action_dim,
                                      max_iters=max_iters,
                                      population_size=population_size,
                                      num_elites=num_elites,
                                      lower_bound=lower_bound,
                                      upper_bound=upper_bound,
                                      )

    def cost_function(self, cur_state, action_trajs):
        env = self.env
        env.reset()
        action_trajs = action_trajs.reshape([-1, self.plan_horizon, self.action_dim])
        n = action_trajs.shape[0]
        costs = []
        for i in range(n):
            env.set_state(cur_state)
            ret = 0
            for j in range(self.plan_horizon):
                _, reward, _, _ = env.step(action_trajs[i, j, :])
                ret += reward
            costs.append(-ret)
        return costs

    def get_action(self, state):
        if len(self.action_buffer) > 0:
            action, self.action_buffer = self.action_buffer[0], self.action_buffer[1:]
            return action
        self.env.debug = False
        env_state = self.env.get_state()
        self.action_buffer = self.optimizer.obtain_solution(env_state).reshape([-1, self.action_dim])
        self.env.set_state(env_state)  # Recover the environment
        print("cem finished planning!")
        return self.get_action(state)


if __name__ == '__main__':
    import gym
    import softgym
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--replay", action="store_true")
    parser.add_argument("--traj_path", default="./data/folding_traj/traj.pkl")
    args = parser.parse_args()
    traj_path = args.traj_path

    softgym.register_flex_envs()
    env = gym.make('ClothFlattenPointControl-v0')

    if not args.replay:
        policy = CEMPolicy(env,
                           plan_horizon=10,
                           max_iters=5,
                           population_size=50,
                           num_elites=5)
        # Run policy
        obs = env.reset()
        initial_state = env.get_state()
        action_traj = []
        for _ in range(env.horizon):
            action = policy.get_action(obs)
            action_traj.append(copy.copy(action))
            obs, reward, _, _ = env.step(action)
            print('reward:', reward)

        traj_dict = {
            'initial_state': initial_state,
            'action_traj': action_traj
        }

        with open(traj_path, 'wb') as f:
            pickle.dump(traj_dict, f)
    else:
        with open(traj_path, 'rb') as f:
            traj_dict = pickle.load(f)
        initial_state, action_traj = traj_dict['initial_state'], traj_dict['action_traj']
        env.start_record(video_path='./data/videos/', video_name='cem_folding.gif')
        env.reset()
        env.set_state(initial_state)
        for action in action_traj:
            env.step(action)
        env.end_record()
    # Save the trajectories and replay
