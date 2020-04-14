import os, logging, argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
from autolab_core import YamlConfig, CSVModel

def uneven_stat(ys, f):
    f_ys = []
    desired_len = int(np.median([len(y) for y in ys]))
    for t in range(desired_len):
        t_vals = []
        for y in ys:
            if t < len(y):
                t_vals.append(y[t])
        
        f_ys.append(f(t_vals))
           
    return np.array(f_ys)
       
def merge_stds(means, stds, ns):
    ns = np.array(ns)
    mean = np.ones(len(means)) * np.mean(means)
    
    return np.sqrt(
        ns.dot(np.power(stds, 2) + np.power(means - mean, 2)) / np.sum(ns)
    )
    
def robust_uneven_stat_std(means, stds, ns, p):
    
    # remove outliers
    robust_means = means
    robust_stds = stds
    if p > 0:
        robust_means = []
        robust_stds = []
        for i, mean in enumerate(means):
            idx = np.where((mean > np.percentile(mean, p)) & (mean <= np.percentile(mean, 100-p)))[0]
            robust_means.append(np.take(mean, idx))
            robust_stds.append(np.take(stds[i], idx))
    

    all_stds = []
    desired_len = int(np.median([len(mean) for mean in means]))
    for t in range(desired_len):
        t_means = []
        t_stds = []
        t_ns = []
        for i, mean in enumerate(robust_means):
            if t < len(mean):
                t_means.append(mean[t])
                t_stds.append(robust_stds[i][t])
                t_ns.append(ns[i])
        
        all_stds.append(merge_stds(t_means, t_stds, t_ns))
        
    return np.array(all_stds)

get_mean = lambda ys: uneven_stat(ys, np.mean)
get_sum = lambda ys: uneven_stat(ys, np.sum)
get_first = lambda xs: xs[0]
def get_std_gen(progresses, n_agents):
    
    means = [p['EpRewMean'] for p in progresses]
    ns = [n_agents] * len(progresses)
    
    def get_stds(stds):
        return robust_uneven_stat_std(means, stds, ns, 0)
    return get_stds

def merge_progresses(exp_dir, n_gpus, n_agents, n_tasks, resources_cfg=None):
    progresses = []
    for i in range(n_gpus):
        progresses.append(pd.read_csv(os.path.join(exp_dir, 'progress_{}.csv'.format(i))))

    reduce_fs = {
        'TimeElapsed': get_first, 

        'EpRewMean': get_mean, 
        'EpLenMean': get_mean, 
        'EpRewStd': get_std_gen(progresses, n_agents),

        'TimestepsSoFar': get_sum, 
        'EpisodesSoFar': get_sum,
        'EpThisIter': get_sum,

        'loss_pol_surr': get_mean, 
        'loss_pol_entpen': get_mean, 
        'loss_vf_loss': get_mean, 
        'loss_kl': get_mean, 
        'loss_ent': get_mean,
        'ev_tdlam_before': get_mean, 
    }

    progress_dict = {}
    for key, f in reduce_fs.items():
        vals = [p[key].values for p in progresses]
        progress_dict[key] = f(vals)

    min_len = min([len(val) for val in progress_dict.values()])
    for key, val in progress_dict.items():
        progress_dict[key] = val[:min_len]

    df = pd.DataFrame(progress_dict)
    df.to_csv(os.path.join(exp_dir, 'progress.csv'))

    if resources_cfg:
        resources_cfg['n_agents'] = n_gpus * n_agents * n_tasks
        resources_cfg.save(os.path.join(exp_dir, 'resources.yaml'))

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    # Either provide the directory to an entire experiment manager experiment 
    # and the name of the environment configs, and whether or not it was multinode
    parser.add_argument('--exp_root_dir', type=str)
    parser.add_argument('--env_cfg_name', '-n', type=str)
    parser.add_argument('--multinode', action='store_true')
    # Or provide the path to a specific experiment run, the number of GPUs used,
    # and the agents per GPU
    parser.add_argument('--exp_dir')
    parser.add_argument('--n_gpus', type=int, default=0)
    parser.add_argument('--n_agents', type=int, default=0)
    args = parser.parse_args()

    if args.exp_root_dir and args.exp_dir:
        raise ValueError('Can only accept either exp_root_dir, which is the path to a folder containing multiple experimnets, or exp_dir, a path to a single experiment!')

    if args.exp_root_dir:
        if args.n_gpus or args.n_agents:
            logging.warn('Got non-zero values for n_pugs or n_agents - ignoring them because they will be found by looking through resources.yaml.')

        exps = CSVModel.load(os.path.join(args.exp_root_dir, 'exps.csv'))
        success_exps = exps.get_rows_by_cols({'finished_status': 'FINISHED_SUCCESS'})
        logging.info('Found {} exps'.format(len(success_exps)))

        for i in tqdm(range(len(success_exps))):
            exp = success_exps[i]
            exp_dir = os.path.join(args.exp_root_dir, exp['exp_dir'])
            resources_cfg = YamlConfig(os.path.join(exp_dir, 'resources.yaml'))

            n_gpus = resources_cfg['resources']['gpus']
            n_agents = YamlConfig(os.path.join(exp_dir, args.env_cfg_name))['scene']['NumAgents']
            n_tasks = 1
            if args.multinode:
                n_tasks = resources_cfg['resources']['tasks']

            merge_progresses(exp_dir, n_gpus, n_agents, n_tasks, resources_cfg)

    else:
        if args.n_gpus == 0 or args.n_agents == 0:
            raise ValueError('Must give n_gpus and n_agents if merging progresses for a single experiment!')

        merge_progresses(args.exp_dir, args.n_gpus, args.n_agents, 1)