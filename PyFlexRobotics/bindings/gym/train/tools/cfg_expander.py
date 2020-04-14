import os, logging, argparse
from autolab_core import YamlConfig
from collections import Mapping
from copy import deepcopy
from tqdm import tqdm

_FLAG_FIRST = object()
def flattenDict(d, join, process, lift=lambda x:x):
    results = []
    def visit(subdict, results, partialKey):
        for k,v in subdict.items():
            newKey = [lift(k)] if partialKey==_FLAG_FIRST else join(partialKey,lift(k))
            if isinstance(v, Mapping):
                visit(v, results, newKey)
            else:
                results.append((newKey, process(v)))
    visit(d, results, _FLAG_FIRST)
    return results

def dict_set(d, keys, val):
    cur_d_ptr = d
    for key in keys[:-1]:
        cur_d_ptr = cur_d_ptr[key]
    cur_d_ptr[keys[-1]] = val

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_base', '-b', type=str)
    parser.add_argument('--cfg_range', '-r', type=str)
    parser.add_argument('--output_dir', '-o', type=str)
    parser.add_argument('--name', '-n', type=str)
    args = parser.parse_args()

    cfg_base = YamlConfig(args.cfg_base)
    cfg_range = YamlConfig(args.cfg_range)

    ranges = flattenDict(cfg_range.config,
        lambda a, b: a + [b], # concat values into a list of values
        lambda v: v
    )

    lengths = set()
    for _, vals in ranges:
        lengths.add(len(vals))
    if len(lengths) > 1:
        raise ValueError('All range configs must have same number of values!')

    for i in tqdm(range(lengths.pop())):
        cfg = deepcopy(cfg_base)
        for keys, vals in ranges:
            dict_set(cfg, keys, vals[i])
        cfg.save(os.path.join(args.output_dir, '{}_{}.yaml'.format(args.name, i)))