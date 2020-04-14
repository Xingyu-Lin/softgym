import os, argparse, sys

from autolab_core import YamlConfig

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", type=str)
    args = parser.parse_args()

    gpu_cfg = YamlConfig(args.cfg)
    if 'tasks' in gpu_cfg['resources']:
        gpus = 8 * gpu_cfg['resources']['tasks']
    else:
        gpus = gpu_cfg['resources']['gpus']
    sys.stdout.write(str(gpus))