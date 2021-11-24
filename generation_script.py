"""
Encapsulate generate data to make it parallel
"""
from os import makedirs
from os.path import join
import argparse
from multiprocessing import Pool
from subprocess import call

parser = argparse.ArgumentParser()
parser.add_argument('--rollouts', type=int, default=10, help="Total Number of rollouts")
parser.add_argument('--threads', type=int, help="Number of threads")
parser.add_argument('--env_name', type=str)
parser.add_argument('--observation_mode', type=str, default='cam_rgb', help='point_cloud, cam_rgb, key_point')
parser.add_argument('--rootdir', type=str, help="Directory to store rollout "
                    "directories of each thread")
args = parser.parse_args()

rpt = args.rollouts // args.threads + 1

def _threaded_generation(i):
    tdir = join(args.rootdir, 'thread_{}'.format(i))
    makedirs(tdir, exist_ok=True)
    cmd = ['xvfb-run', '-s', '"-screen 0 1400x900x24"']
    cmd += ['--server-num={}'.format(i + 1)]
    cmd += ["python", "softgym_rollout.py", "--env_name", args.env_name, "--headless", "--rollout_num", str(rpt), "--observation_mode", args.observation_mode, "--rollout_dir", tdir ]
    cmd = " ".join(cmd)
    print(cmd)
    call(cmd, shell=True)
    return True


with Pool(args.threads) as p:
    p.map(_threaded_generation, range(args.threads))
