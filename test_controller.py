""" Test controller """
import argparse
from os.path import join, exists

from utils.evaluation import RolloutGenerator
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, help='Where models are stored.')
parser.add_argument('--episodes', type=int, help='Number of rollouts to run / episodes.')
args = parser.parse_args()

ctrl_file = join(args.logdir, 'ctrl', 'best.tar')

assert exists(ctrl_file),\
    "Controller was not trained..."

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")


generator = RolloutGenerator(args.logdir, args.episodes, device, 100)

with torch.no_grad():
    generator.rollout(None)
