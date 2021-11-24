""" Test controller """
import argparse
from os.path import join, exists
import random
import numpy as np

from utils.evaluation import RolloutGenerator
import torch
#seed 1,6 ,  \\ 123,7 works well for -90 ctrl 124,125,126
#seeding stuff
torch.manual_seed(126)
random.seed(126)
np.random.seed(126)

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
