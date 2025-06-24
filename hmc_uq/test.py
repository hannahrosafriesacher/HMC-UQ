import os
import yaml
import argparse

import numpy as np
import torch
import pyro
from pyro.infer import MCMC, NUTS
from pyro.infer import Predictive
from torch.utils.data import DataLoader
from collections import OrderedDict

from utils.load_config import get_args
from utils.models import MLP_pyro
from utils.evaluation import HMCPredictivePerformance, HMCSampleEvaluation
from utils.data import SparseDataset
import wandb

from timeit import default_timer as timer


hidden_sizes = 2
num_input_features = 100
prior_scale = 0.1
device = 'cpu'
net = MLP_pyro(
        input_features=num_input_features, 
        output_features=1,
        hidden_sizes=hidden_sizes,          
        prior_scale=prior_scale, 
        device=device
        )

