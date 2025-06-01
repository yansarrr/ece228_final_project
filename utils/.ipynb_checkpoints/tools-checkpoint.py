# utils/tools.py

import torch
import random
import numpy as np

def seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
