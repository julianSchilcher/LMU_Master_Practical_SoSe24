import os 
import sys
from ray import tune
import nevergrad as ng

sys.path.append(os.getcwd())

from practical.DeepClustering.DipECT.dipect import _DipECT_Module

# load dataset + autoencoder? into global variable

# trainable function
# evaluate every x epochs

# searchspace
# logging