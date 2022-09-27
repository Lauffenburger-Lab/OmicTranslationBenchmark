from __future__ import absolute_import, division
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from trainingUtils import MultipleOptimizer, MultipleScheduler, compute_kernel, compute_mmd
from models import Decoder, SimpleEncoder,LocalDiscriminator,PriorDiscriminator,Classifier,SpeciesCovariate
# import argparse
import math
import numpy as np
import pandas as pd
import sys
import random
import os
from sklearn.metrics import silhouette_score,confusion_matrix
from scipy.stats import spearmanr
from evaluationUtils import r_square,get_cindex,pearson_r,pseudoAccuracy
import seaborn as sns
sns.set()
import logging
from captum.attr import IntegratedGradients
from captum.attr import LayerConductance
from captum.attr import NeuronConductance

device = torch.device('cuda')

torch.backends.cudnn.benchmark = True
def seed_everything(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False


encoder_1=torch.load('alldata_cpa_encoder_pc3.pt')
encoder_2=torch.load('alldata_cpa_encoder_ha1e.pt')
classifier = torch.load('alldata_cpa_classifier_pc3_ha1e.pt')
Vsp = torch.load('alldata_cpa_Vsp_pc3_ha1e.pt')

torch.save(encoder_1.state_dict(),'alldata_cpa_encoder_pc3_state_dict.pth')
torch.save(encoder_2.state_dict(),'alldata_cpa_encoder_ha1e_state_dict.pth')
torch.save(classifier.state_dict(),'alldata_cpa_classifier_pc3_ha1e_state_dict.pth')
torch.save(Vsp.state_dict(),'alldata_cpa_Vsp_pc3_ha1e_state_dict.pth')
