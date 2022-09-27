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
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from evaluationUtils import r_square,get_cindex,pearson_r,pseudoAccuracy
#import seaborn as sns
#sns.set()
import logging
# from captum.attr import IntegratedGradients
# from captum.attr import LayerConductance
# from captum.attr import NeuronConductance


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
print2log = logger.info

device = torch.device('cuda')

torch.backends.cudnn.benchmark = True


def seed_everything(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False

ccle = pd.read_csv('ccle_l1000genes.csv',index_col = 0).transpose()
cells = ['A375','HT29']
ccle = ccle.loc[cells,:]

gene_size = len(ccle.columns)
samples = ccle.index.values
genes = ccle.columns

std_scaler = StandardScaler()
ccle_scaled = std_scaler.fit_transform(ccle.to_numpy())
ccle = pd.DataFrame(ccle_scaled, columns=genes)
ccle.index = samples

model_params = {'encoder_1_hiddens':[4096,2048],
                'encoder_2_hiddens':[4096,2048],
                'latent_dim': 1024,
                'decoder_1_hiddens':[2048,4096],
                'decoder_2_hiddens':[2048,4096],
                'dropout_decoder':0.2,
                'dropout_encoder':0.1,
                'encoder_activation':torch.nn.ELU(),
                'decoder_activation':torch.nn.ELU(),
                'V_dropout':0.25,
                'state_class_hidden':[512,256,128],
                'state_class_drop_in':0.5,
                'state_class_drop':0.25,
                'no_states':2,
                'adv_class_hidden':[512,256,128],
                'adv_class_drop_in':0.3,
                'adv_class_drop':0.1,
                'no_adv_class':2,
                'encoding_lr':0.001,
                'adv_lr':0.001,
                'schedule_step_adv':300,
                'gamma_adv':0.5,
                'schedule_step_enc':300,
                'gamma_enc':0.8,
                'batch_size_1':178,
                'batch_size_2':154,
                'batch_size_paired':90,
                'epochs':1000,
                'prior_beta':1.0,
                'no_folds':10,
                'v_reg':1e-04,
                'state_class_reg':1e-02,
                'enc_l2_reg':0.01,
                'dec_l2_reg':0.01,
                'lambda_mi_loss':100,
                'effsize_reg': 10,
                'cosine_loss': 40,
                'adv_penalnty':50,
                'reg_adv':500,
                'reg_classifier': 500,
                'similarity_reg' : 1.,
                'adversary_steps':5,
                'autoencoder_wd': 0,
                'adversary_wd': 0}


encoder_1 = SimpleEncoder(gene_size, model_params['encoder_1_hiddens'], model_params['latent_dim'],
                          dropRate=model_params['dropout_encoder'],
                          activation=model_params['encoder_activation']).to(device)
encoder_2 = SimpleEncoder(gene_size, model_params['encoder_2_hiddens'], model_params['latent_dim'],
                          dropRate=model_params['dropout_encoder'],
                          activation=model_params['encoder_activation']).to(device)

Vsp = SpeciesCovariate(2, model_params['latent_dim'], dropRate=model_params['V_dropout']).to(device)

encoder_1=torch.load('trained_models/alldata_cpa_encoder_a375.pt')
encoder_2=torch.load('trained_models/alldata_cpa_encoder_ht29.pt')
Vsp = torch.load('trained_models/alldata_cpa_Vsp_a375_ht29.pt')

encoder_1.eval()
encoder_2.eval()
Vsp.eval()
print2log('Evaluation mode')

# x_1 = torch.tensor(np.concatenate((cmap.loc[sampleInfo_paired['sig_id.x']].values,
#                                       cmap.loc[sampleInfo_1.sig_id].values))).float().to(device)
# x_2 = torch.tensor(np.concatenate((cmap.loc[sampleInfo_paired['sig_id.y']].values,
#                                       cmap.loc[sampleInfo_2.sig_id].values))).float().to(device)

x = torch.tensor(ccle.values).float().to(device)

z_species_1 = torch.tensor([[1,0]]).float().to(device)
z_species_2 = torch.tensor([[0,1]]).float().to(device)
z_base_latent_1  = encoder_1(x[0,:].view(1,gene_size))
z_latent_1 = Vsp(z_base_latent_1, z_species_1)
z_base_latent_2  = encoder_2(x[1,:].view(1,gene_size))
z_latent_2 = Vsp(z_base_latent_2, z_species_2)

### Save basal embeddings ###
Embs_base_1 = pd.DataFrame(z_base_latent_1.detach().cpu().numpy())
# Embs_base_1.index = np.concatenate((sampleInfo_paired['sig_id.x'].values,sampleInfo_1.sig_id.values))
Embs_base_1.index = [cells[0]]
Embs_base_1.columns = ['z'+str(i) for i in range(model_params['latent_dim'])]
Embs_base_1.to_csv('trained_embs_all/ControlsBasalCCLE_CPA_a375.csv')

Embs_base_2 = pd.DataFrame(z_base_latent_2.detach().cpu().numpy())
# Embs_base_2.index = np.concatenate((sampleInfo_paired['sig_id.y'].values,sampleInfo_2.sig_id.values))
Embs_base_2.index = [cells[1]]
Embs_base_2.columns = ['z'+str(i) for i in range(model_params['latent_dim'])]
Embs_base_2.to_csv('trained_embs_all/ControlsBasalCCLE_CPA_ht29.csv')


### Save embeddings ###
Embs_1 = pd.DataFrame(z_latent_1.detach().cpu().numpy())
# Embs_1.index = np.concatenate((sampleInfo_paired['sig_id.x'].values,sampleInfo_1.sig_id.values))
Embs_1.index = [cells[0]]
Embs_1.columns = ['z'+str(i) for i in range(model_params['latent_dim'])]
Embs_1.to_csv('trained_embs_all/ControlsCCLE_CPA_a375.csv')

Embs_2 = pd.DataFrame(z_latent_2.detach().cpu().numpy())
# Embs_2.index = np.concatenate((sampleInfo_paired['sig_id.y'].values,sampleInfo_2.sig_id.values))
Embs_2.index = [cells[1]]
Embs_2.columns = ['z'+str(i) for i in range(model_params['latent_dim'])]
Embs_2.to_csv('trained_embs_all/ControlsCCLE_CPA_ht29.csv')
