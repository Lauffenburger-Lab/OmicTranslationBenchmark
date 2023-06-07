#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import, division
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from models_satori import Decoder, SimpleEncoder,SpeciesCovariate
# import argparse
import math
import numpy as np
import pandas as pd
import sys
import random
import os
#from IPython.display import clear_output
#from matplotlib import pyplot as plt

#from scipy.stats import pearsonr
from sklearn.metrics import silhouette_score,confusion_matrix
from scipy.stats import spearmanr
from evaluationUtils import r_square,get_cindex,pearson_r,pseudoAccuracy
#import seaborn as sns
#sns.set()
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
print2log = logger.info

device = torch.device('cuda')


# In[3]:


# Initialize environment and seeds for reproducability
torch.backends.cudnn.benchmark = True


def seed_everything(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    
# Read data
#cmap = pd.read_csv('cmap_landmarks_HT29_A375.csv',index_col = 0)
# cmap = pd.read_csv('cmap_landmarks_HA1E_PC3.csv',index_col = 0)
cmap = pd.read_csv('../preprocessing/preprocessed_data/cmap_HT29_A375.csv',index_col = 0)
#cmap_tf = pd.read_csv('../L1000_2021_11_23/cmap_compounds_tfs_repq1_tas03.tsv',
#                       sep='\t', low_memory=False, index_col=0)

gene_size = len(cmap.columns)
samples = cmap.index.values
# gene_size = len(cmap_tf.columns)
# samples = cmap_tf.index.values

# sampleInfo = pd.read_csv('conditions_HT29_A375.csv',index_col = 0)


# In[4]:


# Create a train generators
def getSamples(N, batchSize):
    order = np.random.permutation(N)
    outList = []
    while len(order)>0:
        outList.append(order[0:batchSize])
        order = order[batchSize:]
    return outList


# # Train model

# In[5]:


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

valClassAcc = []
valF1 = []
for i in range(10):
    # Network
    encoder_1 = torch.load('../results/MI_results/models/HT29_A375_withclass/encoder_pc3_%s.pt'%i)
    encoder_2 = torch.load('../results/MI_results/models/HT29_A375_withclass/encoder_ha1e_%s.pt'%i)

    trainInfo_paired = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/train_paired_%s.csv'%i,index_col=0)
    trainInfo_1 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/train_a375_%s.csv'%i,index_col=0)
    trainInfo_2 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/train_ht29_%s.csv'%i,index_col=0)
    
    valInfo_paired = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/val_paired_%s.csv'%i,index_col=0)
    valInfo_1 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/val_a375_%s.csv'%i,index_col=0)
    valInfo_2 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/val_ht29_%s.csv'%i,index_col=0)

    encoder_1.eval()
    encoder_2.eval()

    paired_val_inds = len(valInfo_paired)
    x_1 = torch.tensor(np.concatenate((cmap.loc[valInfo_paired['sig_id.x']].values,
                                          cmap.loc[valInfo_1.sig_id].values))).float().to(device)
    x_2 = torch.tensor(np.concatenate((cmap.loc[valInfo_paired['sig_id.y']].values,
                                          cmap.loc[valInfo_2.sig_id].values))).float().to(device)
    
    z_latent_1 = encoder_1(x_1)
    z_latent_2 = encoder_2(x_2)

    ### Save validaiton embeddings ###
    Embs_1 = pd.DataFrame(z_latent_1.detach().cpu().numpy())
    Embs_1.index = sampleInfo_paired['sig_id.x'].values
    Embs_1.columns = ['z' + str(i) for i in range(model_params['latent_dim'])]
    Embs_1.to_csv('../results/MI_results/embs/HT29_A375_withclass/valEmbs_a375_withclass_%s.csv'%i)

    Embs_2 = pd.DataFrame(z_latent_2.detach().cpu().numpy())
    Embs_2.index = sampleInfo_paired['sig_id.y'].values
    Embs_2.columns = ['z' + str(i) for i in range(model_params['latent_dim'])]
    Embs_2.to_csv('../results/MI_results/embs/HT29_A375_withclass/valEmbs_ht29_withclass_%s.csv'%i)

    x_1 = torch.tensor(np.concatenate((cmap.loc[trainInfo_paired['sig_id.x']].values,
                                       cmap.loc[trainInfo_1.sig_id].values))).float().to(device)
    x_2 = torch.tensor(np.concatenate((cmap.loc[trainInfo_paired['sig_id.y']].values,
                                       cmap.loc[trainInfo_2.sig_id].values))).float().to(device)

    z_latent_1 = encoder_1(x_1)
    z_latent_2 = encoder_2(x_2)

    ### Save training embeddings ###
    Embs_1 = pd.DataFrame(z_latent_1.detach().cpu().numpy())
    Embs_1.index = sampleInfo_paired['sig_id.x'].values
    Embs_1.columns = ['z' + str(i) for i in range(model_params['latent_dim'])]
    Embs_1.to_csv('../results/MI_results/embs/HT29_A375_withclass/trainEmbs_a375_withclass_%s.csv' % i)

    Embs_2 = pd.DataFrame(z_latent_2.detach().cpu().numpy())
    Embs_2.index = sampleInfo_paired['sig_id.y'].values
    Embs_2.columns = ['z' + str(i) for i in range(model_params['latent_dim'])]
    Embs_2.to_csv('../results/MI_results/embs/HT29_A375_withclass/trainEmbs_ht29_withclass_%s.csv' % i)

    print2log('Finished %s'%i)
