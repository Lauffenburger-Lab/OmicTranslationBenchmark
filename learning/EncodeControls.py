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

cmap = pd.read_csv('../preprocessing/preprocessed_data/cmap_untreated_untreated_q1_landmarks.csv',index_col = 0)

gene_size = len(cmap.columns)
samples = cmap.index.values

sampleInfo_1 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/alldata/pc3_unpaired_untreated.csv',index_col=0)
sampleInfo_2 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/alldata/ha1e_unpaired_untreated.csv',index_col=0)
sampleInfo_paired = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/alldata/paired_untreated_pc3_ha1e.csv',index_col=0)


# model_params = {'encoder_1_hiddens':[4096,2048],
#                 'encoder_2_hiddens':[4096,2048],
#                 'latent_dim': 1024,
#                 'decoder_1_hiddens':[2048,4096],
#                 'decoder_2_hiddens':[2048,4096],
#                 'dropout_decoder':0.2,
#                 'dropout_encoder':0.1,
#                 'encoder_activation':torch.nn.ELU(),
#                 'decoder_activation':torch.nn.ELU(),
#                 'V_dropout':0.25,
#                 'state_class_hidden':[512,256,128],
#                 'state_class_drop_in':0.5,
#                 'state_class_drop':0.25,
#                 'no_states':2,
#                 'adv_class_hidden':[512,256,128],
#                 'adv_class_drop_in':0.3,
#                 'adv_class_drop':0.1,
#                 'no_adv_class':2,
#                 'encoding_lr':0.001,
#                 'adv_lr':0.001,
#                 'schedule_step_adv':300,
#                 'gamma_adv':0.5,
#                 'schedule_step_enc':300,
#                 'gamma_enc':0.8,
#                 'batch_size_1':178,
#                 'batch_size_2':154,
#                 'batch_size_paired':90,
#                 'epochs':1000,
#                 'prior_beta':1.0,
#                 'no_folds':10,
#                 'v_reg':1e-04,
#                 'state_class_reg':1e-02,
#                 'enc_l2_reg':0.01,
#                 'dec_l2_reg':0.01,
#                 'lambda_mi_loss':100,
#                 'effsize_reg': 10,
#                 'cosine_loss': 40,
#                 'adv_penalnty':50,
#                 'reg_adv':500,
#                 'reg_classifier': 500,
#                 'similarity_reg' : 1.,
#                 'adversary_steps':5,
#                 'autoencoder_wd': 0,
#                 'adversary_wd': 0}

model_params = {'encoder_1_hiddens':[640,384],
                'encoder_2_hiddens':[640,384],
                'latent_dim': 292,
                'decoder_1_hiddens':[384,640],
                'decoder_2_hiddens':[384,640],
                'dropout_decoder':0.2,
                'dropout_encoder':0.1,
                'encoder_activation':torch.nn.ELU(),
                'decoder_activation':torch.nn.ELU(),
                'V_dropout':0.25,
                'state_class_hidden':[256,128,64],
                'state_class_drop_in':0.5,
                'state_class_drop':0.25,
                'no_states':2,
                'adv_class_hidden':[256,128,64],
                'adv_class_drop_in':0.3,
                'adv_class_drop':0.1,
                'no_adv_class':2,
                'encoding_lr':0.001,
                'adv_lr':0.001,
                'schedule_step_adv':200,
                'gamma_adv':0.5,
                'schedule_step_enc':200,
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
                'effsize_reg': 100,
                'cosine_loss': 10,
                'adv_penalnty':100,
                'reg_adv':1000,
                'reg_classifier': 1000,
                'similarity_reg' : 10,
                'adversary_steps':4,
                'autoencoder_wd': 0.,
                'adversary_wd': 0.}


# encoder_1 = SimpleEncoder(gene_size, model_params['encoder_1_hiddens'], model_params['latent_dim'],
#                           dropRate=model_params['dropout_encoder'],
#                           activation=model_params['encoder_activation']).to(device)
# encoder_2 = SimpleEncoder(gene_size, model_params['encoder_2_hiddens'], model_params['latent_dim'],
#                           dropRate=model_params['dropout_encoder'],
#                           activation=model_params['encoder_activation']).to(device)
#
# Vsp = SpeciesCovariate(2, model_params['latent_dim'], dropRate=model_params['V_dropout']).to(device)

encoder_1=torch.load('../results/trained_models/alldata_landmarks_cpa_encoder_pc3.pt')
encoder_2=torch.load('../results/trained_models/alldata_landmarks_cpa_encoder_ha1e.pt')
classifier = torch.load('../results/trained_models/alldata_landmarks_cpa_classifier_pc3_ha1e.pt')
Vsp = torch.load('../results/trained_models/alldata_landmarks_cpa_Vsp_pc3_ha1e.pt')

encoder_1.eval()
encoder_2.eval()
Vsp.eval()
print2log('Evaluation mode')

# x_1 = torch.tensor(np.concatenate((cmap.loc[sampleInfo_paired['sig_id.x']].values,
#                                       cmap.loc[sampleInfo_1.sig_id].values))).float().to(device)
# x_2 = torch.tensor(np.concatenate((cmap.loc[sampleInfo_paired['sig_id.y']].values,
#                                       cmap.loc[sampleInfo_2.sig_id].values))).float().to(device)

x_1 = torch.tensor(cmap.loc[sampleInfo_paired['sig_id.x']].values).float().to(device)
x_2 = torch.tensor(cmap.loc[sampleInfo_paired['sig_id.y']].values).float().to(device)

z_species_1 = torch.cat((torch.ones(x_1.shape[0], 1),
                         torch.zeros(x_1.shape[0], 1)), 1).to(device)
z_species_2 = torch.cat((torch.zeros(x_2.shape[0], 1),
                         torch.ones(x_2.shape[0], 1)), 1).to(device)

z_base_latent_1  = encoder_1(x_1)
z_latent_1 = Vsp(z_base_latent_1, z_species_1)
z_base_latent_2  = encoder_2(x_2)
z_latent_2 = Vsp(z_base_latent_2, z_species_2)

### Save basal embeddings ###
Embs_base_1 = pd.DataFrame(z_base_latent_1.detach().cpu().numpy())
# Embs_base_1.index = np.concatenate((sampleInfo_paired['sig_id.x'].values,sampleInfo_1.sig_id.values))
Embs_base_1.index = sampleInfo_paired['sig_id.x'].values
Embs_base_1.columns = ['z'+str(i) for i in range(model_params['latent_dim'])]
Embs_base_1.to_csv('../results/trained_embs_all/ControlsBasalEmbs_landmarks_CPA_pc3.csv')

Embs_base_2 = pd.DataFrame(z_base_latent_2.detach().cpu().numpy())
# Embs_base_2.index = np.concatenate((sampleInfo_paired['sig_id.y'].values,sampleInfo_2.sig_id.values))
Embs_base_2.index = sampleInfo_paired['sig_id.y'].values
Embs_base_2.columns = ['z'+str(i) for i in range(model_params['latent_dim'])]
Embs_base_2.to_csv('../results/trained_embs_all/ControlsBasalEmbs_landmarks_CPA_ha1e.csv')


### Save embeddings ###
Embs_1 = pd.DataFrame(z_latent_1.detach().cpu().numpy())
# Embs_1.index = np.concatenate((sampleInfo_paired['sig_id.x'].values,sampleInfo_1.sig_id.values))
Embs_1.index = sampleInfo_paired['sig_id.x'].values
Embs_1.columns = ['z'+str(i) for i in range(model_params['latent_dim'])]
Embs_1.to_csv('../results/trained_embs_all/ControlsEmbs_landmarks_CPA_pc3.csv')

Embs_2 = pd.DataFrame(z_latent_2.detach().cpu().numpy())
# Embs_2.index = np.concatenate((sampleInfo_paired['sig_id.y'].values,sampleInfo_2.sig_id.values))
Embs_2.index = sampleInfo_paired['sig_id.y'].values
Embs_2.columns = ['z'+str(i) for i in range(model_params['latent_dim'])]
Embs_2.to_csv('../results/trained_embs_all/ControlsEmbs_landmarks_CPA_ha1e.csv')

### Trained V matrix
Vsp.eval()
v1 = Vsp.Vspecies(z_species_1[0:1,:])
v2 = Vsp.Vspecies(z_species_2[0:1,:])
v = torch.cat((v1,v2),0)
vEmbs = pd.DataFrame(v.detach().cpu().numpy())
print2log(v.shape)
vEmbs.index = ['PC3','HA1E']
vEmbs.columns = ['z'+str(i) for i in range(model_params['latent_dim'])]
vEmbs.to_csv('../results/trained_embs_all/CellCovariate_landmarks_CPA_pc3_ha1e.csv')
