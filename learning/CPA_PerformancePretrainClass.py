#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


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
cmap = pd.read_csv('cmap_all_genes_q1_tas03.csv',index_col = 0)

gene_size = len(cmap.columns)
samples = cmap.index.values


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



NUM_EPOCHS= 1000
bs_1 = 1
bs_2 =  70
bs_paired =  60
class_criterion = torch.nn.CrossEntropyLoss()

# In[7]:


for i in range(1,2):
   # Network
   encoder = SimpleEncoder(gene_size,model_params['encoder_1_hiddens'],model_params['latent_dim'],
                             dropRate=model_params['dropout_encoder'],
                             activation=model_params['encoder_activation']).to(device)
   prior_d = PriorDiscriminator(model_params['latent_dim']).to(device)
   local_d = LocalDiscriminator(model_params['latent_dim'],model_params['latent_dim']).to(device)

   adverse_classifier = Classifier(in_channel=model_params['latent_dim'],
                                   hidden_layers=model_params['adv_class_hidden'],
                                   num_classes=model_params['no_adv_class'],
                                   drop_in=0.5,
                                   drop=0.3).to(device)

   trainInfo_paired = pd.read_csv('sampledDatasetes/pairedPercs/sample_ratio_683/train_paired_%s.csv' % i, index_col=None)
   trainInfo_1 = pd.read_csv('sampledDatasetes/pairedPercs/sample_ratio_683/train_A375_%s.csv' % i, index_col=None)
   trainInfo_2 = pd.read_csv('sampledDatasetes/pairedPercs/sample_ratio_683/train_HT29_%s.csv' % i, index_col=None)

   valInfo_paired = pd.read_csv('sampledDatasetes/pairedPercs/sample_ratio_683/val_paired_%s.csv' % i, index_col=None)
   valInfo_1 = pd.read_csv('sampledDatasetes/pairedPercs/sample_ratio_683/val_A375_%s.csv' % i, index_col=None)
   valInfo_2 = pd.read_csv('sampledDatasetes/pairedPercs/sample_ratio_683/val_HT29_%s.csv' % i, index_col=None)

   N_paired = len(trainInfo_paired)
   N_1 = len(trainInfo_1)
   N_2 = len(trainInfo_2)
   N = N_1
   if N_2>N:
       N=N_2

   allParams = list(encoder.parameters())
   allParams = allParams + list(prior_d.parameters()) + list(local_d.parameters())
   allParams = allParams + list(adverse_classifier.parameters())
   optimizer = torch.optim.Adam(allParams, lr= model_params['encoding_lr'], weight_decay=model_params['adversary_wd'])
   scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=model_params['schedule_step_enc'],
                                               gamma=model_params['gamma_enc'])
   for e in range(0, NUM_EPOCHS):
       encoder.train()
       prior_d.train()
       local_d.train()
       adverse_classifier.train()

       trainloader_1 = getSamples(N_1, bs_1)
       len_1 = len(trainloader_1)
       trainloader_2 = getSamples(N_2, bs_2)
       len_2 = len(trainloader_2)
       trainloader_paired = getSamples(N_paired, bs_paired)
       len_paired = len(trainloader_paired)

       lens = [len_1,len_2,len_paired]
       maxLen = np.max(lens)

       #print2log(maxLen)
       #print2log(lens)

       if maxLen>lens[0]:
           trainloader_suppl = getSamples(N_1, bs_1)
           for jj in range(maxLen-lens[0]):
               trainloader_1.insert(jj,trainloader_suppl[jj])

       if maxLen>lens[1]:
           trainloader_suppl = getSamples(N_2, bs_2)
           for jj in range(maxLen-lens[1]):
               trainloader_2.insert(jj,trainloader_suppl[jj])

       if maxLen>lens[2]:
           trainloader_suppl = getSamples(N_paired, bs_paired)
           #print2log(len(trainloader_suppl))
           for jj in range(maxLen-lens[2]):
               trainloader_paired.insert(jj,trainloader_suppl[jj])

       for j in range(maxLen):
           dataIndex_1 = trainloader_1[j]
           dataIndex_2 = trainloader_2[j]
           dataIndex_paired = trainloader_paired[j]

           df_pairs = trainInfo_paired.iloc[dataIndex_paired,:]
           df_1 = trainInfo_1.iloc[dataIndex_1,:]
           df_2 = trainInfo_2.iloc[dataIndex_2,:]
           paired_inds = len(df_pairs)


           X_1 = torch.tensor(np.concatenate((cmap.loc[df_pairs['sig_id.x']].values,
                                                cmap.loc[df_1.sig_id].values))).float().to(device)
           X_2 = torch.tensor(np.concatenate((cmap.loc[df_pairs['sig_id.y']].values,
                                                cmap.loc[df_2.sig_id].values))).float().to(device)


           conditions = np.concatenate((df_pairs.conditionId.values,
                                        df_1.conditionId.values,
                                        df_pairs.conditionId.values,
                                        df_2.conditionId.values))
           size = conditions.size
           conditions = conditions.reshape(size,1)
           conditions = conditions == conditions.transpose()
           conditions = conditions*1
           mask = torch.tensor(conditions).to(device).detach()
           pos_mask = mask
           neg_mask = 1 - mask
           log_2 = math.log(2.)
           optimizer.zero_grad()

           #if iteration % model_params["adversary_steps"] == 0:
           z_1 = encoder(X_1)
           z_2 = encoder(X_2)
           latent_vectors = torch.cat((z_1, z_2), 0)
           labels_adv = adverse_classifier(latent_vectors)
           true_labels = torch.cat((torch.ones(z_1.shape[0]),
                                    torch.zeros(z_2.shape[0])),0).long().to(device)
           adv_entropy = class_criterion(labels_adv,true_labels)
           _, predicted = torch.max(labels_adv, 1)
           predicted = predicted.cpu().numpy()
           cf_matrix = confusion_matrix(true_labels.cpu().numpy(),predicted)
           tn, fp, fn, tp = cf_matrix.ravel()
           f1 = 2*tp/(2*tp+fp+fn)


           #z_un = local_d(torch.cat((z_1, z_2), 0))
           z_un = local_d(latent_vectors)
           res_un = torch.matmul(z_un, z_un.t())

           p_samples = res_un * pos_mask.float()
           q_samples = res_un * neg_mask.float()

           Ep = log_2 - F.softplus(- p_samples)
           Eq = F.softplus(-q_samples) + q_samples - log_2

           Ep = (Ep * pos_mask.float()).sum() / pos_mask.float().sum()
           Eq = (Eq * neg_mask.float()).sum() / neg_mask.float().sum()
           mi_loss = Eq - Ep

           #prior = torch.rand_like(torch.cat((z_1, z_2), 0))
           prior = torch.rand_like(latent_vectors)

           term_a = torch.log(prior_d(prior)).mean()
           term_b = torch.log(1.0 - prior_d(latent_vectors)).mean()
           prior_loss = -(term_a + term_b) * model_params['prior_beta']

           # Remove signal from z_basal
           loss = mi_loss + prior_loss + adv_entropy + adverse_classifier.L2Regularization(model_params['state_class_reg']) +encoder.L2Regularization(model_params['enc_l2_reg'])


           loss.backward()
           optimizer.step()

       scheduler.step()
       outString = 'Split {:.0f}: Epoch={:.0f}/{:.0f}'.format(i+1,e+1,NUM_EPOCHS)
       outString += ', MI Loss={:.4f}'.format(mi_loss.item())
       outString += ', Prior Loss={:.4f}'.format(prior_loss.item())
       outString += ', Entropy Loss={:.4f}'.format(adv_entropy.item())
       outString += ', loss={:.4f}'.format(loss.item())
       outString += ', F1 score={:.4f}'.format(f1)
       if (e%250==0):
           print2log(outString)
   print2log(outString)
   encoder.eval()
   prior_d.eval()
   local_d.eval()
   adverse_classifier.eval()

   paired_val_inds = len(valInfo_paired)
   x_1 = torch.tensor(np.concatenate((cmap.loc[valInfo_paired['sig_id.x']].values,
                                         cmap.loc[valInfo_1.sig_id].values))).float().to(device)
   x_2 = torch.tensor(np.concatenate((cmap.loc[valInfo_paired['sig_id.y']].values,
                                         cmap.loc[valInfo_2.sig_id].values))).float().to(device)


   z_latent_1 = encoder(x_1)
   z_latent_2 = encoder(x_2)


   labels = adverse_classifier(torch.cat((z_latent_1, z_latent_2), 0))
   true_labels = torch.cat((torch.ones(z_latent_1.shape[0]).view(z_latent_1.shape[0],1),
                            torch.zeros(z_latent_2.shape[0]).view(z_latent_2.shape[0],1)),0).long()
   _, predicted = torch.max(labels, 1)
   predicted = predicted.cpu().numpy()
   cf_matrix = confusion_matrix(true_labels.numpy(),predicted)
   tn, fp, fn, tp = cf_matrix.ravel()
   class_acc = (tp+tn)/predicted.size
   f1 = 2*tp/(2*tp+fp+fn)

   print2log('Classification accuracy: %s'%class_acc)
   print2log('Classification F1 score: %s'%f1)

   #torch.save(encoder,'MI_results/models/CPA_approach/pre_trained_master_encoder_%s.pt'%i)
   torch.save(adverse_classifier,'sampledDatasetes/pairedPercs/pre_trained_classifier_adverse_%s.pt'%i)

