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
                'no_folds':5,
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


def compute_gradients(output, input):
    grads = torch.autograd.grad(output, input, create_graph=True)
    grads = grads[0].pow(2).mean()
    return grads


# In[9]:


class_criterion = torch.nn.CrossEntropyLoss()
NUM_EPOCHS= model_params['epochs']
#bs = 512
bs_1 = model_params['batch_size_1']
bs_2 =  model_params['batch_size_2']
bs_paired =  model_params['batch_size_paired']


# In[10]:


# NUM_EPOCHS = 300
# #bs = 128
# bs_a375 = 45
# bs_ht29 = 39
# bs_paired = 22
# beta = 1.0


# In[11]:


valR2 = []
valPear = []
valMSE =[]
valSpear = []
valAccuracy = []


valPearDirect = []
valSpearDirect = []
valAccDirect = []

valR2_1 = []
valPear_1 = []
valMSE_1 =[]
valSpear_1 = []
valAccuracy_1 = []

valR2_2 = []
valPear_2 = []
valMSE_2 =[]
valSpear_2 = []
valAccuracy_2 = []

crossCorrelation = []

valF1 = []
valClassAcc = []

for i in range(model_params['no_folds']):
    # Network
    #encoder = torch.load('../results/MI_results/models/CPA_approach/pre_trained_master_encoder_2.pt')
    #master_encoder = SimpleEncoder(gene_size,[640,384],292,dropRate=0.1, activation=torch.nn.ELU())#.to(device)
    encoder_1 = SimpleEncoder(gene_size,model_params['encoder_1_hiddens'],model_params['latent_dim'],
                              dropRate=model_params['dropout_encoder'], 
                              activation=model_params['encoder_activation']).to(device)
    #encoder_1.load_state_dict(encoder.state_dict())
    encoder_2 = SimpleEncoder(gene_size,model_params['encoder_2_hiddens'],model_params['latent_dim'],
                                  dropRate=model_params['dropout_encoder'], 
                                  activation=model_params['encoder_activation']).to(device)
    #encoder_2.load_state_dict(encoder.state_dict())
    prior_d = PriorDiscriminator(model_params['latent_dim']).to(device)
    local_d = LocalDiscriminator(model_params['latent_dim'],model_params['latent_dim']).to(device)
    
    classifier = Classifier(in_channel=model_params['latent_dim'],
                            hidden_layers=model_params['state_class_hidden'],
                            num_classes=model_params['no_states'],
                            drop_in=model_params['state_class_drop_in'],
                            drop=model_params['state_class_drop']).to(device)
    adverse_classifier = Classifier(in_channel=model_params['latent_dim'],
                                    hidden_layers=model_params['adv_class_hidden'],
                                    num_classes=model_params['no_adv_class'],
                                    drop_in=model_params['adv_class_drop_in'],
                                    drop=model_params['adv_class_drop']).to(device)
    pretrained_adv_class = torch.load('../results/MI_results/models/CPA_approach/5fold_validation_spit/pre_trained_classifier_adverse_1.pt')
    adverse_classifier.load_state_dict(pretrained_adv_class.state_dict())
    
    Vsp = SpeciesCovariate(2,model_params['latent_dim'],dropRate=model_params['V_dropout']).to(device)
    
    trainInfo_paired = pd.read_csv('../preprocessing/preprocessed_data/5fold_validation_spit/train_paired_%s.csv'%i,index_col=0)#.sample(frac=0.8)
    trainInfo_1 = pd.read_csv('../preprocessing/preprocessed_data/5fold_validation_spit/train_a375_%s.csv'%i,index_col=0)#.sample(frac=0.8)
    trainInfo_2 = pd.read_csv('../preprocessing/preprocessed_data/5fold_validation_spit/train_ht29_%s.csv'%i,index_col=0)#.sample(frac=0.8)
    
    valInfo_paired = pd.read_csv('../preprocessing/preprocessed_data/5fold_validation_spit/val_paired_%s.csv'%i,index_col=0)
    valInfo_1 = pd.read_csv('../preprocessing/preprocessed_data/5fold_validation_spit/val_a375_%s.csv'%i,index_col=0)
    valInfo_2 = pd.read_csv('../preprocessing/preprocessed_data/5fold_validation_spit/val_ht29_%s.csv'%i,index_col=0)

    #trainInfo_2 = pd.concat((trainInfo_2,
    #                         valInfo_2,
    #                         valInfo_paired.loc[:,['sig_id.y','cell_iname.y','conditionId']].rename(columns = {'sig_id.y':'sig_id','cell_iname.y':'cell_iname'})),0)
    
    N_paired = len(trainInfo_paired)
    N_1 = len(trainInfo_1)
    N_2 = len(trainInfo_2)
    N = N_1
    if N_2>N:
        N=N_2
    
    allParams = list(encoder_1.parameters()) +list(encoder_2.parameters())
    allParams = allParams + list(prior_d.parameters()) + list(local_d.parameters())
    allParams = allParams + list(classifier.parameters())
    allParams = allParams + list(Vsp.parameters())
    optimizer = torch.optim.Adam(allParams, lr= model_params['encoding_lr'], weight_decay=0)
    optimizer_adv = torch.optim.Adam(adverse_classifier.parameters(), lr= model_params['adv_lr'], weight_decay=0)
    if model_params['schedule_step_adv'] is not None:
        scheduler_adv = torch.optim.lr_scheduler.StepLR(optimizer_adv,
                                                        step_size=model_params['schedule_step_adv'],
                                                        gamma=model_params['gamma_adv'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=model_params['schedule_step_enc'],
                                                gamma=model_params['gamma_enc'])
    iteration = 1
    for e in range(0, NUM_EPOCHS):
        encoder_1.train()
        encoder_2.train()
        prior_d.train()
        local_d.train()
        classifier.train()
        adverse_classifier.train()
        Vsp.train()
        #master_encoder.train()
        
        trainloader_1 = getSamples(N_1, bs_1)
        len_1 = len(trainloader_1)
        trainloader_2 = getSamples(N_2, bs_2)
        len_2 = len(trainloader_2)
        trainloader_paired = getSamples(N_paired, bs_paired)
        len_paired = len(trainloader_paired)

        lens = [len_1,len_2,len_paired]
        maxLen = np.max(lens)
        
        iteration = 1

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
            for jj in range(maxLen-lens[2]):
                trainloader_paired.insert(jj,trainloader_suppl[jj])
        #for dataIndex in trainloader:
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
            
            z_species_1 = torch.cat((torch.ones(X_1.shape[0],1),
                                     torch.zeros(X_1.shape[0],1)),1).to(device)
            z_species_2 = torch.cat((torch.zeros(X_2.shape[0],1),
                                     torch.ones(X_2.shape[0],1)),1).to(device)
            
            
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
            optimizer_adv.zero_grad()
                        
            #if e % model_params['adversary_steps']==0:      
            z_base_1 = encoder_1(X_1)
            z_base_2 = encoder_2(X_2)
            latent_base_vectors = torch.cat((z_base_1, z_base_2), 0)
            labels_adv = adverse_classifier(latent_base_vectors)
            true_labels = torch.cat((torch.ones(z_base_1.shape[0]),
                torch.zeros(z_base_2.shape[0])),0).long().to(device)
            _, predicted = torch.max(labels_adv, 1)
            predicted = predicted.cpu().numpy()
            cf_matrix = confusion_matrix(true_labels.cpu().numpy(),predicted)
            tn, fp, fn, tp = cf_matrix.ravel()
            f1_basal_trained = 2*tp/(2*tp+fp+fn)
            adv_entropy = class_criterion(labels_adv,true_labels)
            adversary_drugs_penalty = compute_gradients(labels_adv.sum(), latent_base_vectors)
            loss_adv = adv_entropy + model_params['adv_penalnty'] * adversary_drugs_penalty
            loss_adv.backward()
            optimizer_adv.step()
            #print(f1_basal_trained)
            #else:
            optimizer.zero_grad()
            #f1_basal_trained = None
            z_base_1 = encoder_1(X_1)
            z_base_2 = encoder_2(X_2)
            latent_base_vectors = torch.cat((z_base_1, z_base_2), 0)
            
            #z_un = local_d(torch.cat((z_1, z_2), 0))
            z_un = local_d(latent_base_vectors)
            res_un = torch.matmul(z_un, z_un.t())
            
            z_1 = Vsp(z_base_1,z_species_1)
            z_2 = Vsp(z_base_2,z_species_2)
            latent_vectors = torch.cat((z_1, z_2), 0)

            L2Loss_1 = encoder_1.L2Regularization(model_params['enc_l2_reg'])
            # loss_1 = fitLoss_1 + L2Loss_1
            
            L2Loss_2 = encoder_2.L2Regularization(model_params['enc_l2_reg'])
            # loss_2 = fitLoss_2 + L2Loss_2

            #silimalityLoss = np.sqrt(paired_inds)*torch.mean(torch.sum((z_base_1[0:paired_inds,:] - z_base_2[0:paired_inds,:])**2,
            #                                      dim=-1))/torch.std(torch.sum((z_base_1[0:paired_inds,:] - z_base_2[0:paired_inds,:])**2,
            #                                                                   dim=-1))
            silimalityLoss = torch.mean(torch.sum((z_base_1[0:paired_inds,:] - z_base_2[0:paired_inds,:])**2,dim=-1))
            cosineLoss = torch.nn.functional.cosine_similarity(z_base_1[0:paired_inds,:],z_base_2[0:paired_inds,:],dim=-1).mean()
            
            p_samples = res_un * pos_mask.float()
            q_samples = res_un * neg_mask.float()
    
            Ep = log_2 - F.softplus(- p_samples)
            Eq = F.softplus(-q_samples) + q_samples - log_2

            Ep = (Ep * pos_mask.float()).sum() / pos_mask.float().sum()
            Eq = (Eq * neg_mask.float()).sum() / neg_mask.float().sum()
            mi_loss = Eq - Ep

            #prior = torch.rand_like(torch.cat((z_1, z_2), 0))
            prior = torch.rand_like(latent_base_vectors)

            term_a = torch.log(prior_d(prior)).mean()
            #term_b = torch.log(1.0 - prior_d(torch.cat((z_1, z_2), 0))).mean()
            term_b = torch.log(1.0 - prior_d(latent_base_vectors)).mean()
            prior_loss = -(term_a + term_b) * model_params['prior_beta']

            # Classification loss
            labels = classifier(latent_vectors)
            true_labels = torch.cat((torch.ones(z_1.shape[0]),
                torch.zeros(z_2.shape[0])),0).long().to(device)
            entropy = class_criterion(labels,true_labels)
            _, predicted = torch.max(labels, 1)
            predicted = predicted.cpu().numpy()
            cf_matrix = confusion_matrix(true_labels.cpu().numpy(),predicted)
            tn, fp, fn, tp = cf_matrix.ravel()
            f1_latent = 2*tp/(2*tp+fp+fn)
            
            # Remove signal from z_basal
            labels_adv = adverse_classifier(latent_base_vectors)
            true_labels = torch.cat((torch.ones(z_base_1.shape[0]),
                torch.zeros(z_base_2.shape[0])),0).long().to(device)
            adv_entropy = class_criterion(labels_adv,true_labels)
            _, predicted = torch.max(labels_adv, 1)
            predicted = predicted.cpu().numpy()
            cf_matrix = confusion_matrix(true_labels.cpu().numpy(),predicted)
            tn, fp, fn, tp = cf_matrix.ravel()
            f1_basal = 2*tp/(2*tp+fp+fn)
            
            #loss = loss_1 + loss_2 + 100*mi_loss + prior_loss + 10*silimalityLoss-100*cosineLoss + 1000*entropy +\
            #       classifier.L2Regularization(1e-2) + Vsp.Regularization() - 1000*adv_entropy 
            loss = L2Loss_1 + L2Loss_2 + model_params['similarity_reg'] * silimalityLoss +model_params['lambda_mi_loss']*mi_loss + prior_loss  + model_params['reg_classifier'] * entropy - model_params['reg_adv']*adv_entropy +classifier.L2Regularization(model_params['state_class_reg']) +Vsp.Regularization(model_params['v_reg'])  - model_params['cosine_loss'] * cosineLoss

            loss.backward()
            optimizer.step()

        if model_params['schedule_step_adv'] is not None:
            scheduler_adv.step()
        if (e>=0):
            scheduler.step()
            outString = 'Split {:.0f}: Epoch={:.0f}/{:.0f}'.format(i+1,e+1,NUM_EPOCHS)
            outString += ', MI Loss={:.4f}'.format(mi_loss.item())
            outString += ', Prior Loss={:.4f}'.format(prior_loss.item())
            outString += ', Entropy Loss={:.4f}'.format(entropy.item())
            outString += ', Adverse Entropy={:.4f}'.format(adv_entropy.item())
            outString += ', Cosine Loss={:.4f}'.format(cosineLoss.item())
            outString += ', loss={:.4f}'.format(loss.item())
            outString += ', F1 latent={:.4f}'.format(f1_latent)
            outString += ', F1 basal={:.4f}'.format(f1_basal)
            #if e % model_params["adversary_steps"] == 0 and e>0:
            outString += ', F1 basal trained={:.4f}'.format(f1_basal_trained)
            #else:
            #    outString += ', F1 basal trained= %s'%f1_basal_trained
        if (e==0 or (e%250==0 and e>0)):
            print2log(outString)
    print2log(outString)
    encoder_1.eval()
    encoder_2.eval()
    prior_d.eval()
    local_d.eval()
    classifier.eval()
    adverse_classifier.eval()
    Vsp.eval()
    #model.eval()
    #master_encoder.eval()
    
    paired_val_inds = len(valInfo_paired)
    x_1 = torch.tensor(np.concatenate((cmap.loc[valInfo_paired['sig_id.x']].values,
                                          cmap.loc[valInfo_1.sig_id].values))).float().to(device)
    x_2 = torch.tensor(np.concatenate((cmap.loc[valInfo_paired['sig_id.y']].values,
                                          cmap.loc[valInfo_2.sig_id].values))).float().to(device)
    
    z_species_1 = torch.cat((torch.ones(x_1.shape[0],1),
                             torch.zeros(x_1.shape[0],1)),1).to(device)
    z_species_2 = torch.cat((torch.zeros(x_2.shape[0],1),
                             torch.ones(x_2.shape[0],1)),1).to(device)

    #z_latent_1 = encoder_1(x_1)
    #z_latent_2 = encoder_2(x_2)
    
    z_latent_base_1 = encoder_1(x_1)
    z_latent_base_2 = encoder_2(x_2)
    
    z_latent_1 = Vsp(z_latent_base_1,z_species_1)
    z_latent_2 = Vsp(z_latent_base_2,z_species_2)
    
    labels = classifier(torch.cat((z_latent_1, z_latent_2), 0))
    true_labels = torch.cat((torch.ones(z_latent_1.shape[0]).view(z_latent_1.shape[0],1),
                             torch.zeros(z_latent_2.shape[0]).view(z_latent_2.shape[0],1)),0).long()
    _, predicted = torch.max(labels, 1)
    predicted = predicted.cpu().numpy()
    cf_matrix = confusion_matrix(true_labels.numpy(),predicted)
    tn, fp, fn, tp = cf_matrix.ravel()
    class_acc = (tp+tn)/predicted.size
    f1 = 2*tp/(2*tp+fp+fn)
    
    valF1.append(f1)
    valClassAcc.append(class_acc)
    
    print2log('Classification accuracy: %s'%class_acc)
    print2log('Classification F1 score: %s'%f1)

print2log('Summarize validation results')

print2log(np.mean(valF1))
print2log(np.mean(valClassAcc))


df_result = pd.DataFrame({'F1_score':valF1,'ClassAccuracy':valClassAcc})
df_result.to_csv('../results/MI_results/allgenes_5foldvalidation_withCPA_1000ep512bs_a375_ht29_nodecoders.csv')
print2log(df_result)