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


# In[2]:


device = torch.device('cuda')
print(torch.cuda.is_available())
print(device)


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
lands = pd.read_csv('../preprocessing/preprocessed_data/cmap_landmarks_HT29_A375.csv',index_col = 0)
lands = lands.columns
cmap = pd.read_csv('../preprocessing/preprocessed_data/cmap_all_genes_q1_tas03.csv',index_col = 0)
cmap = cmap.loc[:,lands]


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

# ### Pre-train encoder and then classifier to have apre-trained discriminator

# In[6]:


class_criterion = torch.nn.CrossEntropyLoss()
NUM_EPOCHS= 1000
bs_1 = 120
bs_2 =  120
bs_paired =  80
# ### Train the whole model

# In[8]:


def compute_gradients(output, input):
    grads = torch.autograd.grad(output, input, create_graph=True)
    grads = grads[0].pow(2).mean()
    return grads


# In[9]:


class_criterion = torch.nn.CrossEntropyLoss()
NUM_EPOCHS= model_params['epochs']
bs_1 = model_params['batch_size_1']
bs_2 =  model_params['batch_size_2']
bs_paired =  model_params['batch_size_paired']



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

pretrained_adv_class = torch.load('../preprocessing/preprocessed_data/sampledDatasetes/A375_HT29/pre_trained_classifier_adverse_1_lands.pt')

folders = ['sample_ratio_114','sample_ratio_228','sample_ratio_342',
           'sample_ratio_456','sample_ratio_569','sample_ratio_683']


#folders = ['sample_ratio_0.415823367065317','sample_ratio_0.480220791168353','sample_ratio_',
#           'sample_ratio_','sample_ratio_','sample_ratio_',
#           'sample_ratio_','sample_ratio_','sample_ratio_']
#import os
#folders = [x[0].split('/')[-1] for x in os.walk('../preprocessing/preprocessed_data/sampledDatasetes/A375_HT29/')]
#folders.pop(0)
#folders.sort()

# batchSizes_2 = [10,20,30,40,50,60,70,80,100,120]
# batchSizes_2 = [5,10,15,20,30,50,70,90,120]
# batchSizes_paired = [5,10,15,20,25,30,40,60,90]
# batchSizes_1 = [3,10,15,20,25,35,50,70,95]
# batchSizes_2 = [5,15,20,30,40,60,70,100,120]
batchSizes_paired = [2,9,15,30,50,70]

for fold_id,folder in enumerate(folders):

    # bs_1 = batchSizes_1[fold_id]
    # bs_2 = batchSizes_2[fold_id]
    print('Working with folder:'+folder)
    bs_paired = batchSizes_paired[fold_id]

    valR2 = []
    valPear = []
    valMSE = []
    valSpear = []
    valAccuracy = []

    valPearDirect = []
    valSpearDirect = []
    valAccDirect = []

    valR2_1 = []
    valPear_1 = []
    valMSE_1 = []
    valSpear_1 = []
    valAccuracy_1 = []

    valR2_2 = []
    valPear_2 = []
    valMSE_2 = []
    valSpear_2 = []
    valAccuracy_2 = []

    crossCorrelation = []

    valF1 = []
    valClassAcc = []

    for i in range(1,6):

        # Network
        decoder_1 = Decoder(model_params['latent_dim'],model_params['decoder_1_hiddens'],gene_size,
                        dropRate=model_params['dropout_decoder'], 
                        activation=model_params['decoder_activation']).to(device)
        decoder_2 = Decoder(model_params['latent_dim'],model_params['decoder_2_hiddens'],gene_size,
                        dropRate=model_params['dropout_decoder'], 
                        activation=model_params['decoder_activation']).to(device)
        encoder_1 = SimpleEncoder(gene_size,model_params['encoder_1_hiddens'],model_params['latent_dim'],
                              dropRate=model_params['dropout_encoder'], 
                              activation=model_params['encoder_activation']).to(device)
        encoder_2 = SimpleEncoder(gene_size,model_params['encoder_2_hiddens'],model_params['latent_dim'],
                                  dropRate=model_params['dropout_encoder'], 
                                  activation=model_params['encoder_activation']).to(device)
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

        adverse_classifier.load_state_dict(pretrained_adv_class.state_dict())
    
        Vsp = SpeciesCovariate(2,model_params['latent_dim'],dropRate=model_params['V_dropout']).to(device)
    
        trainInfo_paired = pd.read_csv('../preprocessing/preprocessed_data/sampledDatasetes/pairedPercs/'+folder+'/train_paired_%s.csv'%i,index_col=None)
        trainInfo_1 = pd.read_csv('../preprocessing/preprocessed_data/sampledDatasetes/pairedPercs/'+folder+'/train_A375_%s.csv'%i,index_col=None)
        trainInfo_2 = pd.read_csv('../preprocessing/preprocessed_data/sampledDatasetes/pairedPercs/'+folder+'/train_HT29_%s.csv'%i,index_col=None)

        valInfo_paired = pd.read_csv('../preprocessing/preprocessed_data/sampledDatasetes/pairedPercs/'+folder+'/val_paired_%s.csv'%i,index_col=None)
        valInfo_1 = pd.read_csv('../preprocessing/preprocessed_data/sampledDatasetes/pairedPercs/'+folder+'/val_A375_%s.csv'%i,index_col=None)
        valInfo_2 = pd.read_csv('../preprocessing/preprocessed_data/sampledDatasetes/pairedPercs/'+folder+'/val_HT29_%s.csv'%i,index_col=None)
    
        N_paired = len(trainInfo_paired)
        N_1 = len(trainInfo_1)
        N_2 = len(trainInfo_2)
        N = N_1
        if N_2>N:
            N=N_2
    
        allParams = list(decoder_1.parameters()) +list(decoder_2.parameters())
        allParams = allParams + list(encoder_1.parameters()) +list(encoder_2.parameters())
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
            decoder_1.train()
            decoder_2.train()
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
            
                y_pred_1 = decoder_1(z_1)
                fitLoss_1 = torch.mean(torch.sum((y_pred_1 - X_1)**2,dim=1))
                L2Loss_1 = decoder_1.L2Regularization(model_params['dec_l2_reg']) + encoder_1.L2Regularization(model_params['enc_l2_reg'])
                loss_1 = fitLoss_1 + L2Loss_1
            
                y_pred_2 = decoder_2(z_2)
                fitLoss_2 = torch.mean(torch.sum((y_pred_2 - X_2)**2,dim=1))
                L2Loss_2 = decoder_2.L2Regularization(model_params['dec_l2_reg']) + encoder_2.L2Regularization(model_params['enc_l2_reg'])
                loss_2 = fitLoss_2 + L2Loss_2

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
                loss = loss_1 + loss_2 + model_params['similarity_reg'] * silimalityLoss +model_params['lambda_mi_loss']*mi_loss + prior_loss  + model_params['reg_classifier'] * entropy - model_params['reg_adv']*adv_entropy +classifier.L2Regularization(model_params['state_class_reg']) +Vsp.Regularization(model_params['v_reg'])  - model_params['cosine_loss'] * cosineLoss

                loss.backward()
                optimizer.step()
            
        
                pearson_1 = pearson_r(y_pred_1.detach().flatten(), X_1.detach().flatten())
                r2_1 = r_square(y_pred_1.detach().flatten(), X_1.detach().flatten())
                mse_1 = torch.mean(torch.mean((y_pred_1.detach() - X_1.detach())**2,dim=1))
        
                pearson_2 = pearson_r(y_pred_2.detach().flatten(), X_2.detach().flatten())
                r2_2 = r_square(y_pred_2.detach().flatten(), X_2.detach().flatten())
                mse_2 = torch.mean(torch.mean((y_pred_2.detach() - X_2.detach())**2,dim=1))
            
                #iteration += iteration
            
            
            if model_params['schedule_step_adv'] is not None:
                scheduler_adv.step()
            if (e>=0):
                scheduler.step()
                outString = 'Try {:.0f}: Split {:.0f}: Epoch={:.0f}/{:.0f}'.format(fold_id,i,e+1,NUM_EPOCHS)
                outString += ', r2_1={:.4f}'.format(r2_1.item())
                outString += ', pearson_1={:.4f}'.format(pearson_1.item())
                outString += ', MSE_1={:.4f}'.format(mse_1.item())
                outString += ', r2_2={:.4f}'.format(r2_2.item())
                outString += ', pearson_2={:.4f}'.format(pearson_2.item())
                outString += ', MSE_2={:.4f}'.format(mse_2.item())
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
                print(outString)
        print(outString)
        #trainLoss.append(splitLoss)
        decoder_1.eval()
        decoder_2.eval()
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
    
        print('Classification accuracy: %s'%class_acc)
        print('Classification F1 score: %s'%f1)

        xhat_1 = decoder_1(z_latent_1)
        xhat_2 = decoder_2(z_latent_2)

    
        r2_1 = r_square(xhat_1.detach().flatten(), x_1.detach().flatten())
        pearson_1 = pearson_r(xhat_1.detach().flatten(), x_1.detach().flatten())
        mse_1 = torch.mean(torch.mean((xhat_1 - x_1)**2,dim=1))
        r2_2 = r_square(xhat_2.detach().flatten(), x_2.detach().flatten())
        pearson_2 = pearson_r(xhat_2.detach().flatten(), x_2.detach().flatten())
        mse_2 = torch.mean(torch.mean((xhat_2 - x_2)**2,dim=1))
        rhos = []
        for jj in range(xhat_1.shape[0]):
            rho,p = spearmanr(x_1[jj,:].detach().cpu().numpy(),xhat_1[jj,:].detach().cpu().numpy())
            rhos.append(rho)
        valSpear_1.append(np.mean(rhos))
        acc = pseudoAccuracy(x_1.detach().cpu(),xhat_1.detach().cpu(),eps=1e-6)
        valAccuracy_1.append(np.mean(acc))
        rhos = []
        for jj in range(xhat_2.shape[0]):
            rho,p = spearmanr(x_2[jj,:].detach().cpu().numpy(),xhat_2[jj,:].detach().cpu().numpy())
            rhos.append(rho)
        valSpear_2.append(np.mean(rhos))
        acc = pseudoAccuracy(x_2.detach().cpu(),xhat_2.detach().cpu(),eps=1e-6)
        valAccuracy_2.append(np.mean(acc))
    
        valR2_1.append(r2_1.item())
        valPear_1.append(pearson_1.item())
        valMSE_1.append(mse_1.item())
        valR2_2.append(r2_2.item())
        valPear_2.append(pearson_2.item())
        valMSE_2.append(mse_2.item())
        #print('R^2 1: %s'%r2_1.item())
        print('Pearson correlation 1: %s'%pearson_1.item())
        print('Spearman correlation 1: %s'%valSpear_1[i-1])
        print('Pseudo-Accuracy 1: %s'%valAccuracy_1[i-1])
        #print('R^2 2: %s'%r2_2.item())
        print('Pearson correlation 2: %s'%pearson_2.item())
        print('Spearman correlation 2: %s'%valSpear_2[i-1])
        print('Pseudo-Accuracy 2: %s'%valAccuracy_2[i-1])
    
    
        #x_1_equivalent = torch.tensor(cmap_val.loc[mask.index[np.where(mask>0)[0]],:].values).float().to(device)
        #x_2_equivalent = torch.tensor(cmap_val.loc[mask.columns[np.where(mask>0)[1]],:].values).float().to(device)
        x_1_equivalent = x_1[0:paired_val_inds,:]
        x_2_equivalent = x_2[0:paired_val_inds,:]
    
        z_species_1_equivalent = z_species_1[0:paired_val_inds,:]
        z_species_2_equivalent = z_species_2[0:paired_val_inds,:]
    
        pearDirect = pearson_r(x_1_equivalent.detach().flatten(), x_2_equivalent.detach().flatten())
        rhos = []
        for jj in range(x_1_equivalent.shape[0]):
            rho,p = spearmanr(x_1_equivalent[jj,:].detach().cpu().numpy(),x_2_equivalent[jj,:].detach().cpu().numpy())
            rhos.append(rho)
        spearDirect = np.mean(rhos)
        accDirect_2 = np.mean(pseudoAccuracy(x_2_equivalent.detach().cpu(),x_1_equivalent.detach().cpu(),eps=1e-6))
        accDirect_1 = np.mean(pseudoAccuracy(x_1_equivalent.detach().cpu(),x_2_equivalent.detach().cpu(),eps=1e-6))

        z_latent_base_1_equivalent  = encoder_1(x_1_equivalent)
        z_latent_1_equivalent = Vsp(z_latent_base_1_equivalent,1.-z_species_1_equivalent)
        x_hat_2_equivalent = decoder_2(z_latent_1_equivalent).detach()
        r2_2 = r_square(x_hat_2_equivalent.detach().flatten(), x_2_equivalent.detach().flatten())
        pearson_2 = pearson_r(x_hat_2_equivalent.detach().flatten(), x_2_equivalent.detach().flatten())
        mse_2 = torch.mean(torch.mean((x_hat_2_equivalent - x_2_equivalent)**2,dim=1))
        rhos = []
        for jj in range(x_hat_2_equivalent.shape[0]):
            rho,p = spearmanr(x_2_equivalent[jj,:].detach().cpu().numpy(),x_hat_2_equivalent[jj,:].detach().cpu().numpy())
            rhos.append(rho)
        rho_2 = np.mean(rhos)
        acc_2 = np.mean(pseudoAccuracy(x_2_equivalent.detach().cpu(),x_hat_2_equivalent.detach().cpu(),eps=1e-6))
        print('Pearson of direct translation: %s'%pearDirect.item())
        print('Pearson correlation 1 to 2: %s'%pearson_2.item())
        print('Pseudo accuracy 1 to 2: %s'%acc_2)

        z_latent_base_2_equivalent  = encoder_2(x_2_equivalent)
        z_latent_2_equivalent = Vsp(z_latent_base_2_equivalent,1.-z_species_2_equivalent)
        x_hat_1_equivalent = decoder_1(z_latent_2_equivalent).detach()
        r2_1 = r_square(x_hat_1_equivalent.detach().flatten(), x_1_equivalent.detach().flatten())
        pearson_1 = pearson_r(x_hat_1_equivalent.detach().flatten(), x_1_equivalent.detach().flatten())
        mse_1 = torch.mean(torch.mean((x_hat_1_equivalent - x_1_equivalent)**2,dim=1))
        rhos = []
        for jj in range(x_hat_1_equivalent.shape[0]):
            rho,p = spearmanr(x_1_equivalent[jj,:].detach().cpu().numpy(),x_hat_1_equivalent[jj,:].detach().cpu().numpy())
            rhos.append(rho)
        rho_1 = np.mean(rhos)
        acc_1 = np.mean(pseudoAccuracy(x_1_equivalent.detach().cpu(),x_hat_1_equivalent.detach().cpu(),eps=1e-6))
        print('Pearson correlation 2 to 1: %s'%pearson_1.item())
        print('Pseudo accuracy 2 to 1: %s'%acc_1)
    
    
        valPear.append([pearson_2.item(),pearson_1.item()])
        valSpear.append([rho_2,rho_1])
        valAccuracy.append([acc_2,acc_1])
    
        valPearDirect.append(pearDirect.item())
        valSpearDirect.append(spearDirect)
        valAccDirect.append([accDirect_2,accDirect_1])
    
        #torch.save(decoder_1,'../preprocessing/preprocessed_data/sampledDatasetes/pairedPercs/'+folder+'/decoder_1_%s.pt'%i)
        #torch.save(decoder_2,'../preprocessing/preprocessed_data/sampledDatasetes/pairedPercs/'+folder+'/decoder_2_%s.pt'%i)
        #torch.save(prior_d,'../preprocessing/preprocessed_data/sampledDatasetes/pairedPercs/'+folder+'/priorDiscr_%s.pt'%i)
        #torch.save(local_d,'../preprocessing/preprocessed_data/sampledDatasetes/pairedPercs/'+folder+'/localDiscr_%s.pt'%i)
        #torch.save(encoder_1,'../preprocessing/preprocessed_data/sampledDatasetes/pairedPercs/'+folder+'/encoder_1_%s.pt'%i)
        #torch.save(encoder_2,'../preprocessing/preprocessed_data/sampledDatasetes/pairedPercs/'+folder+'/encoder_2_%s.pt'%i)
        #torch.save(classifier,'../preprocessing/preprocessed_data/sampledDatasetes/pairedPercs/'+folder+'/classifier_%s.pt'%i)
        #torch.save(Vsp,'../preprocessing/preprocessed_data/sampledDatasetes/pairedPercs/'+folder+'/Vspecies_%s.pt'%i)
        #torch.save(adverse_classifier,'../preprocessing/preprocessed_data/sampledDatasetes/pairedPercs/'+folder+'/classifier_adverse_%s.pt'%i)


    print('Summarize validation results folder %s'%fold_id)

    valPear = np.array(valPear)
    valPearDirect = np.array(valPearDirect)
    crossCorrelation = np.array(crossCorrelation)
    valSpear = np.array(valSpear)
    valAccuracy= np.array(valAccuracy)
    valSpearDirect= np.array(valSpearDirect)
    valAccDirect= np.array(valAccDirect)


    # In[18]:


    print(np.mean(valPear,axis=0))
    print(np.mean(valPearDirect))


    # In[19]:


    print(np.mean(valSpear,axis=0))
    print(np.mean(valSpearDirect))


    # In[20]:


    print(np.mean(valAccuracy,axis=0))
    print(np.mean(valAccDirect,axis=0))


    # In[21]:


    print(np.mean(valF1))
    print(np.mean(valClassAcc))


    df_result = pd.DataFrame({'F1_score':valF1,'ClassAccuracy':valClassAcc,
                              'model_pearsonHT29':valPear[:,0],'model_pearsonA375':valPear[:,1],
                              'model_spearHT29':valSpear[:,0],'model_spearA375':valSpear[:,1],
                              'model_accHT29':valAccuracy[:,0],'model_accA375':valAccuracy[:,1],
                              'recon_pear_ht29':valPear_2 ,'recon_pear_a375':valPear_1,
                              'recon_spear_ht29':valSpear_2 ,'recon_spear_a375':valSpear_1,
                              'recon_acc_ht29':valAccuracy_2 ,'recon_acc_a375':valAccuracy_1,
                              'Direct_pearson':valPearDirect,'Direct_spearman':valSpearDirect,
                              'DirectAcc_ht29':valAccDirect[:,0],'DirectAcc_a375':valAccDirect[:,1]})

    df_result.to_csv('../preprocessing/preprocessed_data/sampledDatasetes/pairedPercs/'+folder+'_landmarks.csv')
