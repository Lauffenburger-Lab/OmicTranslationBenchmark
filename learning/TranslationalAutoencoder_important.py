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
#import logging
from captum.attr import IntegratedGradients
from captum.attr import LayerConductance
from captum.attr import NeuronConductance


#logging.basicConfig(level=logging.INFO, format='%(message)s')
#logger = logging.getLogger()
#print2log = logger.info

device = torch.device('cuda')

torch.backends.cudnn.benchmark = True


def seed_everything(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    
# Read data
# cmap = pd.read_csv('../preprocessing/preprocessed_data/cmap_landmarks_2_1.csv',index_col = 0)
cmap = pd.read_csv('../preprocessing/preprocessed_data/cmap_HA1E_PC3.csv',index_col = 0)

gene_size = len(cmap.columns)
samples = cmap.index.values

sampleInfo_1 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/alldata/pc3_unpaired.csv',index_col=0)
sampleInfo_2 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/alldata/ha1e_unpaired.csv',index_col=0)
sampleInfo_paired = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/alldata/paired_pc3_ha1e.csv',index_col=0)


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

# Network
decoder_1 = Decoder(1024,[2048,4096],gene_size,dropRate=0.2, activation=torch.nn.ELU()).to(device)
decoder_2 = Decoder(1024,[2048,4096],gene_size,dropRate=0.2, activation=torch.nn.ELU()).to(device)
    
# Infomax
#master_encoder = SimpleEncoder(gene_size,[640,384],292,dropRate=0.1, activation=torch.nn.ELU())#.to(device)
encoder_1 = SimpleEncoder(gene_size,[4096,2048],1024,dropRate=0.1, activation=torch.nn.ELU()).to(device)
encoder_2 = SimpleEncoder(gene_size,[4096,2048],1024,dropRate=0.1, activation=torch.nn.ELU()).to(device)
prior_d = PriorDiscriminator(1024).to(device)
local_d = LocalDiscriminator(1024,1024).to(device)
classifier = CellBinaryClassifier(in_channel=1024,hidden_layers=[512,256,128],drop_in=0.5,drop=0.25).to(device)


# In[10]:


# Create a train generators
def getSamples(N, batchSize):
    order = np.random.permutation(N)
    outList = []
    while len(order)>0:
        outList.append(order[0:batchSize])
        order = order[batchSize:]
    return outList


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

decoder_1 = Decoder(model_params['latent_dim'], model_params['decoder_1_hiddens'], gene_size,
                    dropRate=model_params['dropout_decoder'],
                    activation=model_params['decoder_activation']).to(device)
decoder_2 = Decoder(model_params['latent_dim'], model_params['decoder_2_hiddens'], gene_size,
                    dropRate=model_params['dropout_decoder'],
                    activation=model_params['decoder_activation']).to(device)
encoder_1 = SimpleEncoder(gene_size, model_params['encoder_1_hiddens'], model_params['latent_dim'],
                          dropRate=model_params['dropout_encoder'],
                          activation=model_params['encoder_activation']).to(device)
encoder_2 = SimpleEncoder(gene_size, model_params['encoder_2_hiddens'], model_params['latent_dim'],
                          dropRate=model_params['dropout_encoder'],
                          activation=model_params['encoder_activation']).to(device)
prior_d = PriorDiscriminator(model_params['latent_dim']).to(device)
local_d = LocalDiscriminator(model_params['latent_dim'], model_params['latent_dim']).to(device)

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
pretrained_adv_class = torch.load('MI_results/models/CPA_approach/pre_trained_classifier_adverse_1.pt')
adverse_classifier.load_state_dict(pretrained_adv_class.state_dict())

Vsp = SpeciesCovariate(2, model_params['latent_dim'], dropRate=model_params['V_dropout']).to(device)

trainInfo_paired = pd.read_csv('sampledDatasetes/A375_2/' + folder + '/train_paired_%s.csv' % i, index_col=0)
trainInfo_1 = pd.read_csv('sampledDatasetes/A375_2/' + folder + '/train_1_%s.csv' % i, index_col=0)
trainInfo_2 = pd.read_csv('sampledDatasetes/A375_2/' + folder + '/train_2_%s.csv' % i, index_col=0)

valInfo_paired = pd.read_csv('sampledDatasetes/A375_2/' + folder + '/val_paired_%s.csv' % i, index_col=0)
valInfo_1 = pd.read_csv('sampledDatasetes/A375_2/' + folder + '/val_1_%s.csv' % i, index_col=0)
valInfo_2 = pd.read_csv('sampledDatasetes/A375_2/' + folder + '/val_2_%s.csv' % i, index_col=0)

N_paired = len(trainInfo_paired)
N_1 = len(trainInfo_1)
N_2 = len(trainInfo_2)
N = N_1
if N_2 > N:
    N = N_2

allParams = list(decoder_1.parameters()) + list(decoder_2.parameters())
allParams = allParams + list(encoder_1.parameters()) + list(encoder_2.parameters())
allParams = allParams + list(prior_d.parameters()) + list(local_d.parameters())
allParams = allParams + list(classifier.parameters())
allParams = allParams + list(Vsp.parameters())
optimizer = torch.optim.Adam(allParams, lr=model_params['encoding_lr'], weight_decay=0)
optimizer_adv = torch.optim.Adam(adverse_classifier.parameters(), lr=model_params['adv_lr'], weight_decay=0)
if model_params['schedule_step_adv'] is not None:
    scheduler_adv = torch.optim.lr_scheduler.StepLR(optimizer_adv,
                                                    step_size=model_params['schedule_step_adv'],
                                                    gamma=model_params['gamma_adv'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size=model_params['schedule_step_enc'],
                                            gamma=model_params['gamma_enc'])

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
    # master_encoder.train()

    trainloader_1 = getSamples(N_1, bs_1)
    len_1 = len(trainloader_1)
    trainloader_2 = getSamples(N_2, bs_2)
    len_2 = len(trainloader_2)
    trainloader_paired = getSamples(N_paired, bs_paired)
    len_paired = len(trainloader_paired)

    lens = [len_1, len_2, len_paired]
    maxLen = np.max(lens)

    iteration = 1

    if maxLen > lens[0]:
        trainloader_suppl = getSamples(N_1, bs_1)
        for jj in range(maxLen - lens[0]):
            trainloader_1.insert(jj, trainloader_suppl[jj])

    if maxLen > lens[1]:
        trainloader_suppl = getSamples(N_2, bs_2)
        for jj in range(maxLen - lens[1]):
            trainloader_2.insert(jj, trainloader_suppl[jj])

    if maxLen > lens[2]:
        trainloader_suppl = getSamples(N_paired, bs_paired)
        for jj in range(maxLen - lens[2]):
            trainloader_paired.insert(jj, trainloader_suppl[jj])
    # for dataIndex in trainloader:
    for j in range(maxLen):
        dataIndex_1 = trainloader_1[j]
        dataIndex_2 = trainloader_2[j]
        dataIndex_paired = trainloader_paired[j]

        df_pairs = trainInfo_paired.iloc[dataIndex_paired, :]
        df_1 = trainInfo_1.iloc[dataIndex_1, :]
        df_2 = trainInfo_2.iloc[dataIndex_2, :]
        paired_inds = len(df_pairs)

        X_1 = torch.tensor(np.concatenate((cmap.loc[df_pairs['sig_id.x']].values,
                                           cmap.loc[df_1.sig_id].values))).float().to(device)
        X_2 = torch.tensor(np.concatenate((cmap.loc[df_pairs['sig_id.y']].values,
                                           cmap.loc[df_2.sig_id].values))).float().to(device)

        z_species_1 = torch.cat((torch.ones(X_1.shape[0], 1),
                                 torch.zeros(X_1.shape[0], 1)), 1).to(device)
        z_species_2 = torch.cat((torch.zeros(X_2.shape[0], 1),
                                 torch.ones(X_2.shape[0], 1)), 1).to(device)

        conditions = np.concatenate((df_pairs.conditionId.values,
                                     df_1.conditionId.values,
                                     df_pairs.conditionId.values,
                                     df_2.conditionId.values))
        size = conditions.size
        conditions = conditions.reshape(size, 1)
        conditions = conditions == conditions.transpose()
        conditions = conditions * 1
        mask = torch.tensor(conditions).to(device).detach()
        pos_mask = mask
        neg_mask = 1 - mask
        log_2 = math.log(2.)
        optimizer.zero_grad()
        optimizer_adv.zero_grad()

        # if e % model_params['adversary_steps']==0:
        z_base_1 = encoder_1(X_1)
        z_base_2 = encoder_2(X_2)
        latent_base_vectors = torch.cat((z_base_1, z_base_2), 0)
        labels_adv = adverse_classifier(latent_base_vectors)
        true_labels = torch.cat((torch.ones(z_base_1.shape[0]),
                                 torch.zeros(z_base_2.shape[0])), 0).long().to(device)
        _, predicted = torch.max(labels_adv, 1)
        predicted = predicted.cpu().numpy()
        cf_matrix = confusion_matrix(true_labels.cpu().numpy(), predicted)
        tn, fp, fn, tp = cf_matrix.ravel()
        f1_basal_trained = 2 * tp / (2 * tp + fp + fn)
        adv_entropy = class_criterion(labels_adv, true_labels)
        adversary_drugs_penalty = compute_gradients(labels_adv.sum(), latent_base_vectors)
        loss_adv = adv_entropy + model_params['adv_penalnty'] * adversary_drugs_penalty
        loss_adv.backward()
        optimizer_adv.step()
        # print(f1_basal_trained)
        # else:
        optimizer.zero_grad()
        # f1_basal_trained = None
        z_base_1 = encoder_1(X_1)
        z_base_2 = encoder_2(X_2)
        latent_base_vectors = torch.cat((z_base_1, z_base_2), 0)

        # z_un = local_d(torch.cat((z_1, z_2), 0))
        z_un = local_d(latent_base_vectors)
        res_un = torch.matmul(z_un, z_un.t())

        z_1 = Vsp(z_base_1, z_species_1)
        z_2 = Vsp(z_base_2, z_species_2)
        latent_vectors = torch.cat((z_1, z_2), 0)

        y_pred_1 = decoder_1(z_1)
        fitLoss_1 = torch.mean(torch.sum((y_pred_1 - X_1) ** 2, dim=1))
        L2Loss_1 = decoder_1.L2Regularization(model_params['dec_l2_reg']) + encoder_1.L2Regularization(
            model_params['enc_l2_reg'])
        loss_1 = fitLoss_1 + L2Loss_1

        y_pred_2 = decoder_2(z_2)
        fitLoss_2 = torch.mean(torch.sum((y_pred_2 - X_2) ** 2, dim=1))
        L2Loss_2 = decoder_2.L2Regularization(model_params['dec_l2_reg']) + encoder_2.L2Regularization(
            model_params['enc_l2_reg'])
        loss_2 = fitLoss_2 + L2Loss_2

        # silimalityLoss = np.sqrt(paired_inds)*torch.mean(torch.sum((z_base_1[0:paired_inds,:] - z_base_2[0:paired_inds,:])**2,
        #                                      dim=-1))/torch.std(torch.sum((z_base_1[0:paired_inds,:] - z_base_2[0:paired_inds,:])**2,
        #                                                                   dim=-1))
        silimalityLoss = torch.mean(torch.sum((z_base_1[0:paired_inds, :] - z_base_2[0:paired_inds, :]) ** 2, dim=-1))
        cosineLoss = torch.nn.functional.cosine_similarity(z_base_1[0:paired_inds, :], z_base_2[0:paired_inds, :],
                                                           dim=-1).mean()

        p_samples = res_un * pos_mask.float()
        q_samples = res_un * neg_mask.float()

        Ep = log_2 - F.softplus(- p_samples)
        Eq = F.softplus(-q_samples) + q_samples - log_2

        Ep = (Ep * pos_mask.float()).sum() / pos_mask.float().sum()
        Eq = (Eq * neg_mask.float()).sum() / neg_mask.float().sum()
        mi_loss = Eq - Ep

        # prior = torch.rand_like(torch.cat((z_1, z_2), 0))
        prior = torch.rand_like(latent_base_vectors)

        term_a = torch.log(prior_d(prior)).mean()
        # term_b = torch.log(1.0 - prior_d(torch.cat((z_1, z_2), 0))).mean()
        term_b = torch.log(1.0 - prior_d(latent_base_vectors)).mean()
        prior_loss = -(term_a + term_b) * model_params['prior_beta']

        # Classification loss
        labels = classifier(latent_vectors)
        true_labels = torch.cat((torch.ones(z_1.shape[0]),
                                 torch.zeros(z_2.shape[0])), 0).long().to(device)
        entropy = class_criterion(labels, true_labels)
        _, predicted = torch.max(labels, 1)
        predicted = predicted.cpu().numpy()
        cf_matrix = confusion_matrix(true_labels.cpu().numpy(), predicted)
        tn, fp, fn, tp = cf_matrix.ravel()
        f1_latent = 2 * tp / (2 * tp + fp + fn)

        # Remove signal from z_basal
        labels_adv = adverse_classifier(latent_base_vectors)
        true_labels = torch.cat((torch.ones(z_base_1.shape[0]),
                                 torch.zeros(z_base_2.shape[0])), 0).long().to(device)
        adv_entropy = class_criterion(labels_adv, true_labels)
        _, predicted = torch.max(labels_adv, 1)
        predicted = predicted.cpu().numpy()
        cf_matrix = confusion_matrix(true_labels.cpu().numpy(), predicted)
        tn, fp, fn, tp = cf_matrix.ravel()
        f1_basal = 2 * tp / (2 * tp + fp + fn)

        # loss = loss_1 + loss_2 + 100*mi_loss + prior_loss + 10*silimalityLoss-100*cosineLoss + 1000*entropy +\
        #       classifier.L2Regularization(1e-2) + Vsp.Regularization() - 1000*adv_entropy
        loss = loss_1 + loss_2 + model_params['similarity_reg'] * silimalityLoss + model_params[
            'lambda_mi_loss'] * mi_loss + prior_loss + model_params['reg_classifier'] * entropy - model_params[
                   'reg_adv'] * adv_entropy + classifier.L2Regularization(
            model_params['state_class_reg']) + Vsp.Regularization(model_params['v_reg']) - model_params[
                   'cosine_loss'] * cosineLoss

        loss.backward()
        optimizer.step()

        pearson_1 = pearson_r(y_pred_1.detach().flatten(), X_1.detach().flatten())
        r2_1 = r_square(y_pred_1.detach().flatten(), X_1.detach().flatten())
        mse_1 = torch.mean(torch.mean((y_pred_1.detach() - X_1.detach()) ** 2, dim=1))

        pearson_2 = pearson_r(y_pred_2.detach().flatten(), X_2.detach().flatten())
        r2_2 = r_square(y_pred_2.detach().flatten(), X_2.detach().flatten())
        mse_2 = torch.mean(torch.mean((y_pred_2.detach() - X_2.detach()) ** 2, dim=1))

        # iteration += iteration

    if model_params['schedule_step_adv'] is not None:
        scheduler_adv.step()
    if (e >= 0):
        scheduler.step()
        outString = 'Try {:.0f}: Split {:.0f}: Epoch={:.0f}/{:.0f}'.format(fold_id, i, e + 1, NUM_EPOCHS)
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
        # if e % model_params["adversary_steps"] == 0 and e>0:
        outString += ', F1 basal trained={:.4f}'.format(f1_basal_trained)
        # else:
        #    outString += ', F1 basal trained= %s'%f1_basal_trained
    if (e == 0 or (e % 250 == 0 and e > 0)):
        print2log(outString)
print2log(outString)

decoder_1.eval()
decoder_2.eval()
encoder_1.eval()
encoder_2.eval()
prior_d.eval()
local_d.eval()
classifier.eval()
adverse_classifier.eval()
Vsp.eval()

print('Evaluate mode')


# In[19]:


x_1 = torch.tensor(np.concatenate((cmap.loc[sampleInfo_paired['sig_id.x']].values,
                                      cmap.loc[sampleInfo_1.sig_id].values))).float().to(device)
x_2 = torch.tensor(np.concatenate((cmap.loc[sampleInfo_paired['sig_id.y']].values,
                                      cmap.loc[sampleInfo_2.sig_id].values))).float().to(device)

z_species_1 = torch.cat((torch.ones(x_1.shape[0], 1),
                         torch.zeros(x_1.shape[0], 1)), 1).to(device)
z_species_2 = torch.cat((torch.zeros(x_2.shape[0], 1),
                         torch.ones(x_2.shape[0], 1)), 1).to(device)

z_latent_1  = encoder_1(x_1)
z_latent_1 = Vsp(z_latent_1, z_species_1)
xhat_1 = decoder_1(z_latent_1)
z_latent_2  = encoder_2(x_2)
z_latent_2 = Vsp(z_latent_2, z_species_2)
xhat_2 = decoder_2(z_latent_2)


# In[26]:


labels = classifier(torch.cat((z_latent_1, z_latent_2), 0))
true_labels = torch.cat((torch.ones(z_latent_1.shape[0]).view(z_latent_1.shape[0],1),
                         torch.zeros(z_latent_2.shape[0]).view(z_latent_2.shape[0],1)),0).long()
_, predicted = torch.max(labels, 1)
predicted = predicted.cpu().numpy()
cf_matrix = confusion_matrix(true_labels.numpy(),predicted)
tn, fp, fn, tp = cf_matrix.ravel()
class_acc = (tp+tn)/predicted.size
f1 = 2*tp/(2*tp+fp+fn)
print('F1 score:%s'%f1)
print('Accuracy:%s'%class_acc)


# In[27]:


### Save embeddings ###
Embs_1 = pd.DataFrame(z_latent_1.detach().cpu().numpy())
Embs_1.index = np.concatenate((sampleInfo_paired['sig_id.x'].values,sampleInfo_1.sig_id.values))
Embs_1.columns = ['z'+str(i) for i in range(model_params['latent_dim'])]
Embs_1.to_csv('../results/trained_embs_all/AllEmbs_CPA_pc3.csv')
display(Embs_1)


# In[28]:


### Save embeddings ###
Embs_2 = pd.DataFrame(z_latent_2.detach().cpu().numpy())
Embs_2.index = np.concatenate((sampleInfo_paired['sig_id.y'].values,sampleInfo_2.sig_id.values))
Embs_2.columns = ['z'+str(i) for i in range(model_params['latent_dim'])]
Embs_2.to_csv('../results/trained_embs_all/AllEmbs_CPA_ha1e.csv')
display(Embs_2)


# In[29]:


r2_1 = r_square(xhat_1.detach().flatten(), x_1.detach().flatten())
pearson_1 = pearson_r(xhat_1.detach().flatten(), x_1.detach().flatten())
mse_1 = torch.mean(torch.mean((xhat_1 - x_1)**2,dim=1))
rhos =[]
for jj in range(xhat_1.shape[0]):
    rho,p = spearmanr(x_1[jj,:].detach().cpu().numpy(),xhat_1[jj,:].detach().cpu().numpy())
    rhos.append(rho)
Spear_1=np.mean(rhos)
acc = pseudoAccuracy(x_1.detach().cpu(),xhat_1.detach().cpu(),eps=1e-6)
Accuracy_1=np.mean(acc)
print('R^2 cell1: %s'%r2_1.item())
print('Pearson correlation cell1: %s'%pearson_1.item())
print('MSE cell1: %s'%mse_1.item())
print('Spearman correlation cell1: %s'%Spear_1)
print('Pseudo-accuracy cell1: %s'%Accuracy_1)


# In[30]:


r2_2 = r_square(xhat_2.detach().flatten(), x_2.detach().flatten())
pearson_2 = pearson_r(xhat_2.detach().flatten(), x_2.detach().flatten())
mse_2 = torch.mean(torch.mean((xhat_2 - x_2)**2,dim=1))
rhos =[]
for jj in range(xhat_2.shape[0]):
    rho,p = spearmanr(x_2[jj,:].detach().cpu().numpy(),xhat_2[jj,:].detach().cpu().numpy())
    rhos.append(rho)
Spear_2=np.mean(rhos)
acc = pseudoAccuracy(x_2.detach().cpu(),xhat_2.detach().cpu(),eps=1e-6)
Accuracy_2=np.mean(acc)
print('R^2 cell2: %s'%r2_2.item())
print('Pearson correlation cell2: %s'%pearson_2.item())
print('MSE cell2: %s'%mse_2.item())
print('Spearman correlation cell2: %s'%Spear_2)
print('Pseudo-accuracy cell2: %s'%Accuracy_2)


# In[31]:


paired_inds = len(sampleInfo_paired)


# In[32]:


x_1_equivalent = x_1[0:paired_inds,:]
x_2_equivalent = x_2[0:paired_inds,:]
pearDirect = pearson_r(x_1_equivalent.detach().flatten(), x_2_equivalent.detach().flatten())
rhos =[]
for jj in range(x_1_equivalent.shape[0]):
    rho,p = spearmanr(x_1_equivalent[jj,:].detach().cpu().numpy(),x_2_equivalent[jj,:].detach().cpu().numpy())
    rhos.append(rho)
Spear_direct=np.mean(rhos)
acc_2 = np.mean(pseudoAccuracy(x_2_equivalent.detach().cpu(),x_1_equivalent.detach().cpu(),eps=1e-6))
acc_1 = np.mean(pseudoAccuracy(x_1_equivalent.detach().cpu(),x_2_equivalent.detach().cpu(),eps=1e-6))
print('Pearson of direct translation: %s'%pearDirect.item())
print('Spearman of direct translation: %s'%Spear_direct)
print('Pdeudo-accuracy of direct translation from cell1 to cell2: %s'%acc_2)
print('Pdeudo-accuracy of direct translation from cell2 to cell1: %s'%acc_1)


# In[33]:


z_latent_1_equivalent  = encoder_1(x_1_equivalent)
z_latent_1_equivalent = Vsp(z_latent_1_equivalent, z_species_2)
x_hat_2_equivalent = decoder_2(z_latent_1_equivalent).detach()
r2_2 = r_square(x_hat_2_equivalent.detach().flatten(), x_2_equivalent.detach().flatten())
pearson_2 = pearson_r(x_hat_2_equivalent.detach().flatten(), x_2_equivalent.detach().flatten())
mse_2 = torch.mean(torch.mean((x_hat_2_equivalent - x_2_equivalent)**2,dim=1))
rhos =[]
for jj in range(x_2_equivalent.shape[0]):
    rho,p = spearmanr(x_2_equivalent[jj,:].detach().cpu().numpy(),x_hat_2_equivalent[jj,:].detach().cpu().numpy())
    rhos.append(rho)
Spear_2=np.mean(rhos)
acc_2 = np.mean(pseudoAccuracy(x_2_equivalent.detach().cpu(),x_hat_2_equivalent.detach().cpu(),eps=1e-6))
print('Pearson correlation cell1 to cell2: %s'%pearson_2.item())
print('Spearman cell1 to cell2: %s'%Spear_2)
print('Pdeudo-accuracy cell1 to cell2: %s'%acc_2)


# In[34]:


z_latent_2_equivalent  = encoder_2(x_2_equivalent)
z_latent_2_equivalent = Vsp(z_latent_2_equivalent, z_species_1)
x_hat_1_equivalent = decoder_1(z_latent_2_equivalent).detach()
r2_1 = r_square(x_hat_1_equivalent.detach().flatten(), x_1_equivalent.detach().flatten())
pearson_1 = pearson_r(x_hat_1_equivalent.detach().flatten(), x_1_equivalent.detach().flatten())
mse_1 = torch.mean(torch.mean((x_hat_1_equivalent - x_1_equivalent)**2,dim=1))
rhos =[]
for jj in range(x_1_equivalent.shape[0]):
    rho,p = spearmanr(x_1_equivalent[jj,:].detach().cpu().numpy(),x_hat_1_equivalent[jj,:].detach().cpu().numpy())
    rhos.append(rho)
Spear_1=np.mean(rhos)
acc_1 = np.mean(pseudoAccuracy(x_1_equivalent.detach().cpu(),x_hat_1_equivalent.detach().cpu(),eps=1e-6))
print('Pearson correlation cell2 to cell1: %s'%pearson_1.item())
print('MSE cell2 to cell1: %s'%mse_1.item())
print('Spearman cell2 to cell1: %s'%Spear_1)
print('Pdeudo-accuracy cell2 to cell1: %s'%acc_1)


# In[ ]:


# plt.figure()
# plt.scatter(x_1_equivalent.detach().flatten().cpu().numpy(),
#             x_2_equivalent.detach().flatten().cpu().numpy(),
#             alpha=0.1)
# plt.text(0, 0.9, 'r {:.2f}'.format(pearDirect.item()))
# plt.savefig('../figures/pc3_to_ha1e_direct_correlation_allgenes.png',dpi=600)


# In[36]:


plt.figure()
plt.scatter(x_hat_1_equivalent.detach().flatten().cpu().numpy(),
            x_1_equivalent.detach().flatten().cpu().numpy(),
           alpha=0.1)
plt.text(0, 0.9, 'r {:.2f}'.format(pearson_1.item()))
plt.savefig('../figures/ha1e_to_pc3_translate_correlation_cpa.png',dpi=600)


# In[37]:


plt.figure()
plt.scatter(x_hat_2_equivalent.detach().flatten().cpu().numpy(),
            x_2_equivalent.detach().flatten().cpu().numpy(),
           alpha=0.1)
plt.text(0, 0.9, 'r {:.2f}'.format(pearson_2.item()))
plt.savefig('../figures/pc3_to_ha1e_translate_correlation_cpa.png',dpi=600)


# In[39]:


torch.save(encoder_1, '../results/trained_models/Encoder_PC3_MI_allgenes_CPA.pth')
torch.save(encoder_2, '../results/trained_models/Encoder_HA1E_MI_allgenes_CPA.pth')
torch.save(decoder_1, '../results/trained_models/Decoder_PC3_MI_allgenes_CPA.pth')
torch.save(decoder_2, '../results/trained_models/Decoder_HA1E_MI_allgenes_CPA.pth')
torch.save(local_d, '../results/trained_models/local_d_MI_pc3_ha1e_allgenes_CPA.pth')
torch.save(prior_d, '../results/trained_models/prior_d_MI_pc3_ha1e_allgenes_CPA.pth')
torch.save(classifier,'../results/trained_models/classifier_pc3_ha1e_CPA.pth')
torch.save(adverse_classifier,'../results/trained_models/adverse_classifier_pc3_ha1e_CPA.pth')
torch.save(Vsp,'../results/trained_models/Vsp_pc3_ha1e_CPA.pth')

import seaborn as sns
sns.set()


# In[49]:


#plt.hist(z_latent_1.detach().cpu().flatten().numpy())
plt.figure(figsize=(10,5), dpi= 80)
sns.set(font_scale=2)
sns.distplot(z_latent_1.detach().cpu().flatten().numpy())
plt.xlabel('PC3 encoded features values')
plt.ylabel('Density')
plt.title('Distribution of latent space features')
plt.savefig('../figures/good_trained_ae_pc3_latent_space_allgenes_CPA.png', bbox_inches='tight',dpi=600)


#plt.hist(z_latent_2.detach().cpu().flatten().numpy())
plt.figure(figsize=(10,5), dpi= 80)
sns.set(font_scale=2)
sns.distplot(z_latent_2.detach().cpu().flatten().numpy())
plt.xlabel('HA1E encoded features values')
plt.ylabel('Density')
plt.title('Distribution of latent space features')
plt.savefig('../figures/good_trained_ae_ha1e_latent_space_allgenes_CPA.png', bbox_inches='tight',dpi=600)


encoder_1=torch.load('../results/trained_models/Encoder_PC3_MI_allgenes_CPA.pth')
# encoder_2=torch.load('../results/trained_models/Encoder_HA1E_MI_allgenes_CPA.pth')
classifier = torch.load('../results/trained_models/classifier_pc3_ha1e_CPA.pth')
Vsp = torch.load('../results/trained_models/Vsp_pc3_ha1e_CPA.pth')

encoder_1.eval()
# encoder_2.eval()
classifier.eval()
Vsp.eval()
print('Evaluation mode')


x_1 = torch.tensor(np.concatenate((cmap.loc[sampleInfo_paired['sig_id.x']].values,
                                      cmap.loc[sampleInfo_1.sig_id].values))).float().to(device)
x_2 = torch.tensor(np.concatenate((cmap.loc[sampleInfo_paired['sig_id.y']].values,
                                      cmap.loc[sampleInfo_2.sig_id].values))).float().to(device)

z_species_1 = torch.cat((torch.ones(x_1.shape[0], 1),
                         torch.zeros(x_1.shape[0], 1)), 1).to(device)
z_species_2 = torch.cat((torch.zeros(x_2.shape[0], 1),
                         torch.ones(x_2.shape[0], 1)), 1).to(device)

paired_inds = len(sampleInfo_paired)
len_samples = x_1.shape[0]


# In[ ]:


# Classifier importance
z_1 = torch.tensor(pd.read_csv('../results/trained_embs_all/AllEmbs_MI_pc3_CPA.csv'index_col=0).drop_duplicates().values).to(device)
z_2 = torch.tensor(pd.read_csv('../results/trained_embs_all/AllEmbs_MI_ha1e_CPA.csv',index_col=0).drop_duplicates().values).to(device)
z = torch.cat((z_1,z_2),0).float()
ig = IntegratedGradients(classifier)
z.requires_grad_()
attr1 = ig.attribute(z,target=1,n_steps=100, return_convergence_delta=False)
attr1 = attr1.detach().cpu().numpy()
attr2 = ig.attribute(z,target=0,n_steps=100, return_convergence_delta=False)
attr2 = attr2.detach().cpu().numpy()
df1 = pd.DataFrame(attr1)
df1.index = cmap.index
df1.columns = ['z'+str(i) for i in range(model_params['latent_dim'])]
df1.to_csv('../results/Importance_results/important_scores_to_classify_as_pc3_cpa.csv')
df2 = pd.DataFrame(attr2)
df2.index = cmap.index
df2.columns = ['z'+str(i) for i in range(model_params['latent_dim'])]
df2.to_csv('../results/Importance_results/important_scores_to_classify_as_ha1e_cpa.csv')


# x_1_equivalent = x_1[0:paired_inds,:]
# x_2_equivalent = x_2[0:paired_inds,:]

class LatentEncode(torch.nn.Module):
    def __init__(self, encoder,V):
        super(LatentEncode, self).__init__()

        self.encoder = encoder
        self.V = V

    def forward(self, x,z_base):

        z = self.encode(x)
        z = self.V(z, z_base)

        return z

fullEncoder = LatentEncode(encoder_1,Vsp).to(device)
fullEncoder.eval()
ig = IntegratedGradients(fullEncoder)
# ig = IntegratedGradients(encoder_1)

# Per output latent variable input importance translation captum
# 1st dimesion input
# 2nd dimesion output
hid_dim = model_params['latent_dim']
scores = torch.zeros ((gene_size,hid_dim)).to(device)
for z in range(hid_dim):
    #encoder_1.zero_grad()
    attr, delta = ig.attribute(x_1,target=z,n_steps=100,return_convergence_delta=True)
    scores[:,z] = torch.mean(attr,0)
print(scores)

plt.figure(figsize=(10,5), dpi= 80)
sns.set(font_scale=2)
sns.distplot(scores.flatten().cpu().numpy())
plt.xlabel('Importance score')
plt.ylabel('Density')
plt.title('Distribution of importance scores of the input features')
plt.savefig('../figures/pc3_important_scores_to_encode.png', bbox_inches='tight')


df = pd.DataFrame(scores.cpu().numpy())
df.columns = ['z'+str(i) for i in range(model_params['latent_dim'])]
df.index = cmap.columns
df.to_csv('../results/Importance_results/important_scores_pc3_to_encode.csv')