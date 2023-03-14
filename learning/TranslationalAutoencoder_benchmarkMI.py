#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from trainingUtils import MultipleOptimizer, MultipleScheduler, compute_kernel, compute_mmd
from models import Encoder,Decoder,VAE,CellStateEncoder,CellStateDecoder, CellStateVAE, SimpleEncoder,LocalDiscriminator,PriorDiscriminator,EmbInfomax,MultiEncInfomax
import math
import numpy as np
import pandas as pd
import sys
import random
import os
import logging
from scipy.stats import spearmanr
from sklearn.metrics import silhouette_score,confusion_matrix
from evaluationUtils import r_square,get_cindex,pearson_r,pseudoAccuracy

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
print2log = logger.info
# In[ ]:


device = torch.device('cuda')


# In[ ]:


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
# cmap = pd.read_csv('cmap_2_1.csv',index_col = 0)
cmap = pd.read_csv('../preprocessing/preprocessed_data/cmap_HT29_A375.csv',index_col = 0)

gene_size = len(cmap.columns)
samples = cmap.index.values

# Create a train generators
def getSamples(N, batchSize):
    order = np.random.permutation(N)
    outList = []
    while len(order)>0:
        outList.append(order[0:batchSize])
        order = order[batchSize:]
    return outList


class CellBinaryClassifier(torch.nn.Module):
    def __init__(self,in_channel,hidden_layers,drop_in=0.5,drop=0.2,bn=0.6,bias=True):
        super(CellBinaryClassifier, self).__init__()
        self.drop_in = drop_in
        self.num_hidden_layers = len(hidden_layers)
        self.bias = bias
        self.bn = torch.nn.ModuleList()
        self.linear_layers = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()
        self.linear_layers.append(torch.nn.Linear(in_channel, hidden_layers[0], bias=bias))
        self.bn.append(torch.nn.BatchNorm1d(num_features=hidden_layers[0], momentum=bn))
        self.dropouts.append(torch.nn.Dropout(drop))
        self.activations.append(torch.nn.ReLU())
        for i in range(1, len(hidden_layers)):
            self.linear_layers.append(torch.nn.Linear(hidden_layers[i - 1], hidden_layers[i], 
                                                      bias=bias))
            self.bn.append(torch.nn.BatchNorm1d(num_features=hidden_layers[i], momentum=bn))
            self.dropouts.append(torch.nn.Dropout(drop))
            self.activations.append(torch.nn.ReLU())
        self.out_linear = torch.nn.Linear(hidden_layers[i],2,bias=bias)
        self.softmax = torch.nn.Softmax(dim=1)
        if self.drop_in>0:
            self.InputDrop = torch.nn.Dropout(self.drop_in)
        
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
    
    def forward(self,x):
        if self.drop_in>0:
            x = self.InputDrop(x)
        for i in range(self.num_hidden_layers):
            x = self.linear_layers[i](x)
            x = self.bn[i](x)
            x = self.activations[i](x)
            x = self.dropouts[i](x)
            
        return self.softmax(self.out_linear(x))
    
    def L2Regularization(self, L2):

        weightLoss = 0.
        biasLoss = 0.
        for i in range(self.num_hidden_layers):
            weightLoss = weightLoss + L2 * torch.sum((self.linear_layers[i].weight)**2)
            if self.bias==True:
                biasLoss = biasLoss + L2 * torch.sum((self.linear_layers[i].bias)**2)
        L2Loss = biasLoss + weightLoss
        return(L2Loss)


# In[7]:


NUM_EPOCHS = 1000
#bs = 512
bs_1 = 178
bs_2 = 154
bs_paired = 90
beta=1.0
class_criterion = torch.nn.CrossEntropyLoss()


# In[8]:


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

for i in range(10):
    # Network
    decoder_1 = Decoder(292,[384,640],gene_size,dropRate=0.2, activation=torch.nn.ELU()).to(device)
    decoder_2 = Decoder(292,[384,640],gene_size,dropRate=0.2, activation=torch.nn.ELU()).to(device)
    
    # Infomax
    #master_encoder = SimpleEncoder(gene_size,[640,384],292,dropRate=0.1, activation=torch.nn.ELU())#.to(device)
    encoder_1 = SimpleEncoder(gene_size,[640,384],292,dropRate=0.1, activation=torch.nn.ELU()).to(device)
    encoder_2 = SimpleEncoder(gene_size,[640,384],292,dropRate=0.1, activation=torch.nn.ELU()).to(device)
    prior_d = PriorDiscriminator(292).to(device)
    local_d = LocalDiscriminator(292,292).to(device)
    
    classifier = CellBinaryClassifier(in_channel=292,hidden_layers=[256,128,64],drop_in=0.5,drop=0.25).to(device)
    
    trainInfo_paired = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/train_paired_%s.csv'%i,index_col=0)
    trainInfo_1 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/train_a375_%s.csv'%i,index_col=0)
    trainInfo_2 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/train_ht29_%s.csv'%i,index_col=0)
    
    valInfo_paired = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/val_paired_%s.csv'%i,index_col=0)
    valInfo_1 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/val_a375_%s.csv'%i,index_col=0)
    valInfo_2 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/val_ht29_%s.csv'%i,index_col=0)

    #trainInfo_2 = pd.concat((trainInfo_2,
    #                         valInfo_2,
    #                         valInfo_paired.loc[:,['sig_id.y','cell_iname.y','conditionId']].rename(columns = {'sig_id.y':'sig_id','cell_iname.y':'cell_iname'})),0)
    
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
    optimizer = torch.optim.Adam(allParams, lr= 0.001, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=200,gamma=0.8)
    for e in range(0, NUM_EPOCHS):
        decoder_1.train()
        decoder_2.train()
        encoder_1.train()
        encoder_2.train()
        prior_d.train()
        local_d.train()
        classifier.train()
        #master_encoder.train()
        
        trainloader_1 = getSamples(N_1, bs_1)
        len_1 = len(trainloader_1)
        trainloader_2 = getSamples(N_2, bs_2)
        len_2 = len(trainloader_2)
        trainloader_paired = getSamples(N_paired, bs_paired)
        len_paired = len(trainloader_paired)

        lens = [len_1,len_2,len_paired]
        maxLen = np.max(lens)

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

            z_1 = encoder_1(X_1)
            z_2 = encoder_2(X_2)
            
            z_un = local_d(torch.cat((z_1, z_2), 0))
            res_un = torch.matmul(z_un, z_un.t())
            
            y_pred_1 = decoder_1(z_1)
            fitLoss_1 = torch.mean(torch.sum((y_pred_1 - X_1)**2,dim=1))
            L2Loss_1 = decoder_1.L2Regularization(0.01) + encoder_1.L2Regularization(0.01)
            loss_1 = fitLoss_1 + L2Loss_1
            
            y_pred_2 = decoder_2(z_2)
            fitLoss_2 = torch.mean(torch.sum((y_pred_2 - X_2)**2,dim=1))
            L2Loss_2 = decoder_2.L2Regularization(0.01) + encoder_2.L2Regularization(0.01)
            loss_2 = fitLoss_2 + L2Loss_2

            silimalityLoss = torch.mean(torch.sum((z_1[0:paired_inds,:] - z_2[0:paired_inds,:])**2,dim=-1))
            
            p_samples = res_un * pos_mask.float()
            q_samples = res_un * neg_mask.float()

            Ep = log_2 - F.softplus(- p_samples)
            Eq = F.softplus(-q_samples) + q_samples - log_2

            Ep = (Ep * pos_mask.float()).sum() / pos_mask.float().sum()
            Eq = (Eq * neg_mask.float()).sum() / neg_mask.float().sum()
            mi_loss = Eq - Ep

            prior = torch.rand_like(torch.cat((z_1, z_2), 0))

            term_a = torch.log(prior_d(prior)).mean()
            term_b = torch.log(1.0 - prior_d(torch.cat((z_1, z_2), 0))).mean()
            prior_loss = -(term_a + term_b) * beta

            # Classification loss
            labels = classifier(torch.cat((z_1, z_2), 0))
            true_labels = torch.cat((torch.ones(z_1.shape[0]),
                                     torch.zeros(z_2.shape[0])),0).long().to(device)
            entropy = class_criterion(labels,true_labels)
            
            loss = loss_1 + loss_2 + mi_loss + prior_loss + silimalityLoss + 100*entropy +classifier.L2Regularization(1e-2)

            loss.backward()

            optimizer.step()
        
            pearson_1 = pearson_r(y_pred_1.detach().flatten(), X_1.detach().flatten())
            r2_1 = r_square(y_pred_1.detach().flatten(), X_1.detach().flatten())
            mse_1 = torch.mean(torch.mean((y_pred_1.detach() - X_1.detach())**2,dim=1))
        
            pearson_2 = pearson_r(y_pred_2.detach().flatten(), X_2.detach().flatten())
            r2_2 = r_square(y_pred_2.detach().flatten(), X_2.detach().flatten())
            mse_2 = torch.mean(torch.mean((y_pred_2.detach() - X_2.detach())**2,dim=1))
            
            
        scheduler.step()
        outString = 'Split {:.0f}: Epoch={:.0f}/{:.0f}'.format(i+1,e+1,NUM_EPOCHS)
        outString += ', r2_1={:.4f}'.format(r2_1.item())
        outString += ', pearson_1={:.4f}'.format(pearson_1.item())
        outString += ', MSE_1={:.4f}'.format(mse_1.item())
        outString += ', r2_2={:.4f}'.format(r2_2.item())
        outString += ', pearson_2={:.4f}'.format(pearson_2.item())
        outString += ', MSE_2={:.4f}'.format(mse_2.item())
        outString += ', MI Loss={:.4f}'.format(mi_loss.item())
        outString += ', Prior Loss={:.4f}'.format(prior_loss.item())
        outString += ', Entropy Loss={:.4f}'.format(entropy.item())
        outString += ', loss={:.4f}'.format(loss.item())
        if (e%250==0):
            print2log(outString)
    print2log(outString)
    #trainLoss.append(splitLoss)
    decoder_1.eval()
    decoder_2.eval()
    encoder_1.eval()
    encoder_2.eval()
    prior_d.eval()
    local_d.eval()
    classifier.eval()
    #model.eval()
    #master_encoder.eval()
    
    paired_val_inds = len(valInfo_paired)
    x_1 = torch.tensor(np.concatenate((cmap.loc[valInfo_paired['sig_id.x']].values,
                                          cmap.loc[valInfo_1.sig_id].values))).float().to(device)
    x_2 = torch.tensor(np.concatenate((cmap.loc[valInfo_paired['sig_id.y']].values,
                                          cmap.loc[valInfo_2.sig_id].values))).float().to(device)

    z_latent_1 = encoder_1(x_1)
    z_latent_2 = encoder_2(x_2)
    
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
    #valMSE_1.append(mse_1.item())
    valR2_2.append(r2_2.item())
    valPear_2.append(pearson_2.item())
    #valMSE_2.append(mse_2.item())
    #print2log('R^2 1: %s'%r2_1.item())
    print2log('Pearson correlation 1: %s'%pearson_1.item())
    #print2log('MSE 1: %s'%mse_1.item())
    #print2log('Spearman correlation 1: %s'%valSpear_1[i])
    #print2log('Pseudo-Accuracy 1: %s'%valAccuracy_1[i])
    #print2log('R^2 2: %s'%r2_2.item())
    print2log('Pearson correlation 2: %s'%pearson_2.item())
    #print2log('MSE 2: %s'%mse_2.item())
    #print2log('Spearman correlation 2: %s'%valSpear_2[i])
    #print2log('Pseudo-Accuracy 2: %s'%valAccuracy_2[i])
    
    
    #x_1_equivalent = torch.tensor(cmap_val.loc[mask.index[np.where(mask>0)[0]],:].values).float().to(device)
    #x_2_equivalent = torch.tensor(cmap_val.loc[mask.columns[np.where(mask>0)[1]],:].values).float().to(device)
    x_1_equivalent = x_1[0:paired_val_inds,:]
    x_2_equivalent = x_2[0:paired_val_inds,:]
    
    pearDirect = pearson_r(x_1_equivalent.detach().flatten(), x_2_equivalent.detach().flatten())
    rhos = []
    for jj in range(x_1_equivalent.shape[0]):
        rho,p = spearmanr(x_1_equivalent[jj,:].detach().cpu().numpy(),x_2_equivalent[jj,:].detach().cpu().numpy())
        rhos.append(rho)
    spearDirect = np.mean(rhos)
    accDirect_2 = np.mean(pseudoAccuracy(x_2_equivalent.detach().cpu(),x_1_equivalent.detach().cpu(),eps=1e-6))
    accDirect_1 = np.mean(pseudoAccuracy(x_1_equivalent.detach().cpu(),x_2_equivalent.detach().cpu(),eps=1e-6))

    z_latent_1_equivalent  = encoder_1(x_1_equivalent)
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
    print2log('Pearson of direct translation: %s'%pearDirect.item())
    print2log('Pearson correlation 1 to 2: %s'%pearson_2.item())
    print2log('Pseudo accuracy 1 to 2: %s'%acc_2)

    z_latent_2_equivalent  = encoder_2(x_2_equivalent)
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
    print2log('Pearson correlation 2 to 1: %s'%pearson_1.item())
    print2log('Pseudo accuracy 2 to 1: %s'%acc_1)
    
    
    valPear.append([pearson_2.item(),pearson_1.item()])
    valSpear.append([rho_2,rho_1])
    valAccuracy.append([acc_2,acc_1])
    
    valPearDirect.append(pearDirect.item())
    valSpearDirect.append(spearDirect)
    valAccDirect.append([accDirect_2,accDirect_1])
    
    torch.save(decoder_1,'../results/MI_results/models/HT29_A375_withclass/decoder_a375_%s.pt'%i)
    torch.save(decoder_2,'../results/MI_results/models/HT29_A375_withclass/decoder_ht29_%s.pt'%i)
    torch.save(prior_d,'../results/MI_results/models/HT29_A375_withclass/priorDiscr_%s.pt'%i)
    torch.save(local_d,'../results/MI_results/models/HT29_A375_withclass/localDiscr_%s.pt'%i)
    torch.save(encoder_1,'../results/MI_results/models/HT29_A375_withclass/encoder_a375_%s.pt'%i)
    torch.save(encoder_2,'../results/MI_results/models/HT29_A375_withclass/encoder_ht29_%s.pt'%i)
    torch.save(classifier,'../results/MI_results/models/HT29_A375_withclass/classifier_%s.pt'%i)



valPear = np.array(valPear)
valPearDirect = np.array(valPearDirect)
crossCorrelation = np.array(crossCorrelation)
valSpear = np.array(valSpear)
valAccuracy= np.array(valAccuracy)
valSpearDirect= np.array(valSpearDirect)
valAccDirect= np.array(valAccDirect)



# In[14]:


print2log(np.mean(valPear))
print2log(np.mean(valPearDirect))


# In[18]:


print2log(np.mean(valSpear))
print2log(np.mean(valSpearDirect))


# In[19]:


print2log(np.mean(valAccuracy))
print2log(np.mean(valAccDirect,axis=0))


#df_result = pd.DataFrame({'model_pearsonA375':valPear,
#                          'model_spearA375':valSpear,
#                          'model_accA375':valAccuracy,
#                          'recon_pear_a375':valPear_1,
#                          'recon_spear_a375':valSpear_1,
#                          'recon_acc_a375':valAccuracy_1,
#                          'Direct_pearson':valPearDirect,'Direct_spearman':valSpearDirect,
#                          'DirectAcc_ht29':valAccDirect[:,0],'DirectAcc_a375':valAccDirect[:,1]})
#df_result
df_result = pd.DataFrame({'model_pearsonHT29':valPear[:,0],'model_pearsonA375':valPear[:,1],
                          'model_spearHT29':valSpear[:,0],'model_spearA375':valSpear[:,1],
                          'model_accHT29':valAccuracy[:,0],'model_accA375':valAccuracy[:,1],
                          'recon_pear_ht29':valPear_2 ,'recon_pear_a375':valPear_1,
                          'recon_spear_ht29':valSpear_2 ,'recon_spear_a375':valSpear_1,
                          'recon_acc_ht29':valAccuracy_2 ,'recon_acc_a375':valAccuracy_1,
                          'Direct_pearson':valPearDirect,'Direct_spearman':valSpearDirect,
                          'DirectAcc_ht29':valAccDirect[:,0],'DirectAcc_a375':valAccDirect[:,1]})
df_result.to_csv('MI_results/landmarks_10foldvalidation_notpretrained_MIuniform_and_l2sim_2encs_1000ep512bs_ht29_a375_withclass.csv')
#df_result.to_csv('yolo_yolo.csv')

