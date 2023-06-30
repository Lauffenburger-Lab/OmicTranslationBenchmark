import pickle
import torch
import torch.nn.functional as F
from trainingUtils import NBLoss,_convert_mean_disp_to_counts_logits #,MultipleOptimizer, MultipleScheduler, compute_kernel, compute_mmd
from models import Encoder,Decoder,GaussianDecoder,VAE,CellStateEncoder,\
                   CellStateDecoder, CellStateVAE,\
                   SimpleEncoder,LocalDiscriminator,PriorDiscriminator,\
                   EmbInfomax,MultiEncInfomax,Classifier,\
                   SpeciesCovariate,GaussianDecoder,VarDecoder,ElementWiseLinear

# import argparse
import math
import numpy as np
import pandas as pd
#from IPython.display import clear_output
#from matplotlib import pyplot as plt
#from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split,cross_validate,KFold
from sklearn.metrics import confusion_matrix,f1_score,r2_score
from sklearn.neighbors import KNeighborsClassifier
from torch.distributions import NegativeBinomial
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


# Create a train generators
def getSamples(N, batchSize):
    order = np.random.permutation(N)
    outList = []
    while len(order)>0:
        outList.append(order[0:batchSize])
        order = order[batchSize:]
    return outList


# ### Train model

model_params = {'encoder_1_hiddens':[4096,2048,1024,512],
                'encoder_2_hiddens':[4096,2048,1024,512],
                'latent_dim': 512, # itan 512
                'decoder_1_hiddens':[512,768,2048,4096],
                'decoder_2_hiddens':[512,768,2048,4096],
                'final_dec_1':6144,
                'final_dec_2':6144,
                'dropout_decoder':0.1,
                'dropout_encoder':0.1,
                'encoder_activation':torch.nn.ELU(),
                'decoder_activation':torch.nn.ELU(),
                'V_dropout':0.25,
                'state_class_hidden':[256,128,64,32],#oxi to 64 kai to 1024 arxika
                'state_class_drop_in':0.2,
                'state_class_drop':0.1,
                'no_states':2,
                'adv_class_hidden':[512,256,128,64],
                'adv_class_drop_in':0.25,
                'adv_class_drop':0.1,
                'no_adv_class':2,
                'cell_class_hidden':[256,128,64,32],
                'cell_class_drop_in': 0.2,
                'cell_class_drop': 0.1,
                'no_cell_types':5,
                'encoding_lr':0.001,
                'adv_lr':0.001,
                'schedule_step_adv':40,
                'gamma_adv':0.5,
                'schedule_step_enc':40,
                'gamma_enc':0.8,
                'batch_size':1024,#itan 1024
                'epochs':200, #itan 100
                'prior_beta':1.0,
                'no_folds':10,
                'v_reg':1e-06,
                'state_class_reg':1e-07,
                'enc_l2_reg': 1e-07,
                'dec_l2_reg': 1e-07,
                'lambda_mi_loss':1.,
                'adv_penalnty':50,#itan 500
                'reg_adv':10,
                'reg_state' : 1.,
                'reg_recon' : 1.0,#itan 10
                'similarity_reg' : 100,
                'cosine_reg': 100,
                'adversary_steps':10,
                'loss_ae':'nb',
                'intermediate_reg':1e-05,
                'intermediateEncoder1':[512,256],
                'intermediateEncoder2':[512,256],
                'intermediate_latent':256,
                'intermediate_dropout':0.1,}

# In[13]:
def compute_gradients(output, input):
    grads = torch.autograd.grad(output, input, create_graph=True)
    grads = grads[0].pow(2).mean()
    return grads

class_criterion = torch.nn.CrossEntropyLoss()
recon_criterion = NBLoss()
cosine = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
bs= model_params['batch_size']
k_folds=model_params['no_folds']
NUM_EPOCHS= model_params['epochs']
kfold=KFold(n_splits=k_folds,shuffle=True)

# In[ ]:

begin_fold = 0
if begin_fold==0:
    valAcc = []
    valF1 = []
    valPrec = []
    valRec = []
    valF1basal = []
    valAccTrans = []
    valF1Trans = []
    valF1KNN = []
    valPrecKNN = []
    valRecKNN = []
    valAccKNN = []
    valF1Species = []
    valAccSpecies = []
    valF1SpeciesTrans = []
    valAccSpeciesTrans = []
    valF1CellTrans = []
    valF1Cell = []
    mean_score_human_val = []
    mean_score_mouse_val = []
    var_score_human_val = []
    var_score_mouse_val = []
    mean_score_trans_1to2 = []
    mean_score_trans_2to1 = []
    var_score_trans_1to2 = []
    var_score_trans_2to1 = []
else:
    results = pd.read_csv('results/pretrain_10foldvalidationResults_lungs_DCSmaster_homologues.csv',index_col=0)
    mean_score_human_val = results['r2_mu_human'].tolist()
    mean_score_mouse_val = results['r2_mu_mouse'].tolist()
    var_score_human_val = results['r2_var_human'].tolist()
    var_score_mouse_val = results['r2_var_mouse'].tolist()

#pretrained_classifier = torch.load('pretrained_classifier.pth')
#begin_fold,model_params['no_folds']

# Load the homologues map
homologues_map = pd.read_csv('results/HumanMouseHomologuesMap.csv',index_col=0)

for fold in range(model_params['no_folds']):
    xtrain_mouse = torch.load('data/10foldcrossval_lung/xtrain_mouse_%s.pt' % fold)
    xtest_mouse = torch.load('data/10foldcrossval_lung/xval_mouse_%s.pt' % fold)
    #xtrain_mouse = xtrain_mouse[:,0:8000]
    #xtest_mouse = xtest_mouse[:,0:8000]
    ytrain_mouse = torch.load('data/10foldcrossval_lung/ytrain_mouse_%s.pt' % fold)
    ytest_mouse = torch.load('data/10foldcrossval_lung/yval_mouse_%s.pt' % fold)

    xtrain_human = torch.load('data/10foldcrossval_lung/xtrain_human_%s.pt' % fold)
    xtest_human = torch.load('data/10foldcrossval_lung/xval_human_%s.pt' % fold)
    #xtrain_human = xtrain_human[:,0:8000]
    #xtest_human = xtest_human[:,0:8000]
    ytrain_human = torch.load('data/10foldcrossval_lung/ytrain_human_%s.pt' % fold)
    ytest_human = torch.load('data/10foldcrossval_lung/yval_human_%s.pt' % fold)

    # keep only hologues
    xtrain_mouse = xtrain_mouse[:,homologues_map.mouse_id.values]
    xtest_mouse = xtest_mouse[:, homologues_map.mouse_id.values]
    xtrain_human = xtrain_human[:, homologues_map.human_id.values]
    xtest_human = xtest_human[:, homologues_map.human_id.values]

    gene_size_mouse = xtrain_mouse.shape[1]
    gene_size_human = xtrain_human.shape[1]

    N_2 = ytrain_mouse.shape[0]
    N_1 = ytrain_human.shape[0]

    N = N_1
    if N_2 > N:
        N = N_2
    
    # Network
    # master_encoder = torch.nn.Sequential(ElementWiseLinear(gene_size_human),
    #                     CellStateEncoder(model_params['latent_dim'],gene_size_human,
    #                                   model_params['encoder_1_hiddens'])).to(device)
    # master_decoder = CellStateDecoder(model_params['latent_dim'],gene_size_human,model_params['decoder_1_hiddens']).to(device)

    master_encoder = torch.nn.Sequential(ElementWiseLinear(gene_size_human),
                                        SimpleEncoder(gene_size_human,
                                        model_params['encoder_1_hiddens'],
                                        model_params['latent_dim'],
                                        dropRate=model_params['dropout_encoder'],
                                        activation=model_params['encoder_activation'],
                                        normalizeOutput=False)).to(device)
    master_decoder = VarDecoder(model_params['latent_dim'],model_params['decoder_1_hiddens'],gene_size_human,
                           dropRate=model_params['dropout_decoder'],
                           activation=model_params['decoder_activation']).to(device)

    allParams = list(master_encoder.parameters()) + list(master_decoder.parameters())
    optimizer = torch.optim.Adam(allParams, lr= model_params['encoding_lr'], weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=model_params['schedule_step_enc'],
                                                gamma=model_params['gamma_enc'])
    trainLoss = []
    train_eps = []
    f1_basal_trained = None 
    for e in range(0, NUM_EPOCHS):
        master_encoder.train()
        master_decoder.train()

        trainloader_1 = getSamples(N_1, bs // 2)
        len_1 = len(trainloader_1)
        trainloader_2 = getSamples(N_2, bs // 2)
        len_2 = len(trainloader_2)
        lens = [len_1, len_2]
        maxLen = np.max(lens)

        if maxLen > lens[0]:
            trainloader_suppl = getSamples(N_1, bs // 2)
            for jj in range(maxLen - lens[0]):
                trainloader_1.insert(jj, trainloader_suppl[jj])
        if maxLen > lens[1]:
            trainloader_suppl = getSamples(N_2, bs // 2)
            for jj in range(maxLen - lens[1]):
                trainloader_2.insert(jj, trainloader_suppl[jj])

        for j in range(maxLen):
            dataIndex_1 = trainloader_1[j]
            dataIndex_2 = trainloader_2[j]

            X_human = xtrain_human[dataIndex_1].to(device)
            X_mouse = xtrain_mouse[dataIndex_2].to(device)
            Y_human = ytrain_human[dataIndex_1, 0]
            Y_mouse = ytrain_mouse[dataIndex_2, 0]
            conds_human = ytrain_human[dataIndex_1, 2]
            conds_mouse = ytrain_mouse[dataIndex_2, 2]

            z_1 = master_encoder(torch.log1p(X_human))
            z_2 = master_encoder(torch.log1p(X_mouse))

            gene_means_1, gene_vars_1 = master_decoder(z_1)
            reconstruction_loss_1 = recon_criterion(gene_means_1, X_human, gene_vars_1)

            gene_means_2, gene_vars_2 = master_decoder(z_2)
            reconstruction_loss_2 = recon_criterion(gene_means_2, X_mouse, gene_vars_2)

            L1Loss = 1e-7 * (torch.mean(torch.sum(torch.abs(z_1), dim=1)) + torch.mean(torch.sum(torch.abs(z_2), dim=1)))


            loss =  model_params['reg_recon']*reconstruction_loss_1+ model_params['reg_recon']*reconstruction_loss_2 +\
                    master_encoder[1].L2Regularization(model_params['enc_l2_reg']) +\
                    model_params['enc_l2_reg'] * (torch.sum(torch.square(master_encoder[0].weight)) + torch.sum(torch.abs(master_encoder[0].bias))) + \
                    master_decoder.L2Regularization(model_params['enc_l2_reg']) + L1Loss

            loss.backward()
            optimizer.step()

        if model_params['loss_ae'] == 'nb':
            counts1, logits1 = _convert_mean_disp_to_counts_logits(
                torch.clamp(
                    gene_means_1.detach(),
                    min=1e-4,
                    max=1e4,
                ),
                torch.clamp(
                    gene_vars_1.detach(),
                    min=1e-4,
                    max=1e4,
                )
            )
            # print2log(logits1)
            dist1 = NegativeBinomial(
                total_count=counts1,
                logits=logits1
            )
            nb_sample = dist1.sample().cpu().numpy()
            yp_m1 = nb_sample.mean(0)
            yp_v1 = nb_sample.var(0)

            counts2, logits2 = _convert_mean_disp_to_counts_logits(
                torch.clamp(
                    gene_means_2.detach(),
                    min=1e-4,
                    max=1e4,
                ),
                torch.clamp(
                    gene_vars_2.detach(),
                    min=1e-4,
                    max=1e4,
                )
            )
            dist2 = NegativeBinomial(
                total_count=counts2,
                logits=logits2
            )
            nb_sample = dist2.sample().cpu().numpy()
            yp_m2 = nb_sample.mean(0)
            yp_v2 = nb_sample.var(0)
        else:
            # predicted means and variances
            yp_m1 = gene_means_1.mean(0)
            yp_v1 = gene_vars_1.mean(0)
            yp_m2 = gene_means_2.mean(0)
            yp_v2 = gene_vars_2.mean(0)
            # estimate metrics only for reasonably-sized drug/cell-type combos
        # true means and variances
        yt_m1 = X_human.detach().cpu().numpy().mean(axis=0)
        yt_v1 = X_human.detach().cpu().numpy().var(axis=0)
        yt_m2 = X_mouse.detach().cpu().numpy().mean(axis=0)
        yt_v2 = X_mouse.detach().cpu().numpy().var(axis=0)

        mean_score_human = r2_score(yt_m1, yp_m1)
        var_score_human = r2_score(yt_v1, yp_v1)
        mean_score_mouse = r2_score(yt_m2, yp_m2)
        var_score_mouse = r2_score(yt_v2, yp_v2)


        scheduler.step()
        outString = 'Split {:.0f}: Epoch={:.0f}/{:.0f}'.format(fold + 1, e + 1, NUM_EPOCHS)
        outString += ', NBloss_human={:.4f}'.format(reconstruction_loss_1.item())
        outString += ', NBloss_mouse={:.4f}'.format(reconstruction_loss_2.item())
        outString += ', mean_score_human={:.4f}'.format(mean_score_human)
        outString += ', var_score_human={:.4f}'.format(var_score_human)
        outString += ', mean_score_mouse={:.4f}'.format(mean_score_mouse)
        outString += ', var_score_mouse={:.4f}'.format(var_score_mouse)
        outString += ', loss={:.4f}'.format(loss.item())
    print2log(outString)

    master_encoder.eval()
    master_decoder.eval()

    X_human = xtest_human.to(device)
    X_mouse = xtest_mouse.to(device)
    Y_human = ytest_human[:,0]
    Y_mouse = ytest_mouse[:,0]
    conds_human = ytest_human[:, 2]
    conds_mouse = ytest_mouse[:, 2]

    z1 = master_encoder(torch.log1p(X_human))
    z2 = master_encoder(torch.log1p(X_mouse))

    gene_means_1, gene_vars_1 = master_decoder(z1)
    gene_means_2, gene_vars_2 = master_decoder(z2)

    if model_params['loss_ae'] == 'nb':
        counts1, logits1 = _convert_mean_disp_to_counts_logits(
            torch.clamp(
                gene_means_1,
                min=1e-4,
                max=1e4,
            ),
            torch.clamp(
                gene_vars_1,
                min=1e-4,
                max=1e4,
                )
            )
        dist1 = NegativeBinomial(
            total_count=counts1,
            logits=logits1
        )
        nb_sample = dist1.sample().cpu().numpy()
        yp_m1 = nb_sample.mean(0)
        yp_v1 = nb_sample.var(0)

        counts2, logits2 = _convert_mean_disp_to_counts_logits(
            torch.clamp(
                gene_means_2,
                min=1e-4,
                max=1e4,
                ),
            torch.clamp(
                gene_vars_2,
                min=1e-4,
                max=1e4,
            )
        )
        dist2 = NegativeBinomial(
            total_count=counts2,
            logits=logits2
        )
        nb_sample = dist2.sample().cpu().numpy()
        yp_m2 = nb_sample.mean(0)
        yp_v2 = nb_sample.var(0)
    else:
        # predicted means and variances
        yp_m1 = gene_means_1.mean(0)
        yp_v1 = gene_vars_1.mean(0)
        yp_m2 = gene_means_2.mean(0)
        yp_v2 = gene_vars_2.mean(0)
        # estimate metrics only for reasonably-sized drug/cell-type combos
    # true means and variances
    yt_m1 = X_human.detach().cpu().numpy().mean(axis=0)
    yt_v1 = X_human.detach().cpu().numpy().var(axis=0)
    yt_m2 = X_mouse.detach().cpu().numpy().mean(axis=0)
    yt_v2 = X_mouse.detach().cpu().numpy().var(axis=0)
    mean_score_human_val.append(r2_score(yt_m1, yp_m1))
    var_score_human_val.append(r2_score(yt_v1, yp_v1))
    mean_score_mouse_val.append(r2_score(yt_m2, yp_m2))
    var_score_mouse_val.append(r2_score(yt_v2, yp_v2))

    outString = 'Validation-set performance: Fold={:.0f}'.format(fold)
    outString += ', r2 mean score human={:.4f}'.format(mean_score_human_val[fold])
    outString += ', r2 mean score mouse={:.4f}'.format(mean_score_mouse_val[fold])
    outString += ', r2 var score human={:.4f}'.format(var_score_human_val[fold])
    outString += ', r2 var score mouse={:.4f}'.format(var_score_mouse_val[fold])
    print2log(outString)

    torch.save(master_encoder,'pre_models/master_encoder_homologues_%s.pth'%fold)
    torch.save(master_decoder,'pre_models/master_decoder_homologues_%s.pth'%fold)

    # In[ ]:
    results = pd.DataFrame({'r2_mu_human':mean_score_human_val,'r2_mu_mouse':mean_score_mouse_val,
                        'r2_var_human':var_score_human_val,'r2_var_mouse':var_score_mouse_val})
    results.to_csv('results/pretrain_10foldvalidationResults_lungs_DCSmaster_homologues.csv')
# In[ ]:
results = pd.DataFrame({'r2_mu_human':mean_score_human_val,'r2_mu_mouse':mean_score_mouse_val,
                        'r2_var_human':var_score_human_val,'r2_var_mouse':var_score_mouse_val})
results.to_csv('results/pretrain_10foldvalidationResults_lungs_DCSmaster_homologues.csv')
print2log(results)
