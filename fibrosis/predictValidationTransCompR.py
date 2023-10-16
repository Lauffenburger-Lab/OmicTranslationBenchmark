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
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
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
folds = 10

homologues_map = pd.read_csv('human lung fibrosis/HumanMouseHomologuesMap.csv',index_col=0)

nComps1 = 512
nComps2 = 512
pca1 = PCA(n_components=nComps1)
pca2 = PCA(n_components=nComps2)

mean_score_human_val = []
var_score_human_val = []
mean_score_mouse_val = []
var_score_mouse_val = []
mean_score_trans_2to1 = []
var_score_trans_2to1 = []
mean_score_trans_1to2 = []
var_score_trans_1to2 = []
for fold in range(folds):
    xtrain_mouse = torch.load('10foldcrossval_lung/xtrain_mouse_%s.pt' % fold)
    ytrain_mouse = torch.load('10foldcrossval_lung/ytrain_mouse_%s.pt' % fold)
    xtrain_human = torch.load('10foldcrossval_lung/xtrain_human_%s.pt' % fold)
    ytrain_human = torch.load('10foldcrossval_lung/ytrain_human_%s.pt' % fold)
    xtest_mouse = torch.load('10foldcrossval_lung/xval_mouse_%s.pt' % fold)
    ytest_mouse = torch.load('10foldcrossval_lung/yval_mouse_%s.pt' % fold)
    xtest_human = torch.load('10foldcrossval_lung/xval_human_%s.pt' % fold)
    ytest_human = torch.load('10foldcrossval_lung/yval_human_%s.pt' % fold)

    # keep only hologues
    xtrain_mouse = xtrain_mouse[:, homologues_map.mouse_id.values]
    xtest_mouse = xtest_mouse[:, homologues_map.mouse_id.values]
    xtrain_human = xtrain_human[:, homologues_map.human_id.values]
    xtest_human = xtest_human[:, homologues_map.human_id.values]

    gene_size_mouse = xtest_mouse.shape[1]
    gene_size_human = xtest_human.shape[1]

    # Load trained models
    decoder_human = torch.load('human lung fibrosis/models2/TransCompR_decoder_human_%s.pth' % fold)
    decoder_mouse = torch.load('human lung fibrosis/models2/TransCompR_decoder_mouse_%s.pth' % fold)

    decoder_human.eval()
    decoder_mouse.eval()

    X_human = xtest_human.to(device)
    X_mouse = xtest_mouse.to(device)
    Y_human = ytest_human[:,0]
    Y_mouse = ytest_mouse[:,0]
    conds_human = ytest_human[:, 2]
    conds_mouse = ytest_mouse[:, 2]

    # Scale data
    scaler = StandardScaler()
    X_scaled_mouse = scaler.fit_transform(torch.cat((xtrain_mouse, xtest_mouse), 0).detach().cpu().numpy())
    xtrain_mouse_scaled = torch.tensor(X_scaled_mouse[:-xtest_mouse.shape[0], :])
    xtest_mouse_scaled = torch.tensor(X_scaled_mouse[-xtest_mouse.shape[0]:, :])
    X_scaled_human = scaler.fit_transform(torch.cat((xtrain_human, xtest_human), 0).detach().cpu().numpy())
    xtrain_human_scaled = torch.tensor(X_scaled_human[:-xtest_human.shape[0], :])
    xtest_human_scaled = torch.tensor(X_scaled_human[-xtest_human.shape[0]:, :])
    pca_space_1 = pca1.fit(xtrain_human_scaled.detach().cpu().numpy())
    pca_space_2 = pca2.fit(xtrain_mouse_scaled.cpu().numpy())
    
    z1 = torch.tensor(pca_space_1.transform(xtest_human_scaled.detach().cpu().numpy())).to(device)
    z2 = torch.tensor(pca_space_2.transform(xtest_mouse_scaled.cpu().numpy())).to(device)

    gene_means_1 , gene_vars_1 = decoder_human(z1)
    gene_means_2 , gene_vars_2 = decoder_mouse(z2)


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
    nb_sample_recon1 = dist1.sample().cpu().numpy()
    yp_m1 = nb_sample_recon1.mean(0)
    yp_v1 = nb_sample_recon1.var(0)

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
    nb_sample_recon2 = dist2.sample().cpu().numpy()
    yp_m2 = nb_sample_recon2.mean(0)
    yp_v2 = nb_sample_recon2.var(0)
    # true means and variances
    yt_m1 = X_human.detach().cpu().numpy().mean(axis=0)
    yt_v1 = X_human.detach().cpu().numpy().var(axis=0)
    yt_m2 = X_mouse.detach().cpu().numpy().mean(axis=0)
    yt_v2 = X_mouse.detach().cpu().numpy().var(axis=0)
    mean_score_human_val.append(r2_score(yt_m1, yp_m1))
    var_score_human_val.append(r2_score(yt_v1, yp_v1))
    mean_score_mouse_val.append(r2_score(yt_m2, yp_m2))
    var_score_mouse_val.append(r2_score(yt_v2, yp_v2))

    # Translate to other species
    y_mu_1 , y_var_1 = decoder_human(torch.tensor(pca_space_1.transform(xtest_mouse_scaled.detach().cpu().numpy())).to(device))
    y_mu_2 , y_var_2 = decoder_mouse(torch.tensor(pca_space_2.transform(xtest_human_scaled.detach().cpu().numpy())).to(device))
    counts1, logits1 = _convert_mean_disp_to_counts_logits(
            torch.clamp(
            y_mu_1,
            min=1e-4,
            max=1e4,
        ),
            torch.clamp(
            y_var_1,
            min=1e-4,
            max=1e4,
            )
    )
    dist1 = NegativeBinomial(
            total_count=counts1,
            logits=logits1
    )
    nb_sample1 = dist1.sample().cpu().numpy()
    yp_m1 = nb_sample1.mean(0)
    yp_v1 = nb_sample1.var(0)

    counts2, logits2 = _convert_mean_disp_to_counts_logits(
        torch.clamp(
            y_mu_2,
            min=1e-4,
            max=1e4,
            ),
        torch.clamp(
            y_var_2,
            min=1e-4,
            max=1e4,
        )
    )
    dist2 = NegativeBinomial(
        total_count=counts2,
        logits=logits2
    )
    nb_sample2 = dist2.sample().cpu().numpy()
    yp_m2 = nb_sample2.mean(0)
    yp_v2 = nb_sample2.var(0)
    
    mean_score_trans_2to1.append(r2_score(yt_m1, yp_m1))
    var_score_trans_2to1.append(r2_score(yt_v1, yp_v1))
    mean_score_trans_1to2.append(r2_score(yt_m2, yp_m2))
    var_score_trans_1to2.append(r2_score(yt_v2, yp_v2))

    outString = 'Validation-set performance: Fold={:.0f}'.format(fold)
    outString += ', r2 mean score human={:.4f}'.format(mean_score_human_val[fold])
    outString += ', r2 mean score mouse={:.4f}'.format(mean_score_mouse_val[fold])
    outString += ', r2 var score human={:.4f}'.format(var_score_human_val[fold])
    outString += ', r2 var score mouse={:.4f}'.format(var_score_mouse_val[fold])
    outString += ', r2 mean score translation human to mouse={:.4f}'.format(r2_score(yt_m2, yp_m2))
    outString += ', r2 mean score translation mouse to human={:.4f}'.format(r2_score(yt_m1, yp_m1))
    print2log(outString)

    valPreds_reconstructed_human = pd.DataFrame(nb_sample_recon1)
    valPreds_reconstructed_human['diagnosis'] = Y_human.detach().cpu().numpy()
    valPreds_reconstructed_human['cell_type'] = ytest_human[:, 2]
    valPreds_reconstructed_mouse = pd.DataFrame(nb_sample_recon2)
    valPreds_reconstructed_mouse['diagnosis'] = Y_mouse.detach().cpu().numpy()
    valPreds_reconstructed_mouse['cell_type'] = ytest_mouse[:, 2]
    valPreds_reconstructed_human.to_csv('human lung fibrosis/preds/validation/valPreds_TransCompR_reconstructed_%s_human.csv' % fold)
    valPreds_reconstructed_mouse.to_csv('human lung fibrosis/preds/validation/valPreds_TransCompR_reconstructed_%s_mouse.csv' % fold)

    valPreds_translated_human = pd.DataFrame(nb_sample1)
    valPreds_translated_human['mouse_diagnosis'] = Y_mouse.detach().cpu().numpy()
    valPreds_translated_human['mouse_cell_type'] = ytest_mouse[:, 2]
    valPreds_translated_mouse = pd.DataFrame(nb_sample2)
    valPreds_translated_mouse['human_diagnosis'] = Y_human.detach().cpu().numpy()
    valPreds_translated_mouse['human_cell_type'] = ytest_human[:, 2]
    valPreds_translated_human.to_csv('human lung fibrosis/preds/validation/valPreds_TransCompR_translated_%s_tohuman.csv' % fold)
    valPreds_translated_mouse.to_csv('human lung fibrosis/preds/validation/valPreds_TransCompR_translated_%s_tomouse.csv' % fold)

    # In[ ]:
    results = pd.DataFrame({'r2_mu_human':mean_score_human_val,'r2_mu_mouse':mean_score_mouse_val,
                        'r2_var_human':var_score_human_val,'r2_var_mouse':var_score_mouse_val,
                        'r2_mu_human_to_mouse':mean_score_trans_1to2,'r2_mu_mouse_to_human':mean_score_trans_2to1,
                        'r2_var_human_to_mouse':var_score_trans_1to2,'r2_var_mouse_to_human':var_score_trans_2to1})
    results.to_csv('human lung fibrosis/10foldvalidationResults_TransCompR_lungs_perCellType.csv')
print2log(results)
