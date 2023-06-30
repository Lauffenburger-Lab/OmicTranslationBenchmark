import pickle
import torch
import torch.nn.functional as F
# from trainingUtils import MultipleOptimizer, MultipleScheduler, compute_kernel, compute_mmd
from models import Encoder,Decoder,GaussianDecoder,VAE,CellStateEncoder,\
                   CellStateDecoder, CellStateVAE,\
                   SimpleEncoder,LocalDiscriminator,PriorDiscriminator,\
                   EmbInfomax,MultiEncInfomax,Classifier,\
                   SpeciesCovariate,GaussianDecoder

# import argparse
import math
import numpy as np
import pandas as pd
#from IPython.display import clear_output
#from matplotlib import pyplot as plt
#from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split,cross_validate,KFold
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from evaluationUtils import r_square,get_cindex,pearson_r,pseudoAccuracy
#import seaborn as sns
#sns.set()


import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
print2log = logger.info
# In[2]:

mouse_df = pd.read_csv('data/all_mouse_lung.csv',index_col=0)
human_df = pd.read_csv('data/all_human_lung.csv',index_col=0)

Xm_train,Xm_test,Ym_train ,Ym_test = train_test_split(mouse_df.iloc[:,:-3].values,
                                                      mouse_df.iloc[:,-3:].values)
Xh_train,Xh_test,Yh_train, Yh_test = train_test_split(human_df.iloc[:,:-3].values,
                                                     human_df.iloc[:,-3:].values)

torch.save(Xm_test,'data/10foldcrossval_lung/xtest_mouse.pt')
torch.save(Ym_test,'data/10foldcrossval_lung/ytest_mouse.pt')
torch.save(Xh_test,'data/10foldcrossval_lung/xtest_human.pt')
torch.save(Yh_test,'data/10foldcrossval_lung/ytest_human.pt')

dataset_human = torch.utils.data.TensorDataset(torch.tensor(Xh_train).float(),torch.tensor(Yh_train).long())
dataset_mouse = torch.utils.data.TensorDataset(torch.tensor(Xm_train).float(),torch.tensor(Ym_train).long())


k_folds=10
kfold=KFold(n_splits=k_folds,shuffle=True)

lm = []
for train_idx,test_idx in kfold.split(dataset_mouse):
    lm.append((train_idx,test_idx))
    
lh = []
for train_idx,test_idx in kfold.split(dataset_human):
    lh.append((train_idx,test_idx))

for fold in range(10):
    xtrain_mouse,ytrain_mouse = dataset_mouse[lm[fold][0]]
    xtest_mouse,ytest_mouse = dataset_mouse[lm[fold][1]]
    
    xtrain_human,ytrain_human = dataset_human[lh[fold][0]]
    xtest_human,ytest_human = dataset_human[lh[fold][1]]
    
    torch.save(xtrain_mouse,'data/10foldcrossval_lung/xtrain_mouse_%s.pt'%fold)
    torch.save(xtest_mouse,'data/10foldcrossval_lung/xval_mouse_%s.pt'%fold)
    torch.save(ytrain_mouse,'data/10foldcrossval_lung/ytrain_mouse_%s.pt'%fold)
    torch.save(ytest_mouse,'data/10foldcrossval_lung/yval_mouse_%s.pt'%fold)
    
    torch.save(xtrain_human,'data/10foldcrossval_lung/xtrain_human_%s.pt'%fold)
    torch.save(xtest_human,'data/10foldcrossval_lung/xval_human_%s.pt'%fold)
    torch.save(ytrain_human,'data/10foldcrossval_lung/ytrain_human_%s.pt'%fold)
    torch.save(ytest_human,'data/10foldcrossval_lung/yval_human_%s.pt'%fold)

    print2log('Fold %s finished!'%fold)
