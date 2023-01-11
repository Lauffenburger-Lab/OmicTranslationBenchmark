import torch
import torch.nn.functional as F
from torch.distributions import NegativeBinomial
from torch.distributions.gamma import Gamma
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,cross_validate,KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score,confusion_matrix,r2_score
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
from trainingUtils import MultipleOptimizer, MultipleScheduler, compute_kernel, compute_mmd,NBLoss,_convert_mean_disp_to_counts_logits, GammaLoss#,GaussLoss
from models import Decoder, SimpleEncoder,LocalDiscriminator,PriorDiscriminator,Classifier,SpeciesCovariate, VarDecoder
from evaluationUtils import r_square,get_cindex,pearson_r,pseudoAccuracy
import argparse
import logging
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
print2log = logger.info

parser = argparse.ArgumentParser(prog='Serology cross-validation split')
parser.add_argument('--folds', action='store', default=False)
args = parser.parse_args()
k_folds = int(args.folds)

### Load data
print2log('Loading data...')
human_exprs = pd.read_csv('../data/human_exprs.csv',index_col=0)
human_metadata = pd.read_csv('../data/human_metadata.csv',index_col=0)
primates_exprs = pd.read_csv('../data/primates_exprs.csv',index_col=0)
primates_metadata = pd.read_csv('../data/primates_metadata.csv',index_col=0)
Xh = torch.tensor(human_exprs.values).double()
Xm = torch.tensor(primates_exprs.values).double()
Yh = torch.tensor(human_metadata.loc[:,['trt','infect']].values).long()
Ym = torch.tensor(primates_metadata.loc[:,['Vaccine','ProtectBinary']].values).long()

gene_size_human = len(human_exprs.columns)
gene_size_primates = len(primates_exprs.columns)


## Split in 10fold validation
dataset_human = torch.utils.data.TensorDataset(Xh,Yh)
dataset_primates = torch.utils.data.TensorDataset(Xm,Ym)
kfold=KFold(n_splits=k_folds,shuffle=True)

lm = []
for train_idx,test_idx in kfold.split(dataset_primates):
    lm.append((train_idx,test_idx))

lh = []
for train_idx,test_idx in kfold.split(dataset_human):
    lh.append((train_idx,test_idx))

print2log('Begin splitting and saving splits...')
### Begin splitting and saving splits
for i in range(k_folds):
    # Network
    xtrain_primates, ytrain_primates = dataset_primates[lm[i][0]]
    xtest_primates, ytest_primates = dataset_primates[lm[i][1]]
    xtrain_human, ytrain_human = dataset_human[lh[i][0]]
    xtest_human, ytest_human = dataset_human[lh[i][1]]

    torch.save(xtrain_primates, '../data/10fold_cross_validation/train/xtrain_primates_%s.pt' % i)
    torch.save(ytrain_primates, '../data/10fold_cross_validation/train/ytrain_primates_%s.pt' % i)
    torch.save(xtest_primates, '../data/10fold_cross_validation/train/xtest_primates_%s.pt' % i)
    torch.save(ytest_primates, '../data/10fold_cross_validation/train/ytest_primates_%s.pt' % i)

    torch.save(xtrain_human, '../data/10fold_cross_validation/train/xtrain_human_%s.pt' % i)
    torch.save(ytrain_human, '../data/10fold_cross_validation/train/ytrain_human_%s.pt' % i)
    torch.save(xtest_human, '../data/10fold_cross_validation/train/xtest_human_%s.pt' % i)
    torch.save(ytest_human, '../data/10fold_cross_validation/train/ytest_human_%s.pt' % i)