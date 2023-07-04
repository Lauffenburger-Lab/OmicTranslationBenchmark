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
import matplotlib.patches as patches
import seaborn as sns
sns.set()

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
print2log = logger.info

parser = argparse.ArgumentParser(prog='TransCompR mixed with ANNs approaches')
parser.add_argument('--filter_pcs', action='store', default=False)
parser.add_argument('--latent_dim', action='store', default=32)
# parser.add_argument('--outPattern', action='store')
args = parser.parse_args()
filter_pcs = args.filter_pcs
latent_dim = args.latent_dim
# outPattern = args.outPattern
if latent_dim is None:
    print2log('No latent dimension was defined.Filtered PCs will be used.')
    filter_pcs = True
else:
    latent_dim = int(latent_dim)

device = torch.device('cuda')
# Initialize environment and seeds for reproducability
torch.backends.cudnn.benchmark = True
def seed_everything(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
# Create a train generators
def getSamples(N, batchSize):
    order = np.random.permutation(N)
    outList = []
    while len(order)>0:
        outList.append(order[0:batchSize])
        order = order[batchSize:]
    return outList
# Gradient penalnty function
def compute_gradients(output, input):
    grads = torch.autograd.grad(output, input, create_graph=True)
    grads = grads[0].pow(2).mean()
    return grads
### Load cell-line data
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


# ## Split in 10fold validation
# dataset_human = torch.utils.data.TensorDataset(Xh,Yh)
# dataset_primates = torch.utils.data.TensorDataset(Xm,Ym)
k_folds=10
# kfold=KFold(n_splits=k_folds,shuffle=True)
#
# lm = []
# for train_idx,test_idx in kfold.split(dataset_primates):
#     lm.append((train_idx,test_idx))
#
# lh = []
# for train_idx,test_idx in kfold.split(dataset_human):
#     lh.append((train_idx,test_idx))

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


# ## Split in 10fold validation
# dataset_human = torch.utils.data.TensorDataset(Xh,Yh)
# dataset_primates = torch.utils.data.TensorDataset(Xm,Ym)
#k_folds=10
if filter_pcs==True:
    exp_var_pca1 = pca_space_1.explained_variance_ratio_
    cum_sum_eigenvalues_1 = np.cumsum(exp_var_pca1)
    nComps1 = np.min(np.where(cum_sum_eigenvalues_1>=0.99)[0])
    pca1 = PCA(n_components=nComps1)
    pca_space_1 = pca1.fit(human_exprs.values)
    exp_var_pca2 = pca_space_2.explained_variance_ratio_
    cum_sum_eigenvalues_2 = np.cumsum(exp_var_pca2)
    nComps2 = np.min(np.where(cum_sum_eigenvalues_2 >= 0.99)[0])
    pca2 = PCA(n_components=nComps2)
    pca_space_2 = pca2.fit(primates_exprs.values)
    pca_transformed_1 = pca_space_1.transform(human_exprs.values)
    pca_transformed_2 = pca_space_2.transform(primates_exprs.values)
else:
    nComps1 = latent_dim
    nComps2 = latent_dim
    pca1 = PCA(n_components=nComps1)
    pca2 = PCA(n_components=nComps2)
    pca_space_1 = pca1.fit(human_exprs.values)
    pca_space_2 = pca2.fit(primates_exprs.values)
    pca_transformed_1 = pca_space_1.transform(human_exprs.values)
    pca_transformed_2 = pca_space_2.transform(primates_exprs.values)
    exp_var_pca1 = pca_space_1.explained_variance_ratio_
    exp_var_pca2 = pca_space_2.explained_variance_ratio_

model_params = {'encoder_1_hiddens':[64],
                'encoder_2_hiddens':[64],
                'decoder_1_hiddens': [64],
                'decoder_2_hiddens': [128],
                'latent_dim1': nComps1,
                'latent_dim2': nComps2,
                'dropout_decoder': 0.2,
                'decoder_dropIn':0.0,
                'dropout_encoder':0.1,
                'encoder_activation':torch.nn.ELU(),#torch.nn.LeakyReLU(negative_slope=0.9),
                'decoder_activation':torch.nn.ELU(),
                'V_dropout':0.5,
                'intermediateEncoder1':[nComps1],
                'intermediateEncoder2':[nComps2],
                'intermediate_latent1':nComps1,
                'intermediate_latent2':nComps2,
                'intermediate_enc_l2_reg':1e-05,
                'intermediate_dropout':0.1,
                'inter_dropIn': 0.1,
                'state_class_hidden':[32,16,8],#[32,16,8,4],
                'state_class_drop_in':0.25,
                'state_class_drop':0.1,
                'no_states':2,
                'protect_class_hidden':[32,16,8],#[32,16,8,4,2],
                'protect_class_drop_in':0.25,
                'protect_class_drop':0.1,
                'protect_states':2,
                'species_class_hidden':[32,16,8],
                'species_class_drop_in':0.1,
                'species_class_drop':0.1,
                'no_species':2,
                'adv_class_hidden':[32,16,8],
                'adv_class_drop_in':0.2,
                'adv_class_drop':0.1,
                'no_adv_class':2,
                'encoding_lr':0.001,
                'adv_lr':0.001,
                'schedule_step_adv':300,
                'gamma_adv':0.5,
                'schedule_step_enc':300,
                'gamma_enc':0.8,
                'batch_size_1':35,
                'batch_size_2':15,
                'epochs':2000, # it was 1000 for only Vsp
                'prior_beta':1.0,
                'no_folds':k_folds,
                'v_reg':1e-04,
                'state_class_reg':1e-05,
                'species_class_reg':1e-04,
                'enc_l2_reg':1e-06,
                'dec_l2_reg':1e-06,
                'lambda_mi_loss':100.,
                'effsize_reg': 1.,
                'cosine_loss': 70,
                'adv_penalnty':1.,
                'reg_adv':100,
                'reg_classifier': 100.,
                'similarity_reg' : 40.,
                'adversary_steps':4,
                'autoencoder_wd': 0.,
                'adversary_wd': 0.}


### Begin pre-training decoder
NUM_EPOCHS= model_params['epochs']
bs_1 = model_params['batch_size_1']
bs_2 =  model_params['batch_size_2']
# recon_criterion = GammaLoss()
recon_criterion = torch.nn.GaussianNLLLoss(full=True)

print2log('Training Decoder architecture to predict serology...')
mean_human = []
var_human = []
mean_primates = []
var_primates = []

r2_human = []
pear_human = []
r2_primates = []
pear_primates = []

pear_matrix_primates = np.zeros((model_params['no_folds'],Xm.shape[1]))
pear_matrix_human = np.zeros((model_params['no_folds'],Xh.shape[1]))

pear_matrix_primates_latent = np.zeros((model_params['no_folds'],nComps2))
pear_matrix_human_latent = np.zeros((model_params['no_folds'],nComps1))

# print2log('Train decoder for primates')
# for i in range(model_params['no_folds']):
#     # Network
#     xtrain_primates = torch.load('../data/10fold_cross_validation/train/xtrain_primates_%s.pt' % i)
#     ytrain_primates = torch.load('../data/10fold_cross_validation/train/ytrain_primates_%s.pt' % i)
#     xtest_primates = torch.load('../data/10fold_cross_validation/train/xtest_primates_%s.pt' % i)
#     ytest_primates = torch.load('../data/10fold_cross_validation/train/ytest_primates_%s.pt' % i)
#     xtrain_human = torch.load('../data/10fold_cross_validation/train/xtrain_human_%s.pt' % i)
#     ytrain_human = torch.load('../data/10fold_cross_validation/train/ytrain_human_%s.pt' % i)
#     xtest_human = torch.load('../data/10fold_cross_validation/train/xtest_human_%s.pt' % i)
#     ytest_human = torch.load('../data/10fold_cross_validation/train/ytest_human_%s.pt' % i)
#
#     gene_size_primates = xtrain_primates.shape[1]
#     gene_size_human = xtrain_human.shape[1]
#
#     N_2 = ytrain_primates.shape[0]
#     N_1 = ytrain_human.shape[0]
#
#     N = N_2
#
#     decoder_2 = Decoder(nComps2, model_params['decoder_2_hiddens'], gene_size_primates,
#                         dropRate=model_params['dropout_decoder'],dropIn= model_params['decoder_dropIn'],
#                         activation=model_params['decoder_activation']).to(device)
#
#     allParams = list(decoder_2.parameters())
#     optimizer = torch.optim.Adam(allParams, lr=model_params['encoding_lr'], weight_decay=0)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
#                                                 step_size=model_params['schedule_step_enc'],
#                                                 gamma=model_params['gamma_enc'])
#     for e in range(0, NUM_EPOCHS):
#         decoder_2.train()
#         trainloader_2 = getSamples(N, bs_2)
#         len_2 = len(trainloader_2)
#
#         lens = [len_2]
#         maxLen = np.max(lens)
#
#         for j in range(maxLen):
#             dataIndex_2 = trainloader_2[j]
#
#             X_primates = xtrain_primates[dataIndex_2,:].float().to(device)
#             X2_transformed = torch.tensor(pca_space_2.transform(xtrain_primates[dataIndex_2,:].numpy())).float()
#             z = X2_transformed.to(device)
#             optimizer.zero_grad()
#
#
#             y_pred_2 = decoder_2(z)
#             # gene_means_2, gene_vars_2 = decoder_2(z)
#             # reconstruction_loss_2 = recon_criterion(gene_means_2, X_primates, gene_vars_2)
#             fitLoss = torch.mean(torch.sum((y_pred_2 - X_primates) ** 2, dim=1))
#             L2Loss = decoder_2.L2Regularization(model_params['dec_l2_reg'])
#             # loss = reconstruction_loss_2 + L2Loss
#             loss = fitLoss + L2Loss
#
#             loss.backward()
#             optimizer.step()
#
#             # # dist2 = Gamma(concentration=torch.clamp(gene_means_2.detach(),min=1e-4,max=1e4)/torch.clamp(gene_vars_2.detach(),min=1e-4,max=1e4),
#             # #               rate=1./torch.clamp(gene_vars_2.detach(),min=1e-4,max=1e4))
#             # # nb_sample = dist2.sample().cpu().numpy()
#             # # yp_m2 = nb_sample.mean(0)
#             # # yp_v2 = nb_sample.var(0)
#             # yp_m2 = gene_means_2.detach().cpu().numpy().mean(0)
#             # yp_v2 = gene_vars_2.detach().cpu().numpy().mean(0)
#             # yt_m2 = X_primates.detach().cpu().numpy().mean(axis=0)
#             # yt_v2 = X_primates.detach().cpu().numpy().var(axis=0)
#             # mean_score_primates = r2_score(yt_m2, yp_m2)
#             # var_score_primates = r2_score(yt_v2, yp_v2)
#
#         pearson = pearson_r(y_pred_2.detach(), X_primates.detach())
#         r2 = r_square(y_pred_2.detach(), X_primates.detach())
#         mse = torch.mean(torch.mean((y_pred_2.detach() - X_primates.detach()) ** 2, dim=1))
#
#         scheduler.step()
#         outString = 'Split {:.0f}: Epoch={:.0f}/{:.0f}'.format(i + 1, e + 1, NUM_EPOCHS)
#         outString += ', r2={:.4f}'.format(torch.mean(r2).item())
#         outString += ', pearson={:.4f}'.format(torch.mean(pearson).item())
#         outString += ', MSE={:.4f}'.format(mse.item())
#         # outString += ', recon_loss={:.4f}'.format(reconstruction_loss_2.item())
#         # outString += ', r2_mean={:.4f}'.format(mean_score_primates.item())
#         # outString += ', r2_var={:.4f}'.format(var_score_primates.item())
#
#         outString += ', loss={:.4f}'.format(loss.item())
#         if (e % 200 == 0):
#             print2log(outString)
#     print2log(outString)
#     torch.save(decoder_2, '../results_intermediate_encoders/pretrained_models/decoder_primates_%s.pt' % i)

# print2log('Train decoder for human')
# for i in range(model_params['no_folds']):
#     # Network
#     xtrain_primates = torch.load('../data/10fold_cross_validation/train/xtrain_primates_%s.pt' % i)
#     ytrain_primates = torch.load('../data/10fold_cross_validation/train/ytrain_primates_%s.pt' % i)
#     xtest_primates = torch.load('../data/10fold_cross_validation/train/xtest_primates_%s.pt' % i)
#     ytest_primates = torch.load('../data/10fold_cross_validation/train/ytest_primates_%s.pt' % i)
#     xtrain_human = torch.load('../data/10fold_cross_validation/train/xtrain_human_%s.pt' % i)
#     ytrain_human = torch.load('../data/10fold_cross_validation/train/ytrain_human_%s.pt' % i)
#     xtest_human = torch.load('../data/10fold_cross_validation/train/xtest_human_%s.pt' % i)
#     ytest_human = torch.load('../data/10fold_cross_validation/train/ytest_human_%s.pt' % i)
#
#     gene_size_primates = xtrain_primates.shape[1]
#     gene_size_human = xtrain_human.shape[1]
#
#     N_2 = ytrain_primates.shape[0]
#     N_1 = ytrain_human.shape[0]
#
#     N = N_1
#
#     decoder_1 = Decoder(nComps1, model_params['decoder_1_hiddens'], gene_size_human,
#                         dropRate=model_params['dropout_decoder'],dropIn= model_params['decoder_dropIn'],
#                         activation=model_params['decoder_activation']).to(device)
#
#     allParams = list(decoder_1.parameters())
#     optimizer = torch.optim.Adam(allParams, lr=model_params['encoding_lr'], weight_decay=0)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
#                                                 step_size=model_params['schedule_step_enc'],
#                                                 gamma=model_params['gamma_enc'])
#     for e in range(0, NUM_EPOCHS):
#         decoder_1.train()
#         trainloader_1 = getSamples(N, bs_1)
#         len_1 = len(trainloader_2)
#
#         lens = [len_1]
#         maxLen = np.max(lens)
#
#         for j in range(maxLen):
#             dataIndex_1 = trainloader_1[j]
#
#             X_human= xtrain_human[dataIndex_1,:].float().to(device)
#             X1_transformed = torch.tensor(pca_space_1.transform(xtrain_human[dataIndex_1,:].numpy())).float().to(device)
#             z = X1_transformed.to(device)
#             optimizer.zero_grad()
#
#             y_pred_1 = decoder_1(z)
#             # gene_means_1, gene_vars_1 = decoder_1(z)
#             # reconstruction_loss_1 = recon_criterion(gene_means_1, X_human, gene_vars_1)
#             fitLoss = torch.mean(torch.sum((y_pred_1 - X_human) ** 2, dim=1))
#             L2Loss = decoder_1.L2Regularization(model_params['dec_l2_reg'])
#             # loss = reconstruction_loss_1 + L2Loss
#             loss = fitLoss + L2Loss
#
#             loss.backward()
#             optimizer.step()
#
#             # # dist1 = Gamma(concentration=torch.clamp(gene_means_1.detach(), min=1e-4, max=1e4) / torch.clamp(gene_vars_1.detach(),min=1e-4, max=1e4),
#             # #               rate=1. / torch.clamp(gene_vars_1.detach(), min=1e-4, max=1e4))
#             # # nb_sample = dist1.sample().cpu().numpy()
#             # # yp_m1 = nb_sample.mean(0)
#             # # yp_v1 = nb_sample.var(0)
#             # yp_m1 = gene_means_1.detach().cpu().numpy().mean(0)
#             # yp_v1 = gene_vars_1.detach().cpu().numpy().mean(0)
#             # yt_m1 = X_human.detach().cpu().numpy().mean(axis=0)
#             # yt_v1 = X_human.detach().cpu().numpy().var(axis=0)
#             # mean_score_human = r2_score(yt_m1, yp_m1)
#             # var_score_human = r2_score(yt_v1, yp_v1)
#
#         pearson = pearson_r(y_pred_1.detach(), X_human.detach())
#         r2 = r_square(y_pred_1.detach(), X_human.detach())
#         mse = torch.mean(torch.mean((y_pred_1.detach() - X_human.detach()) ** 2, dim=1))
#
#         scheduler.step()
#         outString = 'Split {:.0f}: Epoch={:.0f}/{:.0f}'.format(i + 1, e + 1, NUM_EPOCHS)
#         outString += ', r2={:.4f}'.format(torch.mean(r2).item())
#         outString += ', pearson={:.4f}'.format(torch.mean(pearson).item())
#         outString += ', MSE={:.4f}'.format(mse.item())
#         # outString += ', recon_loss={:.4f}'.format(reconstruction_loss_1.item())
#         # outString += ', r2_mean={:.4f}'.format(mean_score_human.item())
#         # outString += ', r2_var={:.4f}'.format(var_score_human.item())
#         outString += ', loss={:.4f}'.format(loss.item())
#         if (e % 200 == 0):
#             print2log(outString)
#     print2log(outString)
#     torch.save(decoder_1, '../results_intermediate_encoders/pretrained_models/decoder_human_%s.pt' % i)
#
# print2log('Evaluate translation using decoders')
# for i in range(model_params['no_folds']):
#     decoder_1 = torch.load('../results_intermediate_encoders/pretrained_models/decoder_human_%s.pt' % i)
#     decoder_2 = torch.load('../results_intermediate_encoders/pretrained_models/decoder_primates_%s.pt' % i)
#     xtrain_primates = torch.load('../data/10fold_cross_validation/train/xtrain_primates_%s.pt' % i)
#     ytrain_primates = torch.load('../data/10fold_cross_validation/train/ytrain_primates_%s.pt' % i)
#     xtest_primates = torch.load('../data/10fold_cross_validation/train/xtest_primates_%s.pt' % i)
#     ytest_primates = torch.load('../data/10fold_cross_validation/train/ytest_primates_%s.pt' % i)
#     xtrain_human = torch.load('../data/10fold_cross_validation/train/xtrain_human_%s.pt' % i)
#     ytrain_human = torch.load('../data/10fold_cross_validation/train/ytrain_human_%s.pt' % i)
#     xtest_human = torch.load('../data/10fold_cross_validation/train/xtest_human_%s.pt' % i)
#     ytest_human = torch.load('../data/10fold_cross_validation/train/ytest_human_%s.pt' % i)
#     decoder_1.eval()
#     decoder_2.eval()
#
#     x1_all = xtest_human.float().to(device)
#     x2_all = xtest_primates.float().to(device)
#     ypred_2 = decoder_2(torch.tensor(pca_space_2.transform(xtest_primates.numpy())).float().to(device))
#     ypred_1 = decoder_1(torch.tensor(pca_space_1.transform(xtest_human.numpy())).float().to(device))
#
#     # # dist2 = Gamma(
#     # #     concentration=torch.clamp(gene_means_2.detach(), min=1e-4, max=1e4) / torch.clamp(gene_vars_2.detach(),
#     # #                                                                                       min=1e-4, max=1e4),
#     # #     rate=1. / torch.clamp(gene_vars_2.detach(), min=1e-4, max=1e4))
#     # # nb_sample = dist2.sample().cpu().numpy()
#     # # yp_m2 = nb_sample.mean(0)
#     # # yp_v2 = nb_sample.var(0)
#     # yp_m2 = gene_means_2.detach().cpu().numpy().mean(0)
#     # yp_v2 = gene_vars_2.detach().cpu().numpy().mean(0)
#     # yt_m2 = X_primates.detach().cpu().numpy().mean(axis=0)
#     # yt_v2 = X_primates.detach().cpu().numpy().var(axis=0)
#     # mean_score_primates = r2_score(yt_m2, yp_m2)
#     # var_score_primates = r2_score(yt_v2, yp_v2)
#     # # dist1 = Gamma(
#     # #     concentration=torch.clamp(gene_means_1.detach(), min=1e-4, max=1e4) / torch.clamp(gene_vars_1.detach(),
#     # #                                                                                       min=1e-4, max=1e4),
#     # #     rate=1. / torch.clamp(gene_vars_1.detach(), min=1e-4, max=1e4))
#     # # nb_sample = dist1.sample().cpu().numpy()
#     # # yp_m1 = nb_sample.mean(0)
#     # # yp_v1 = nb_sample.var(0)
#     # yp_m1 = gene_means_1.detach().cpu().numpy().mean(0)
#     # yp_v1 = gene_vars_1.detach().cpu().numpy().mean(0)
#     # yt_m1 = X_human.detach().cpu().numpy().mean(axis=0)
#     # yt_v1 = X_human.detach().cpu().numpy().var(axis=0)
#     # mean_score_human = r2_score(yt_m1, yp_m1)
#     # var_score_human = r2_score(yt_v1, yp_v1)
#     #
#     # mean_human.append(mean_score_human)
#     # var_human.append(var_score_human)
#     # mean_primates.append(mean_score_primates)
#     # var_primates.append(var_score_primates)
#     #
#     # print2log('R2 mean human: %s' % mean_score_human)
#     # print2log('R2 var human: %s' % var_score_human)
#     # print2log('R2 mean primates: %s' % mean_score_primates)
#     # print2log('R2 var primates: %s' % var_score_primates)
#
#     pearson_1 = pearson_r(ypred_1.detach(), x1_all.detach())
#     r2_1 = r_square(ypred_1.detach(), x1_all.detach())
#     pearson_2 = pearson_r(ypred_2.detach(), x2_all.detach())
#     r2_2 = r_square(ypred_2.detach(), x2_all.detach())
#
#     print2log('R2  human: %s' % torch.mean(r2_1).item())
#     print2log('Pearson  human: %s' % torch.mean(pearson_1).item())
#     print2log('R2  primates: %s' % torch.mean(r2_2).item())
#     print2log('Pearson primates: %s' % torch.mean(pearson_2).item())
#
#     r2_primates.append(torch.mean(r2_2).item())
#     pear_primates.append(torch.mean(pearson_2).item())
#     r2_human.append( torch.mean(r2_1).item())
#     pear_human.append( torch.mean(pearson_1).item())
#
#     pear_matrix_primates[i,:] = pearson_2.detach().cpu().numpy()
#     pear_matrix_human[i,:] = pearson_1.detach().cpu().numpy()
#
#
#
#
# # df_result = pd.DataFrame({'r2_mean_human':mean_human ,'r2_var_human':var_human,
# #                           'r2_mean_primates':mean_primates ,'r2_var_primates':var_primates})
# df_result = pd.DataFrame({'r2_human':r2_human ,'pear_human':pear_human,
#                           'r2_primates':r2_primates ,'pear_primates':pear_primates})
# df_result.to_csv('../results_intermediate_encoders/10foldvalidation_pretrained_decoders_'+str(latent_dim)+'dim'+str(NUM_EPOCHS)+'ep.csv')
# print2log(df_result)
#
# pear_matrix_primates = pd.DataFrame(pear_matrix_primates)
# pear_matrix_primates.columns = primates_exprs.columns
# pear_matrix_primates.to_csv('../results_intermediate_encoders/10foldvalidation_pretrained_decoders_'+str(latent_dim)+'dim'+str(NUM_EPOCHS)+'ep_perFeature_primates.csv')
# pear_matrix_primates = pd.melt(pear_matrix_primates)
# pear_matrix_primates.columns = ['feature','pearson']
# grouped = pear_matrix_primates.groupby(['feature']).median().sort_values(by='pearson',ascending=False)
# sns.set_theme(style="whitegrid")
# plt.figure(figsize=(9,12), dpi= 80)
# ax = sns.boxplot(x="pearson", y="feature", data=pear_matrix_primates,order=grouped.index,orient='h')
# plt.legend(loc='lower left')
# plt.gca().set(title='Per feature performance of primate decoder in 10-fold cross-validation',
#               xlabel = 'pearson correlation',
#               ylabel='feature names')
# ax.yaxis.set_tick_params(labelsize = 5)
# for ind, label in enumerate(ax.get_yticklabels()):
#     if ind % 5 == 0:  # every 10th label is kept
#         label.set_visible(True)
#     else:
#         label.set_visible(False)
# #plt.xlim(0,1)
# plt.savefig('../results_intermediate_encoders/perFeature_performance_pretrained_decoder_'+str(latent_dim)+'dim'+str(NUM_EPOCHS)+'ep_primates.png', bbox_inches='tight',dpi=600)
#
#
# pear_matrix_human = pd.DataFrame(pear_matrix_human)
# pear_matrix_human.columns = human_exprs.columns
# pear_matrix_human.to_csv('../results_intermediate_encoders/10foldvalidation_pretrained_decoders_'+str(latent_dim)+'dim'+str(NUM_EPOCHS)+'ep_perFeature_human.csv')
# pear_matrix_human = pd.melt(pear_matrix_human)
# pear_matrix_human.columns = ['feature','pearson']
# grouped = pear_matrix_human.groupby(['feature']).median().sort_values(by='pearson',ascending=False)
# sns.set_theme(style="whitegrid")
# plt.figure(figsize=(9,12), dpi= 80)
# ax = sns.boxplot(x="pearson", y="feature", data=pear_matrix_human,order=grouped.index,orient='h')
# # ax.yaxis.tick_right()
# plt.legend(loc='lower left')
# plt.gca().set(title='Per feature performance of human decoder in 10-fold cross-validation',
#               xlabel = 'pearson correlation',
#               ylabel='feature names')
# plt.xlim(0,1)
# ax.yaxis.set_tick_params(labelsize = 5)
# plt.savefig('../results_intermediate_encoders/perFeature_performance_pretrained_decoder_'+str(latent_dim)+'dim'+str(NUM_EPOCHS)+'ep_human.png', bbox_inches='tight',dpi=600)
#
#
# ## Train encoders
# print2log('Training encoder architecture to predict PCA...')
#
# print2log('Train basal encoder for primates')
# # NUM_EPOCHS_similarity = 2000
# #model_params["no_folds"]
# for i in range(model_params['no_folds']):
#     # Network
#     xtrain_primates = torch.load('../data/10fold_cross_validation/train/xtrain_primates_%s.pt' % i)
#     ytrain_primates = torch.load('../data/10fold_cross_validation/train/ytrain_primates_%s.pt' % i)
#     xtest_primates = torch.load('../data/10fold_cross_validation/train/xtest_primates_%s.pt' % i)
#     ytest_primates = torch.load('../data/10fold_cross_validation/train/ytest_primates_%s.pt' % i)
#     xtrain_human = torch.load('../data/10fold_cross_validation/train/xtrain_human_%s.pt' % i)
#     ytrain_human = torch.load('../data/10fold_cross_validation/train/ytrain_human_%s.pt' % i)
#     xtest_human = torch.load('../data/10fold_cross_validation/train/xtest_human_%s.pt' % i)
#     ytest_human = torch.load('../data/10fold_cross_validation/train/ytest_human_%s.pt' % i)
#
#     gene_size_primates = xtrain_primates.shape[1]
#     gene_size_human = xtrain_human.shape[1]
#
#     N_2 = ytrain_primates.shape[0]
#     N_1 = ytrain_human.shape[0]
#
#     N = N_2
#
#     encoder_2 = SimpleEncoder(gene_size_primates, model_params['encoder_2_hiddens'], model_params['latent_dim2'],
#                         dropRate=model_params['dropout_encoder'],dropIn=0,
#                         activation=model_params['encoder_activation']).to(device)
#
#
#     allParams = list(encoder_2.parameters())
#     optimizer = torch.optim.Adam(allParams, lr=model_params['encoding_lr'], weight_decay=0)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
#                                                 step_size=model_params['schedule_step_enc'],
#                                                 gamma=model_params['gamma_enc'])
#     for e in range(0, NUM_EPOCHS): #NUM_EPOCHS
#         encoder_2.train()
#         # Vsp.train()
#         trainloader_2 = getSamples(N, bs_2*3)
#         len_2 = len(trainloader_2)
#
#         lens = [len_2]
#         maxLen = np.max(lens)
#
#         for j in range(maxLen):
#             dataIndex_2 = trainloader_2[j]
#
#             X_primates = xtrain_primates[dataIndex_2, :].float().to(device)
#             X2_transformed = torch.tensor(pca_space_2.transform(xtrain_primates[dataIndex_2, :].numpy())).float()
#             z = X2_transformed.to(device)
#             # z_species_2 = torch.cat((torch.zeros(X_primates.shape[0], 1),
#             #                          torch.ones(X_primates.shape[0], 1)), 1).to(device)
#             conditions = np.concatenate((ytrain_primates[dataIndex_2, 0], ytrain_primates[dataIndex_2, 0]))
#             size = conditions.size
#             conditions = conditions.reshape(size, 1)
#             conditions = conditions == conditions.transpose()
#             conditions = conditions * 1
#             mask = torch.tensor(conditions).to(device).detach()
#             optimizer.zero_grad()
#
#             z_base_2 = encoder_2(X_primates)
#             L2Loss = encoder_2.L2Regularization(model_params['enc_l2_reg'])
#
#             # Similarity in basal space
#             silimalityLoss = torch.sum(torch.cdist(torch.cat((z_base_2, z_base_2), 0), torch.cat((z_base_2, z_base_2), 0)) * mask.float()) / mask.float().sum()
#             w1 = torch.cat((z_base_2, z_base_2), 0).norm(p=2, dim=1, keepdim=True)
#             w2 = torch.cat((z_base_2, z_base_2), 0).norm(p=2, dim=1, keepdim=True)
#             cosineLoss = torch.mm(torch.cat((z_base_2, z_base_2), 0), torch.cat((z_base_2, z_base_2), 0).t()) / (w1 * w2.t()).clamp(min=1e-6)
#             cosineLoss = torch.sum(cosineLoss * mask.float()) / mask.float().sum()
#
#             loss =  L2Loss + silimalityLoss - 10*cosineLoss
#
#             loss.backward()
#             optimizer.step()
#
#         scheduler.step()
#         outString = 'Split {:.0f}: Epoch={:.0f}/{:.0f}'.format(i + 1, e + 1, NUM_EPOCHS)
#         outString += ', CosineLoss={:.4f}'.format(cosineLoss.item())
#         outString += ', silimalityLoss={:.4f}'.format(silimalityLoss.item())
#         outString += ', loss={:.4f}'.format(loss.item())
#         if (e % 200 == 0):
#             print2log(outString)
#     print2log(outString)
#     torch.save(encoder_2, '../results_intermediate_encoders/pretrained_models/encoder_primates_%s.pt' % i)
#
# print2log('Train basal encoder for human')
# for i in range(model_params['no_folds']):
#     # Network
#     xtrain_primates = torch.load('../data/10fold_cross_validation/train/xtrain_primates_%s.pt' % i)
#     ytrain_primates = torch.load('../data/10fold_cross_validation/train/ytrain_primates_%s.pt' % i)
#     xtest_primates = torch.load('../data/10fold_cross_validation/train/xtest_primates_%s.pt' % i)
#     ytest_primates = torch.load('../data/10fold_cross_validation/train/ytest_primates_%s.pt' % i)
#     xtrain_human = torch.load('../data/10fold_cross_validation/train/xtrain_human_%s.pt' % i)
#     ytrain_human = torch.load('../data/10fold_cross_validation/train/ytrain_human_%s.pt' % i)
#     xtest_human = torch.load('../data/10fold_cross_validation/train/xtest_human_%s.pt' % i)
#     ytest_human = torch.load('../data/10fold_cross_validation/train/ytest_human_%s.pt' % i)
#
#     gene_size_primates = xtrain_primates.shape[1]
#     gene_size_human = xtrain_human.shape[1]
#
#     N_2 = ytrain_primates.shape[0]
#     N_1 = ytrain_human.shape[0]
#
#     N = N_1
#
#     encoder_1 = SimpleEncoder(gene_size_human, model_params['encoder_1_hiddens'], model_params['latent_dim1'],
#                         dropRate=model_params['dropout_encoder'],dropIn=0,
#                         activation=model_params['encoder_activation']).to(device)
#
#     allParams = list(encoder_1.parameters())
#     optimizer = torch.optim.Adam(allParams, lr=model_params['encoding_lr'], weight_decay=0)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
#                                                 step_size=model_params['schedule_step_enc'],
#                                                 gamma=model_params['gamma_enc'])
#     for e in range(0, NUM_EPOCHS):
#         encoder_1.train()
#         # Vsp.train()
#         trainloader_1 = getSamples(N, bs_1*2)
#         len_1 = len(trainloader_2)
#
#         lens = [len_1]
#         maxLen = np.max(lens)
#
#         for j in range(maxLen):
#             dataIndex_1 = trainloader_1[j]
#
#             X_human = xtrain_human[dataIndex_1, :].float().to(device)
#             X1_transformed = torch.tensor(pca_space_1.transform(xtrain_human[dataIndex_1, :].numpy())).float()
#             z = X1_transformed.to(device)
#             # z_species_1 = torch.cat((torch.ones(X_human.shape[0], 1),
#             #                          torch.zeros(X_human.shape[0], 1)), 1).to(device)
#             conditions = np.concatenate((ytrain_human[dataIndex_1, 0], ytrain_human[dataIndex_1, 0]))
#             size = conditions.size
#             conditions = conditions.reshape(size, 1)
#             conditions = conditions == conditions.transpose()
#             conditions = conditions * 1
#             mask = torch.tensor(conditions).to(device).detach()
#
#             optimizer.zero_grad()
#
#             z_base_1 = encoder_1(X_human)
#             L2Loss = encoder_1.L2Regularization(model_params['enc_l2_reg'])
#
#             # Similarity in basal space
#             silimalityLoss = torch.sum(torch.cdist(torch.cat((z_base_1, z_base_1), 0), torch.cat((z_base_1, z_base_1), 0)) * mask.float()) / mask.float().sum()
#             w1 = torch.cat((z_base_1, z_base_1), 0).norm(p=2, dim=1, keepdim=True)
#             w2 = torch.cat((z_base_1, z_base_1), 0).norm(p=2, dim=1, keepdim=True)
#             cosineLoss = torch.mm(torch.cat((z_base_1, z_base_1), 0), torch.cat((z_base_1, z_base_1), 0).t()) / (w1 * w2.t()).clamp(min=1e-6)
#             cosineLoss = torch.sum(cosineLoss * mask.float()) / mask.float().sum()
#
#             loss = L2Loss + silimalityLoss - 10*cosineLoss
#
#             loss.backward()
#             optimizer.step()
#
#         scheduler.step()
#         outString = 'Split {:.0f}: Epoch={:.0f}/{:.0f}'.format(i + 1, e + 1, NUM_EPOCHS)
#         outString += ', CosineLoss={:.4f}'.format(cosineLoss.item())
#         outString += ', silimalityLoss={:.4f}'.format(silimalityLoss.item())
#         outString += ', loss={:.4f}'.format(loss.item())
#         if (e % 200 == 0):
#             print2log(outString)
#     print2log(outString)
#     torch.save(encoder_1, '../results_intermediate_encoders/pretrained_models/encoder_human_%s.pt' % i)
#
# mean_human = []
# var_human = []
# mean_primates = []
# var_primates = []
#
# r2_human = []
# pear_human = []
# r2_primates = []
# pear_primates = []
#
# pear_matrix_primates = np.zeros((model_params['no_folds'],nComps2))
# pear_matrix_human = np.zeros((model_params['no_folds'],nComps1))
#
# print2log('Train encoder for primates')
# #model_params["no_folds"]
# for i in range(model_params['no_folds']):
#     # Network
#     xtrain_primates = torch.load('../data/10fold_cross_validation/train/xtrain_primates_%s.pt' % i)
#     ytrain_primates = torch.load('../data/10fold_cross_validation/train/ytrain_primates_%s.pt' % i)
#     xtest_primates = torch.load('../data/10fold_cross_validation/train/xtest_primates_%s.pt' % i)
#     ytest_primates = torch.load('../data/10fold_cross_validation/train/ytest_primates_%s.pt' % i)
#     xtrain_human = torch.load('../data/10fold_cross_validation/train/xtrain_human_%s.pt' % i)
#     ytrain_human = torch.load('../data/10fold_cross_validation/train/ytrain_human_%s.pt' % i)
#     xtest_human = torch.load('../data/10fold_cross_validation/train/xtest_human_%s.pt' % i)
#     ytest_human = torch.load('../data/10fold_cross_validation/train/ytest_human_%s.pt' % i)
#
#     gene_size_primates = xtrain_primates.shape[1]
#     gene_size_human = xtrain_human.shape[1]
#
#     N_2 = ytrain_primates.shape[0]
#     N_1 = ytrain_human.shape[0]
#
#     N = N_2
#
#     encoder_2 = torch.load('../results_intermediate_encoders/pretrained_models/encoder_primates_%s.pt' % i)
#     # encoder_2 = SimpleEncoder(gene_size_primates, model_params['encoder_2_hiddens'], model_params['latent_dim2'],
#     #                     dropRate=model_params['dropout_encoder'],dropIn=0,
#     #                     activation=model_params['encoder_activation']).to(device)
#     # Vsp = SpeciesCovariate(2, model_params['latent_dim2'], dropRate=model_params['V_dropout']).to(device)
#     encoder_interm_2 = SimpleEncoder(model_params['latent_dim2'],
#                                      model_params['intermediateEncoder2'],
#                                      model_params['intermediate_latent2'],
#                                      dropRate=model_params['intermediate_dropout'],dropIn = model_params['inter_dropIn'],
#                                      activation=model_params['encoder_activation']).to(device)
#     # pseudoInverse = torch.matmul(encoder_2.linear_layers[0].weight.data.T,torch.inverse(torch.matmul(encoder_2.linear_layers[0].weight.data,encoder_2.linear_layers[0].weight.data.T)))
#     k=0
#     for m in encoder_2.modules():
#         if isinstance(m, torch.nn.Linear):
#             if k ==0:
#                 W_enc = m.weight.data
#             else:
#                 W_enc = torch.matmul(m.weight.data,W_enc)
#             k+=1
#     pseudoInverse =torch.matmul(W_enc.T,torch.inverse(torch.matmul(W_enc,W_enc.T)))
#     Winit = torch.matmul(torch.tensor(pca_space_2.components_).float().to(device), pseudoInverse)
#     # encoder_2.linear_latent.weight.data = Winit
#     k=0
#     for m in encoder_interm_2.modules():
#         if isinstance(m, torch.nn.Linear):
#             if k==0:
#                 m.weight.data = Winit
#             else:
#                 m.weight.data = torch.eye(m.weight.data.shape[0]).to(device)
#             k += 1
#             if m.bias is not None:
#                 m.bias.data.fill_(0.0)
#     #encoder_interm_2.linear_latent.weight.data = torch.eye(model_params['intermediate_latent2']).to(device)
#     # Vsp.Vspecies.weight.data.fill_(0.0)
#
#
#     # allParams = list(encoder_2.parameters())+ list(Vsp.parameters())
#     allParams = list(encoder_2.parameters()) + list(encoder_interm_2.parameters())
#     # allParams = list(encoder_interm_2.parameters())
#     optimizer = torch.optim.Adam(allParams, lr=model_params['encoding_lr'], weight_decay=0)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
#                                                 step_size=model_params['schedule_step_enc'],
#                                                 gamma=model_params['gamma_enc'])
#     for e in range(0, NUM_EPOCHS): #NUM_EPOCHS
#         encoder_2.train()
#         # encoder_2.eval()
#         # Vsp.train()
#         encoder_interm_2.train()
#         trainloader_2 = getSamples(N, bs_2*3)
#         len_2 = len(trainloader_2)
#
#         lens = [len_2]
#         maxLen = np.max(lens)
#
#         for j in range(maxLen):
#             dataIndex_2 = trainloader_2[j]
#
#             X_primates = xtrain_primates[dataIndex_2, :].float().to(device)
#             X2_transformed = torch.tensor(pca_space_2.transform(xtrain_primates[dataIndex_2, :].numpy())).float()
#             z = X2_transformed.to(device)
#             # z_species_2 = torch.cat((torch.zeros(X_primates.shape[0], 1),
#             #                          torch.ones(X_primates.shape[0], 1)), 1).to(device)
#             conditions = np.concatenate((ytrain_primates[dataIndex_2, 0], ytrain_primates[dataIndex_2, 0]))
#             size = conditions.size
#             conditions = conditions.reshape(size, 1)
#             conditions = conditions == conditions.transpose()
#             conditions = conditions * 1
#             mask = torch.tensor(conditions).to(device).detach()
#             optimizer.zero_grad()
#
#             z_base_2 = encoder_2(X_primates)
#             # y_pred_2 = Vsp(z_base_2, z_species_2)
#             y_pred_2 = encoder_interm_2(z_base_2)
#             fitLoss = torch.mean(torch.sum((y_pred_2 - z) ** 2, dim=1))
#             # fitLoss = torch.sum(torch.tensor(exp_var_pca2).to(device) * torch.sum((y_pred_2 - z) ** 2, dim=0)) # try weighting based on the importance of each PC
#             # L2Loss = encoder_2.L2Regularization(model_params['enc_l2_reg']) + Vsp.Regularization(model_params['v_reg'])
#             L2Loss = encoder_2.L2Regularization(model_params['enc_l2_reg'])
#             # Similarity in basal space
#             silimalityLoss = torch.sum(torch.cdist(torch.cat((z_base_2, z_base_2), 0), torch.cat((z_base_2, z_base_2),
#                                                                                                  0)) * mask.float()) / mask.float().sum()
#             w1 = torch.cat((z_base_2, z_base_2), 0).norm(p=2, dim=1, keepdim=True)
#             w2 = torch.cat((z_base_2, z_base_2), 0).norm(p=2, dim=1, keepdim=True)
#             cosineLoss = torch.mm(torch.cat((z_base_2, z_base_2), 0), torch.cat((z_base_2, z_base_2), 0).t()) / (
#                     w1 * w2.t()).clamp(min=1e-6)
#             cosineLoss = torch.sum(cosineLoss * mask.float()) / mask.float().sum()
#
#             loss = L2Loss + fitLoss + silimalityLoss - 10*cosineLoss
#
#             loss.backward()
#             optimizer.step()
#
#             pearson = pearson_r(y_pred_2.detach(), z.detach())
#             r2 = r_square(y_pred_2.detach(), z.detach())
#             mse = torch.mean(torch.mean((y_pred_2.detach() - z.detach()) ** 2, dim=1))
#
#         scheduler.step()
#         outString = 'Split {:.0f}: Epoch={:.0f}/{:.0f}'.format(i + 1, e + 1, NUM_EPOCHS)
#         outString += ', r2={:.4f}'.format(torch.mean(r2).item())
#         outString += ', pearson={:.4f}'.format(torch.mean(pearson).item())
#         outString += ', MSE={:.4f}'.format(mse.item())
#         outString += ', CosineLoss={:.4f}'.format(cosineLoss.item())
#         outString += ', silimalityLoss={:.4f}'.format(silimalityLoss.item())
#         outString += ', fit_loss={:.4f}'.format(fitLoss.item())
#         # outString += ', recon_loss={:.4f}'.format(reconstruction_loss_1.item())
#         # outString += ', r2_mean={:.4f}'.format(mean_score_human.item())
#         # outString += ', r2_var={:.4f}'.format(var_score_human.item())
#         outString += ', loss={:.4f}'.format(loss.item())
#         if (e % 200 == 0):
#             print2log(outString)
#     print2log(outString)
#     torch.save(encoder_2, '../results_intermediate_encoders/pretrained_models/encoder_primates_%s.pt' % i)
#     # torch.save(Vsp, '../results_intermediate_encoders/pretrained_models/pre_trained_Vsp_%s.pt' % i)
#     torch.save(encoder_interm_2, '../results_intermediate_encoders/pretrained_models/encoder_intermediate_primates_%s.pt' % i)
#
# print2log('Train encoder for human')
# for i in range(model_params['no_folds']):
#     # Network
#     xtrain_primates = torch.load('../data/10fold_cross_validation/train/xtrain_primates_%s.pt' % i)
#     ytrain_primates = torch.load('../data/10fold_cross_validation/train/ytrain_primates_%s.pt' % i)
#     xtest_primates = torch.load('../data/10fold_cross_validation/train/xtest_primates_%s.pt' % i)
#     ytest_primates = torch.load('../data/10fold_cross_validation/train/ytest_primates_%s.pt' % i)
#     xtrain_human = torch.load('../data/10fold_cross_validation/train/xtrain_human_%s.pt' % i)
#     ytrain_human = torch.load('../data/10fold_cross_validation/train/ytrain_human_%s.pt' % i)
#     xtest_human = torch.load('../data/10fold_cross_validation/train/xtest_human_%s.pt' % i)
#     ytest_human = torch.load('../data/10fold_cross_validation/train/ytest_human_%s.pt' % i)
#
#     gene_size_primates = xtrain_primates.shape[1]
#     gene_size_human = xtrain_human.shape[1]
#
#     N_2 = ytrain_primates.shape[0]
#     N_1 = ytrain_human.shape[0]
#
#     N = N_1
#
#     encoder_1 = torch.load('../results_intermediate_encoders/pretrained_models/encoder_human_%s.pt' % i)
#     # encoder_1 = SimpleEncoder(gene_size_human, model_params['encoder_1_hiddens'], model_params['latent_dim1'],
#     #                     dropRate=model_params['dropout_encoder'],dropIn=0,
#     #                     activation=model_params['encoder_activation']).to(device)
#     # Vsp = SpeciesCovariate(2, model_params['latent_dim1'], dropRate=model_params['V_dropout']).to(device)
#     # pretrained_Vsp = torch.load('../results_intermediate_encoders/pretrained_models/pre_trained_Vsp_%s.pt' % i)
#     # Vsp.load_state_dict(pretrained_Vsp.state_dict())
#     encoder_interm_1 = SimpleEncoder(model_params['latent_dim1'],
#                                      model_params['intermediateEncoder1'],
#                                      model_params['intermediate_latent1'],
#                                      dropRate=model_params['intermediate_dropout'],dropIn = model_params['inter_dropIn'],
#                                      activation=model_params['encoder_activation']).to(device)
#     # pseudoInverse = torch.matmul(encoder_1.linear_layers[0].weight.data.T,torch.inverse(torch.matmul(encoder_1.linear_layers[0].weight.data,encoder_1.linear_layers[0].weight.data.T)))
#     k = 0
#     for m in encoder_1.modules():
#         if isinstance(m, torch.nn.Linear):
#             if k == 0:
#                 W_enc = m.weight.data
#             else:
#                 W_enc = torch.matmul(m.weight.data,W_enc)
#             k += 1
#     pseudoInverse =torch.matmul(W_enc.T,torch.inverse(torch.matmul(W_enc,W_enc.T)))
#     Winit = torch.matmul(torch.tensor(pca_space_1.components_).float().to(device), pseudoInverse)
#     # encoder_1.linear_latent.weight.data = Winit
#     k = 0
#     for m in encoder_interm_1.modules():
#         if isinstance(m, torch.nn.Linear):
#             if k == 0:
#                 m.weight.data = Winit
#             else:
#                 m.weight.data = torch.eye(m.weight.data.shape[0]).to(device)
#             k += 1
#             if m.bias is not None:
#                 m.bias.data.fill_(0.0)
#
#     # allParams = list(encoder_1.parameters()) + list(Vsp.parameters())
#     allParams = list(encoder_1.parameters()) + list(encoder_interm_1.parameters())
#     # allParams = list(encoder_interm_1.parameters())
#     optimizer = torch.optim.Adam(allParams, lr=model_params['encoding_lr'], weight_decay=0)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
#                                                 step_size=model_params['schedule_step_enc'],
#                                                 gamma=model_params['gamma_enc'])
#     for e in range(0, NUM_EPOCHS):
#         encoder_1.train()
#         # encoder_1.eval()
#         # Vsp.train()
#         encoder_interm_1.train()
#         trainloader_1 = getSamples(N, bs_1*2)
#         len_1 = len(trainloader_2)
#
#         lens = [len_1]
#         maxLen = np.max(lens)
#
#         for j in range(maxLen):
#             dataIndex_1 = trainloader_1[j]
#
#             X_human = xtrain_human[dataIndex_1, :].float().to(device)
#             X1_transformed = torch.tensor(pca_space_1.transform(xtrain_human[dataIndex_1, :].numpy())).float()
#             z = X1_transformed.to(device)
#             # z_species_1 = torch.cat((torch.ones(X_human.shape[0], 1),
#             #                          torch.zeros(X_human.shape[0], 1)), 1).to(device)
#             conditions = np.concatenate((ytrain_human[dataIndex_1, 0], ytrain_human[dataIndex_1, 0]))
#             size = conditions.size
#             conditions = conditions.reshape(size, 1)
#             conditions = conditions == conditions.transpose()
#             conditions = conditions * 1
#             mask = torch.tensor(conditions).to(device).detach()
#
#             optimizer.zero_grad()
#
#             z_base_1 = encoder_1(X_human)
#             # y_pred_1 = Vsp(z_base_1, z_species_1)
#             y_pred_1 = encoder_interm_1(z_base_1)
#             fitLoss = torch.mean(torch.sum((y_pred_1 - z) ** 2, dim=1))
#             # fitLoss = torch.sum(torch.tensor(exp_var_pca1).to(device) * torch.sum((y_pred_1 - z) ** 2,dim=0))  # try weighting based on the importance of each PC
#             # L2Loss = encoder_1.L2Regularization(model_params['enc_l2_reg']) #+ Vsp.Regularization(model_params['v_reg'])
#             L2Loss = encoder_1.L2Regularization(model_params['enc_l2_reg'])
#
#             # Similarity in basal space
#             silimalityLoss = torch.sum(torch.cdist(torch.cat((z_base_1, z_base_1), 0), torch.cat((z_base_1, z_base_1),
#                                                                                                  0)) * mask.float()) / mask.float().sum()
#             w1 = torch.cat((z_base_1, z_base_1), 0).norm(p=2, dim=1, keepdim=True)
#             w2 = torch.cat((z_base_1, z_base_1), 0).norm(p=2, dim=1, keepdim=True)
#             cosineLoss = torch.mm(torch.cat((z_base_1, z_base_1), 0), torch.cat((z_base_1, z_base_1), 0).t()) / (
#                     w1 * w2.t()).clamp(min=1e-6)
#             cosineLoss = torch.sum(cosineLoss * mask.float()) / mask.float().sum()
#
#             loss = L2Loss + fitLoss +silimalityLoss - 10*cosineLoss
#
#             loss.backward()
#             optimizer.step()
#
#             pearson = pearson_r(y_pred_1.detach(), z.detach())
#             r2 = r_square(y_pred_1.detach(), z.detach())
#             mse = torch.mean(torch.mean((y_pred_1.detach() - z.detach()) ** 2, dim=1))
#
#         scheduler.step()
#         outString = 'Split {:.0f}: Epoch={:.0f}/{:.0f}'.format(i + 1, e + 1, NUM_EPOCHS)
#         outString += ', r2={:.4f}'.format(torch.mean(r2).item())
#         outString += ', pearson={:.4f}'.format(torch.mean(pearson).item())
#         outString += ', MSE={:.4f}'.format(mse.item())
#         outString += ', CosineLoss={:.4f}'.format(cosineLoss.item())
#         outString += ', silimalityLoss={:.4f}'.format(silimalityLoss.item())
#         outString += ', fit_loss={:.4f}'.format(fitLoss.item())
#         # outString += ', recon_loss={:.4f}'.format(reconstruction_loss_1.item())
#         # outString += ', r2_mean={:.4f}'.format(mean_score_human.item())
#         # outString += ', r2_var={:.4f}'.format(var_score_human.item())
#         outString += ', loss={:.4f}'.format(loss.item())
#         if (e % 200 == 0):
#             print2log(outString)
#     print2log(outString)
#     torch.save(encoder_1, '../results_intermediate_encoders/pretrained_models/encoder_human_%s.pt' % i)
#     # torch.save(Vsp,'../results_intermediate_encoders/pretrained_models/pre_trained_Vsp_%s.pt' % i)
#     torch.save(encoder_interm_2,'../results_intermediate_encoders/pretrained_models/encoder_intermediate_human_%s.pt' % i)
#
# print2log('Evaluate translation using encoders')
# for i in range(model_params['no_folds']):
#     encoder_1 = torch.load('../results_intermediate_encoders/pretrained_models/encoder_human_%s.pt' % i)
#     encoder_2 = torch.load('../results_intermediate_encoders/pretrained_models/encoder_primates_%s.pt' % i)
#     # Vsp = torch.load('../results_intermediate_encoders/pretrained_models/pre_trained_Vsp_%s.pt' % i)
#     encoder_interm_1 = torch.load('../results_intermediate_encoders/pretrained_models/encoder_intermediate_human_%s.pt' % i)
#     encoder_interm_2 = torch.load('../results_intermediate_encoders/pretrained_models/encoder_intermediate_primates_%s.pt' % i)
#
#     encoder_1.eval()
#     encoder_2.eval()
#     # Vsp.eval()
#     encoder_interm_1.eval()
#     encoder_interm_2.eval()
#
#     xtrain_primates = torch.load('../data/10fold_cross_validation/train/xtrain_primates_%s.pt' % i)
#     ytrain_primates = torch.load('../data/10fold_cross_validation/train/ytrain_primates_%s.pt' % i)
#     xtest_primates = torch.load('../data/10fold_cross_validation/train/xtest_primates_%s.pt' % i)
#     ytest_primates = torch.load('../data/10fold_cross_validation/train/ytest_primates_%s.pt' % i)
#     xtrain_human = torch.load('../data/10fold_cross_validation/train/xtrain_human_%s.pt' % i)
#     ytrain_human = torch.load('../data/10fold_cross_validation/train/ytrain_human_%s.pt' % i)
#     xtest_human = torch.load('../data/10fold_cross_validation/train/xtest_human_%s.pt' % i)
#     ytest_human = torch.load('../data/10fold_cross_validation/train/ytest_human_%s.pt' % i)
#
#     x1_all = xtest_human.float().to(device)
#     x2_all = xtest_primates.float().to(device)
#     x1_transformed = torch.tensor(pca_space_1.transform(xtest_human.numpy())).float().to(device)
#     x2_transformed = torch.tensor(pca_space_2.transform(xtest_primates.numpy())).float().to(device)
#     # z_species_1 = torch.cat((torch.ones(x1_all.shape[0], 1),
#     #                          torch.zeros(x1_all.shape[0], 1)), 1).to(device)
#     # z_species_2 = torch.cat((torch.zeros(x2_all.shape[0], 1),
#     #                          torch.ones(x2_all.shape[0], 1)), 1).to(device)
#     z_base_1 = encoder_1(x1_all)
#     # ypred_1 = Vsp(z_base_1, z_species_1)
#     z_base_2 = encoder_2(x2_all)
#     # ypred_2 = Vsp(z_base_2, z_species_2)
#     ypred_2 = encoder_interm_2(z_base_2)
#     ypred_1 = encoder_interm_1(z_base_1)
#
#     pearson_1 = pearson_r(ypred_1.detach(), x1_transformed.detach())
#     r2_1 = r_square(ypred_1.detach(), x1_transformed.detach())
#     pearson_2 = pearson_r(ypred_2.detach(), x2_transformed.detach())
#     r2_2 = r_square(ypred_2.detach(), x2_transformed.detach())
#
#     print2log('R2  human: %s' % torch.mean(r2_1).item())
#     print2log('Pearson  human: %s' % torch.mean(pearson_1).item())
#     print2log('R2  primates: %s' % torch.mean(r2_2).item())
#     print2log('Pearson primates: %s' % torch.mean(pearson_2).item())
#
#     r2_primates.append(torch.mean(r2_2).item())
#     pear_primates.append(torch.mean(pearson_2).item())
#     r2_human.append(torch.mean(r2_1).item())
#     pear_human.append(torch.mean(pearson_1).item())
#
#     pear_matrix_primates[i, :] = pearson_2.detach().cpu().numpy()
#     pear_matrix_human[i,:] = pearson_1.detach().cpu().numpy()
#
# df_result = pd.DataFrame({'r2_human':r2_human ,'pear_human':pear_human,
#                           'r2_primates':r2_primates ,'pear_primates':pear_primates})
# df_result.to_csv('../results_intermediate_encoders/10foldvalidation_pretrained_encoders_'+str(latent_dim)+'dim'+str(NUM_EPOCHS)+'ep.csv')
# print2log(df_result)
#
# pear_matrix_primates = pd.DataFrame(pear_matrix_primates)
# pear_matrix_primates.columns = ['PC'+str(d+1) for d in range(latent_dim)]
# pear_matrix_primates.to_csv('../results_intermediate_encoders/10foldvalidation_pretrained_encoders_'+str(latent_dim)+'dim'+str(NUM_EPOCHS)+'ep_perFeature_primates.csv')
# pear_matrix_primates = pd.melt(pear_matrix_primates)
# pear_matrix_primates.columns = ['PC','pearson']
# # grouped = pear_matrix_primates.groupby(['PC']).median().sort_values(by='pearson',ascending=False)
# sns.set_theme(style="whitegrid")
# plt.figure(figsize=(9,12), dpi= 80)
# ax = sns.boxplot(x="pearson", y="PC", data=pear_matrix_primates,orient='h') #order=grouped.index
# ax.axhline(n2,ls='--',color='red')
# plt.legend(loc='lower left')
# plt.gca().set(title='Per principal component performance of primate encoder in 10-fold cross-validation',
#               xlabel = 'pearson correlation',
#               ylabel='PC')
# # ax.yaxis.set_tick_params(labelsize = 5)
# # for ind, label in enumerate(ax.get_yticklabels()):
# #     if ind % 5 == 0:  # every 10th label is kept
# #         label.set_visible(True)
# #     else:
# #         label.set_visible(False)
# # plt.xlim(0,1)
# plt.savefig('../results_intermediate_encoders/perFeature_performance_pretrained_encoder_'+str(latent_dim)+'dim'+str(NUM_EPOCHS)+'ep_primates.png', bbox_inches='tight',dpi=600)
#
#
# pear_matrix_human = pd.DataFrame(pear_matrix_human)
# pear_matrix_human.columns = ['PC'+str(d+1) for d in range(latent_dim)]
# pear_matrix_human.to_csv('../results_intermediate_encoders/10foldvalidation_pretrained_encoders_'+str(latent_dim)+'dim'+str(NUM_EPOCHS)+'ep_perFeature_human.csv')
# pear_matrix_human = pd.melt(pear_matrix_human)
# pear_matrix_human.columns = ['PC','pearson']
# # grouped = pear_matrix_human.groupby(['PC']).median().sort_values(by='pearson',ascending=False)
# sns.set_theme(style="whitegrid")
# plt.figure(figsize=(9,12), dpi= 80)
# ax = sns.boxplot(x="pearson", y="PC", data=pear_matrix_human,orient='h') #order=grouped.index
# ax.axhline(n1,ls='--',color='red')
# # ax.yaxis.tick_right()
# plt.legend(loc='lower left')
# plt.gca().set(title='Per principal component performance of human encoder in 10-fold cross-validation',
#               xlabel = 'pearson correlation',
#               ylabel='PC')
# # plt.xlim(0,1)
# # ax.yaxis.set_tick_params(labelsize = 5)
# plt.savefig('../results_intermediate_encoders/perFeature_performance_pretrained_encoder_'+str(latent_dim)+'dim'+str(NUM_EPOCHS)+'ep_human.png', bbox_inches='tight',dpi=600)



## Pre-train adverse classifier
print2log('Pre-train adverse classifier')
class_criterion = torch.nn.CrossEntropyLoss()
for i in range(model_params["no_folds"]):
    # # Network
    # pre_encoder_1 = torch.load('../results_intermediate_encoders/pretrained_models/encoder_human_%s.pt'%i)
    # pre_encoder_2 = torch.load('../results_intermediate_encoders/pretrained_models/encoder_primates_%s.pt' % i)
    # # pre_encoder_interm_1 = torch.load('../results_intermediate_encoders/pretrained_models/encoder_intermediate_human_%s.pt' % i)
    # # pre_encoder_interm_2 = torch.load('../results_intermediate_encoders/pretrained_models/encoder_intermediate_primates_%s.pt' % i)
    pre_encoder_1 = SimpleEncoder(gene_size_human, model_params['encoder_1_hiddens'], nComps1,
                              dropRate=model_params['dropout_encoder'],
                              activation=model_params['encoder_activation']).to(device)
    pre_encoder_2 = SimpleEncoder(gene_size_primates, model_params['encoder_2_hiddens'], nComps2,
                              dropRate=model_params['dropout_encoder'],
                              activation=model_params['encoder_activation']).to(device)
    adverse_classifier = Classifier(in_channel=model_params['latent_dim1'],
                                    hidden_layers=model_params['adv_class_hidden'],
                                    num_classes=model_params['no_adv_class'],
                                    drop_in=0.25,
                                    drop=0.1).to(device)

    xtrain_primates = torch.load('../data/10fold_cross_validation/train/xtrain_primates_%s.pt' % i)
    ytrain_primates = torch.load('../data/10fold_cross_validation/train/ytrain_primates_%s.pt' % i)
    xtest_primates = torch.load('../data/10fold_cross_validation/train/xtest_primates_%s.pt' % i)
    ytest_primates = torch.load('../data/10fold_cross_validation/train/ytest_primates_%s.pt' % i)
    xtrain_human = torch.load('../data/10fold_cross_validation/train/xtrain_human_%s.pt' % i)
    ytrain_human = torch.load('../data/10fold_cross_validation/train/ytrain_human_%s.pt' % i)
    xtest_human = torch.load('../data/10fold_cross_validation/train/xtest_human_%s.pt' % i)
    ytest_human = torch.load('../data/10fold_cross_validation/train/ytest_human_%s.pt' % i)

    gene_size_primates = xtrain_primates.shape[1]
    gene_size_human = xtrain_human.shape[1]

    N_2 = ytrain_primates.shape[0]
    N_1 = ytrain_human.shape[0]
    N = N_1
    if N_2 > N:
        N = N_2

    allParams = list(pre_encoder_1.parameters()) + list(pre_encoder_2.parameters())
    allParams = allParams + list(adverse_classifier.parameters())
    optimizer_adv = torch.optim.Adam(allParams, lr=model_params['encoding_lr'], weight_decay=0)
    scheduler_adv = torch.optim.lr_scheduler.StepLR(optimizer_adv,
                                                    step_size=model_params['schedule_step_enc'],
                                                    gamma=model_params['gamma_enc'])
    for e in range(0, int(NUM_EPOCHS/2)):
        # pre_encoder_1.eval()
        # pre_encoder_2.eval()
        pre_encoder_1.train()
        pre_encoder_2.train()
        adverse_classifier.train()

        trainloader_1 = getSamples(N_1, bs_1)
        len_1 = len(trainloader_1)
        trainloader_2 = getSamples(N_2, bs_2)
        len_2 = len(trainloader_2)

        lens = [len_1, len_2]
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

        for j in range(maxLen):
            dataIndex_1 = trainloader_1[j]
            dataIndex_2 = trainloader_2[j]

            X_2 = xtrain_primates[dataIndex_2, :].float().to(device)
            # z_species_2 = torch.cat((torch.zeros(X_2.shape[0], 1),
            #                          torch.ones(X_2.shape[0], 1)), 1).to(device)
            X_1 = xtrain_human[dataIndex_1, :].float().to(device)
            # z_species_1 = torch.cat((torch.ones(X_1.shape[0], 1),
            #                          torch.zeros(X_1.shape[0], 1)), 1).to(device)
            optimizer_adv.zero_grad()

            z_base_1 = pre_encoder_1(X_1)
            z_base_2 = pre_encoder_2(X_2)

            latent_base_vectors = torch.cat((z_base_1, z_base_2), 0)

            # Remove signal from z_basal
            labels_adv = adverse_classifier(latent_base_vectors)
            true_labels = torch.cat((torch.ones(z_base_1.shape[0]),
                                     torch.zeros(z_base_2.shape[0])), 0).long().to(device)
            adv_entropy = class_criterion(labels_adv, true_labels)
            _, predicted = torch.max(labels_adv, 1)
            predicted = predicted.cpu().numpy()
            cf_matrix = confusion_matrix(true_labels.cpu().numpy(), predicted, labels=[0, 1])
            tn, fp, fn, tp = cf_matrix.ravel()
            f1_basal = 2 * tp / (2 * tp + fp + fn)

            loss =   adv_entropy + adverse_classifier.L2Regularization(model_params['state_class_reg'])

            loss.backward()
            optimizer_adv.step()

        if (e >= 0):
            scheduler_adv.step()
            outString = 'Split {:.0f}: Epoch={:.0f}/{:.0f}'.format(i + 1, e + 1, int(NUM_EPOCHS/2))
            outString += ', Adverse Entropy={:.4f}'.format(adv_entropy.item())
            outString += ', loss={:.4f}'.format(loss.item())
            outString += ', F1 basal={:.4f}'.format(f1_basal)
        if (e % 50 == 0):
            print2log(outString)
    print2log(outString)
    pre_encoder_1.eval()
    pre_encoder_2.eval()
    adverse_classifier.eval()

    x_1 = xtest_human.float().to(device)
    x_2 = xtest_primates.float().to(device)

    z_latent_base_1 = pre_encoder_1(x_1)
    z_latent_base_2 = pre_encoder_2(x_2)

    labels = adverse_classifier(torch.cat((z_latent_base_1, z_latent_base_2), 0))
    true_labels = torch.cat((torch.ones(z_latent_base_1.shape[0]).view(z_latent_base_1.shape[0], 1),
                             torch.zeros(z_latent_base_2.shape[0]).view(z_latent_base_2.shape[0], 1)), 0).long()
    _, predicted = torch.max(labels, 1)
    predicted = predicted.cpu().numpy()
    cf_matrix = confusion_matrix(true_labels.numpy(), predicted, labels=[0, 1])
    tn, fp, fn, tp = cf_matrix.ravel()
    class_acc = (tp + tn) / predicted.size
    f1 = 2 * tp / (2 * tp + fp + fn)

    print2log('Classification accuracy: %s' % class_acc)
    print2log('Classification F1 score: %s' % f1)

    torch.save(adverse_classifier, '../results_intermediate_encoders/pretrained_models/10fold/pre_trained_classifier_adverse_%s.pt' % i)

### Train whole translational model
print2log('Train translation model')
valR2 = []
valPear = []

valPear_1 = []
valR2_1 = []
valPear_2 = []
valR2_2 = []

valPear_1_latent = []
valPear_2_latent = []

valPearDirect = []

valF1 = []
valClassAcc = []

valF1Species = []
valAccSpecies = []
valF1Protection = []

valF1KNNTrans = []
valF1ClassTrans = []

valAccProtection = []
valF1Global = []
valClassAccGlobal = []

#Reduce epochs and sceduler step
# NUM_EPOCHS = int(NUM_EPOCHS/2)
# model_params['epochs'] = NUM_EPOCHS
bs_1 = 70
bs_2 =  30
# model_params['schedule_step_enc'] = int(model_params['schedule_step_enc']/2)
# model_params['schedule_step_adv'] = int(model_params['schedule_step_adv']/2)
class_criterion = torch.nn.CrossEntropyLoss()

pear_matrix_primates = np.zeros((model_params['no_folds'],Xm.shape[1]))
pear_matrix_human = np.zeros((model_params['no_folds'],Xh.shape[1]))

# pear_matrix_primates_latent = np.zeros((model_params['no_folds'],nComps2))
# pear_matrix_human_latent = np.zeros((model_params['no_folds'],nComps1))

pear_matrix_primates_latent = np.zeros((model_params['no_folds'],nComps2,nComps2))
pear_matrix_human_latent = np.zeros((model_params['no_folds'],nComps1,nComps1))

for i in range(model_params["no_folds"]):
    xtrain_primates = torch.load('../data/10fold_cross_validation/train/xtrain_primates_%s.pt' % i)
    ytrain_primates = torch.load('../data/10fold_cross_validation/train/ytrain_primates_%s.pt' % i)
    xtest_primates = torch.load('../data/10fold_cross_validation/train/xtest_primates_%s.pt' % i)
    ytest_primates = torch.load('../data/10fold_cross_validation/train/ytest_primates_%s.pt' % i)
    xtrain_human = torch.load('../data/10fold_cross_validation/train/xtrain_human_%s.pt' % i)
    ytrain_human = torch.load('../data/10fold_cross_validation/train/ytrain_human_%s.pt' % i)
    xtest_human = torch.load('../data/10fold_cross_validation/train/xtest_human_%s.pt' % i)
    ytest_human = torch.load('../data/10fold_cross_validation/train/ytest_human_%s.pt' % i)

    gene_size_primates = xtrain_primates.shape[1]
    gene_size_human = xtrain_human.shape[1]

    N_2 = ytrain_primates.shape[0]
    N_1 = ytrain_human.shape[0]
    N = N_1
    if N_2 > N:
        N = N_2

    # # Network
    # # pre_encoder_1 = torch.load('../results_intermediate_encoders/models/pre_encoder_human_%s.pt'%i)
    # # pre_encoder_2 = torch.load('../results_intermediate_encoders/models/pre_encoder_primates_%s.pt' % i)
    # pre_encoder_1 = torch.load('../results_intermediate_encoders/pretrained_models/encoder_human_%s.pt' % i)
    # pre_encoder_2 = torch.load('../results_intermediate_encoders/pretrained_models/encoder_primates_%s.pt' % i)
    # pre_decoder_1 = torch.load('../results_intermediate_encoders/pretrained_models/decoder_human_%s.pt' % i)
    # pre_decoder_2 = torch.load('../results_intermediate_encoders/pretrained_models/decoder_primates_%s.pt' % i)
    # #pre_classifier = torch.load('../results_intermediate_encoders/models/pre_classifier_%s.pt' % i)
    # #pre_species_classifier = torch.load('../results_intermediate_encoders/models/pre_species_classifier_%s.pt' % i)
    # # pre_prior_d = torch.load('../results_intermediate_encoders/models/pre_prior_d_%s.pt' % i)
    # # pre_local_d = torch.load('../results_intermediate_encoders/models/pre_local_d_%s.pt' % i)
    # pre_encoder_interm_1 = torch.load('../results_intermediate_encoders/pretrained_models/encoder_intermediate_human_%s.pt' % i)
    # pre_encoder_interm_2 = torch.load('../results_intermediate_encoders/pretrained_models/encoder_intermediate_primates_%s.pt' % i)

    decoder_1 = Decoder(nComps1, model_params['decoder_1_hiddens'], gene_size_human,
                        dropRate=model_params['dropout_decoder'],dropIn= model_params['decoder_dropIn'],
                        activation=model_params['decoder_activation']).to(device)
    # decoder_1.load_state_dict(pre_decoder_1.state_dict())
    decoder_2 = Decoder(nComps2, model_params['decoder_2_hiddens'], gene_size_primates,
                        dropRate=model_params['dropout_decoder'],dropIn= model_params['decoder_dropIn'],
                        activation=model_params['decoder_activation']).to(device)
    # decoder_2.load_state_dict(pre_decoder_2.state_dict())
    encoder_1 = SimpleEncoder(gene_size_human, model_params['encoder_1_hiddens'], nComps1,
                              dropRate=model_params['dropout_encoder'],
                              activation=model_params['encoder_activation']).to(device)
    # encoder_1.load_state_dict(pre_encoder_1.state_dict())
    encoder_2 = SimpleEncoder(gene_size_primates, model_params['encoder_2_hiddens'], nComps2,
                              dropRate=model_params['dropout_encoder'],
                              activation=model_params['encoder_activation']).to(device)
    # encoder_2.load_state_dict(pre_encoder_2.state_dict())
    prior_d = PriorDiscriminator(model_params['latent_dim1']).to(device)
    # prior_d.load_state_dict(pre_prior_d.state_dict())
    local_d = LocalDiscriminator(model_params['latent_dim1'], model_params['latent_dim1']).to(device)
    # local_d.load_state_dict(pre_local_d.state_dict())

    classifier = Classifier(in_channel=model_params['latent_dim1'],
                            hidden_layers=model_params['state_class_hidden'],
                            num_classes=model_params['no_states'],
                            drop_in=model_params['state_class_drop_in'],
                            drop=model_params['state_class_drop']).to(device)

    protection_classifier = Classifier(in_channel=model_params['latent_dim1'],
                            hidden_layers=model_params['protect_class_hidden'],
                            num_classes=model_params['protect_states'],
                            drop_in=model_params['protect_class_drop_in'],
                            drop=model_params['protect_class_drop']).to(device)
    # protection_classifier_global = Classifier(in_channel=model_params['latent_dim1'],
    #                                    hidden_layers=model_params['protect_class_hidden'],
    #                                    num_classes=model_params['protect_states'],
    #                                    drop_in=model_params['protect_class_drop_in'],
    #                                    drop=model_params['protect_class_drop']).to(device)
    # protection_classifier.out_linear.bias.data.fill_(0.9282193) # ln(pos/neg)

    # classifier.load_state_dict(pre_classifier.state_dict())
    species_classifier = Classifier(in_channel=model_params['latent_dim1'],
                            hidden_layers=model_params['species_class_hidden'],
                            num_classes=model_params['no_species'],
                            drop_in=model_params['species_class_drop_in'],
                            drop=model_params['species_class_drop']).to(device)
    # species_classifier.load_state_dict(pre_species_classifier.state_dict())
    pretrained_adv_class = torch.load('../results_intermediate_encoders/pretrained_models/pre_trained_classifier_adverse_%s.pt'%i)
    adverse_classifier = Classifier(in_channel=model_params['latent_dim1'],
                                    hidden_layers=model_params['adv_class_hidden'],
                                    num_classes=model_params['no_adv_class'],
                                    drop_in=model_params['adv_class_drop_in'],
                                    drop=model_params['adv_class_drop']).to(device)
    adverse_classifier.load_state_dict(pretrained_adv_class.state_dict())

    # Vsp = SpeciesCovariate(2, model_params['latent_dim1'], dropRate=model_params['V_dropout']).to(device)
    # pretrained_Vsp = torch.load('../results_intermediate_encoders/pretrained_models/pre_trained_Vsp_%s.pt'%i)
    # pretrained_Vsp = torch.load('../results_intermediate_encoders/models/pre_Vspecies_%s.pt' % i)
    # Vsp.load_state_dict(pretrained_Vsp.state_dict())

    encoder_interm_1 = SimpleEncoder(model_params['latent_dim1'],
                                     model_params['intermediateEncoder1'],
                                     model_params['intermediate_latent1'],
                                     dropRate=model_params['intermediate_dropout'],dropIn = model_params['inter_dropIn'],
                                     activation=model_params['encoder_activation']).to(device)
    #encoder_interm_1.load_state_dict(pre_encoder_interm_1.state_dict())
    encoder_interm_2 = SimpleEncoder(model_params['latent_dim2'],
                                     model_params['intermediateEncoder2'],
                                     model_params['intermediate_latent2'],
                                     dropRate=model_params['intermediate_dropout'],dropIn = model_params['inter_dropIn'],
                                     activation=model_params['encoder_activation']).to(device)
    #encoder_interm_2.load_state_dict(pre_encoder_interm_2.state_dict())

    allParams = list(encoder_1.parameters()) + list(encoder_2.parameters())
    allParams = allParams + list(decoder_1.parameters()) + list(decoder_2.parameters())
    allParams = allParams + list(prior_d.parameters()) + list(local_d.parameters())
    allParams = allParams + list(classifier.parameters()) + list(protection_classifier.parameters())
    allParams = allParams + list(species_classifier.parameters())
    # allParams = allParams + list(Vsp.parameters())
    allParams = allParams + list(encoder_interm_1.parameters()) + list(encoder_interm_2.parameters())
    optimizer = torch.optim.Adam(allParams, lr=model_params['encoding_lr'], weight_decay=0)
    optimizer_adv = torch.optim.Adam(adverse_classifier.parameters(), lr=model_params['adv_lr'], weight_decay=0)
    if model_params['schedule_step_adv'] is not None:
        scheduler_adv = torch.optim.lr_scheduler.StepLR(optimizer_adv,
                                                        step_size=model_params['schedule_step_adv'],
                                                        gamma=model_params['gamma_adv'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=model_params['schedule_step_enc'],
                                                gamma=model_params['gamma_enc'])

    mLoss = torch.mean(torch.sum((xtrain_human - torch.mean(xtrain_human,0)*torch.ones(xtrain_human.shape))**2, dim=1))
    mLoss = mLoss + torch.mean(torch.sum((xtrain_primates - torch.mean(xtrain_primates,0)*torch.ones(xtrain_primates.shape))**2, dim=1))

    # trainLossMEAN = []
    # trainLossSTD = []
    for e in range(0, NUM_EPOCHS):
        decoder_1.train()
        decoder_2.train()
        encoder_1.train()
        encoder_2.train()
        prior_d.train()
        local_d.train()
        classifier.train()
        adverse_classifier.train()
        species_classifier.train()
        # Vsp.train()
        encoder_interm_1.train()
        encoder_interm_2.train()
        protection_classifier.train()

        trainloader_1 = getSamples(N_1, bs_1)
        len_1 = len(trainloader_1)
        trainloader_2 = getSamples(N_2, bs_2)
        len_2 = len(trainloader_2)

        lens = [len_1, len_2]
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

        # trainLoss = []
        for j in range(maxLen):
            dataIndex_1 = trainloader_1[j]
            dataIndex_2 = trainloader_2[j]

            X_2 = xtrain_primates[dataIndex_2, :].float().to(device)
            X2_transformed = torch.tensor(pca_space_2.transform(xtrain_primates[dataIndex_2, :].numpy())).float().to(
                device)
            # z_species_2 = torch.cat((torch.zeros(X_2.shape[0], 1),
            #                          torch.ones(X_2.shape[0], 1)), 1).to(device)
            X_1 = xtrain_human[dataIndex_1, :].float().to(device)
            X1_transformed = torch.tensor(pca_space_1.transform(xtrain_human[dataIndex_1, :].numpy())).float().to(
                device)
            # z_species_1 = torch.cat((torch.ones(X_1.shape[0], 1),
            #                          torch.zeros(X_1.shape[0], 1)), 1).to(device)

            # conditions = np.concatenate((ytrain_human[dataIndex_1, 0], ytrain_primates[dataIndex_2, 0]))
            # size = conditions.size
            # conditions = conditions.reshape(size, 1)
            # conditions = conditions == conditions.transpose()
            # conditions = conditions * 1

            conditions_protect = np.concatenate((ytrain_human[dataIndex_1, 1], ytrain_primates[dataIndex_2, 1]))
            size = conditions_protect.size
            conditions_protect = conditions_protect.reshape(size, 1)
            conditions_protect = conditions_protect == conditions_protect.transpose()
            conditions_protect = conditions_protect * 1

            # conditions = np.multiply(conditions_protect,conditions)
            conditions = np.copy(conditions_protect)

            mask = torch.tensor(conditions).to(device).detach()
            pos_mask = mask
            neg_mask = 1 - mask
            log_2 = math.log(2.)
            optimizer_adv.zero_grad()
            optimizer.zero_grad()

            # if e % model_params['adversary_steps']==0 or e % model_params['adversary_steps']+1==0 or e % model_params['adversary_steps']+2==0:
            z_base_1 = encoder_1(X_1)
            z_base_2 = encoder_2(X_2)
            latent_base_vectors = torch.cat((z_base_1, z_base_2), 0)
            labels_adv = adverse_classifier(latent_base_vectors)
            true_labels = torch.cat((torch.ones(z_base_1.shape[0]),
                                     torch.zeros(z_base_2.shape[0])), 0).long().to(device)
            _, predicted = torch.max(labels_adv, 1)
            predicted = predicted.cpu().numpy()
            cf_matrix = confusion_matrix(true_labels.cpu().numpy(), predicted, labels=[0, 1])
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
            # z_1 = encoder_1(X_1)
            # z_2 = encoder_2(X_2)
            latent_base_vectors = torch.cat((z_base_1, z_base_2), 0)
            # latent_vectors = torch.cat((z_1, z_2), 0)

            # z_un = local_d(latent_vectors)
            z_un = local_d(latent_base_vectors)
            res_un = torch.matmul(z_un, z_un.t())

            z_1 = encoder_interm_1(z_base_1)
            z_2 = encoder_interm_2(z_base_2)
            # z_1 = Vsp(z_base_1, z_species_1)
            # z_2 = Vsp(z_base_2, z_species_2)
            latent_vectors = torch.cat((z_1, z_2), 0)

            # y_pred_1 = decoder_1(z_1)
            y_pred_1 = decoder_1(z_1)
            fitLoss_1 = torch.mean(torch.sum((y_pred_1 - X_1) ** 2, dim=1))
            L2Loss_1 = encoder_1.L2Regularization(model_params['enc_l2_reg']) + decoder_1.L2Regularization(
                model_params['dec_l2_reg'])
            loss_1 = fitLoss_1 + L2Loss_1

            # y_pred_2 = decoder_2(z_2)
            y_pred_2 = decoder_2(z_2)
            fitLoss_2 = torch.mean(torch.sum((y_pred_2 - X_2) ** 2, dim=1))
            L2Loss_2 = encoder_2.L2Regularization(model_params['enc_l2_reg']) + decoder_2.L2Regularization(
                model_params['dec_l2_reg'])
            loss_2 = fitLoss_2 + L2Loss_2

            silimalityLoss = torch.sum(
                torch.cdist(latent_base_vectors, latent_base_vectors) * mask.float()) / mask.float().sum()
            w1 = latent_base_vectors.norm(p=2, dim=1, keepdim=True)
            w2 = latent_base_vectors.norm(p=2, dim=1, keepdim=True)
            cosineLoss = torch.mm(latent_base_vectors, latent_base_vectors.t()) / (w1 * w2.t()).clamp(min=1e-6)
            cosineLoss = torch.sum(cosineLoss * mask.float()) / mask.float().sum()

            p_samples = res_un * pos_mask.float()
            q_samples = res_un * neg_mask.float()

            Ep = log_2 - F.softplus(- p_samples)
            Eq = F.softplus(-q_samples) + q_samples - log_2

            Ep = (Ep * pos_mask.float()).sum() / pos_mask.float().sum()
            Eq = (Eq * neg_mask.float()).sum() / neg_mask.float().sum()
            mi_loss = Eq - Ep

            # prior = torch.rand_like(latent_vectors)
            prior = torch.rand_like(latent_base_vectors)

            term_a = torch.log(prior_d(prior)).mean()
            term_b = torch.log(1.0 - prior_d(latent_base_vectors)).mean()
            # term_b = torch.log(1.0 - prior_d(latent_vectors)).mean()
            prior_loss = -(term_a + term_b) * model_params['prior_beta']

            # Classification loss
            labels = classifier(latent_vectors)
            true_labels = torch.cat((ytrain_human[dataIndex_1, 0],
                                     ytrain_primates[dataIndex_2, 0]), 0).long().to(device)
            entropy = class_criterion(labels, true_labels)
            _, predicted = torch.max(labels, 1)
            predicted = predicted.cpu().numpy()
            cf_matrix = confusion_matrix(true_labels.cpu().numpy(), predicted, labels=[0, 1])
            tn, fp, fn, tp = cf_matrix.ravel()
            f1_latent = 2 * tp / (2 * tp + fp + fn)
            # Classification in basal also
            labels = classifier(latent_base_vectors)
            true_labels = torch.cat((ytrain_human[dataIndex_1, 0],
                                     ytrain_primates[dataIndex_2, 0]), 0).long().to(device)
            entropy_global = class_criterion(labels, true_labels)
            _, predicted = torch.max(labels, 1)
            predicted = predicted.cpu().numpy()
            cf_matrix = confusion_matrix(true_labels.cpu().numpy(), predicted, labels=[0, 1])
            tn, fp, fn, tp = cf_matrix.ravel()
            f1_latent_global = 2 * tp / (2 * tp + fp + fn)

            # Protection classification
            labels = protection_classifier(latent_base_vectors)
            true_labels = torch.cat((ytrain_human[dataIndex_1, 1],
                                     ytrain_primates[dataIndex_2, 1]), 0).long().to(device)
            protection_entropy = class_criterion(labels, true_labels)
            _, predicted = torch.max(labels, 1)
            predicted = predicted.cpu().numpy()
            cf_matrix = confusion_matrix(true_labels.cpu().numpy(), predicted, labels=[0, 1])
            tn, fp, fn, tp = cf_matrix.ravel()
            f1_protection = 2 * tp / (2 * tp + fp + fn)
            sens = tp / (tp + fn)
            spec = tn / (tn + fp)
            acc = (tp + tn) / labels.shape[0]
            # Protection in latent
            labels = protection_classifier(latent_vectors)
            true_labels = torch.cat((ytrain_human[dataIndex_1, 1],
                                     ytrain_primates[dataIndex_2, 1]), 0).long().to(device)
            protection_entropy_latent = class_criterion(labels, true_labels)
            _, predicted = torch.max(labels, 1)
            predicted = predicted.cpu().numpy()
            cf_matrix = confusion_matrix(true_labels.cpu().numpy(), predicted, labels=[0, 1])
            tn, fp, fn, tp = cf_matrix.ravel()
            f1_protection_latent = 2 * tp / (2 * tp + fp + fn)
            no_protected = torch.sum(true_labels) / true_labels.shape[0]
            # no_protected = true_labels

            # Species classification loss
            labels = species_classifier(latent_vectors)
            true_labels = torch.cat((torch.ones(z_1.shape[0]),
                                     torch.zeros(z_2.shape[0])), 0).long().to(device)
            entropy_species = class_criterion(labels, true_labels)
            _, predicted = torch.max(labels, 1)
            predicted = predicted.cpu().numpy()
            cf_matrix = confusion_matrix(true_labels.cpu().numpy(), predicted, labels=[0, 1])
            tn, fp, fn, tp = cf_matrix.ravel()
            f1_species = 2 * tp / (2 * tp + fp + fn)

            # Remove signal from z_basal
            labels_adv = adverse_classifier(latent_base_vectors)
            true_labels = torch.cat((torch.ones(z_base_1.shape[0]),
                                     torch.zeros(z_base_2.shape[0])), 0).long().to(device)
            adv_entropy = class_criterion(labels_adv, true_labels)
            _, predicted = torch.max(labels_adv, 1)
            predicted = predicted.cpu().numpy()
            cf_matrix = confusion_matrix(true_labels.cpu().numpy(), predicted, labels=[0, 1])
            tn, fp, fn, tp = cf_matrix.ravel()
            f1_basal = 2 * tp / (2 * tp + fp + fn)


            # loss = loss_1 + loss_2 + model_params['similarity_reg'] * silimalityLoss + model_params['lambda_mi_loss'] * mi_loss + prior_loss - model_params['cosine_loss'] * cosineLoss + Vsp.Regularization(model_params["v_reg"])
            loss = loss_1 + loss_2 + model_params['similarity_reg'] * silimalityLoss + model_params['lambda_mi_loss'] * mi_loss + prior_loss - model_params[
                'cosine_loss'] * cosineLoss + model_params['reg_classifier'] * (entropy_global) + model_params['reg_classifier'] * entropy_species + model_params['reg_classifier'] *(protection_entropy) - model_params[
                        'reg_adv'] * adv_entropy + classifier.L2Regularization(
                model_params['state_class_reg']) + species_classifier.L2Regularization(
                model_params['species_class_reg'])+ encoder_interm_1.L2Regularization(
                model_params['intermediate_enc_l2_reg'])+ encoder_interm_2.L2Regularization(model_params['intermediate_enc_l2_reg'])

            loss.backward()
            optimizer.step()


            pearson_1 = pearson_r(y_pred_1.detach(), X_1.detach())
            r2_1 = r_square(y_pred_1.detach(), X_1.detach())
            mse_1 = torch.mean(torch.mean((y_pred_1.detach() - X_1.detach()) ** 2, dim=1))

            pearson_2 = pearson_r(y_pred_2.detach(), X_2.detach())
            r2_2 = r_square(y_pred_2.detach(), X_2.detach())
            mse_2 = torch.mean(torch.mean((y_pred_2.detach() - X_2.detach()) ** 2, dim=1))

        # trainLoss.append(loss.item())

        # iteration += iteration

        if model_params['schedule_step_adv'] is not None:
            scheduler_adv.step()
        if (e >= 0):
            scheduler.step()
            outString = 'Split {:.0f}: Epoch={:.0f}/{:.0f}'.format(i + 1, e + 1, NUM_EPOCHS)
            outString += ', r2_1={:.4f}'.format(torch.mean(r2_1).item())
            outString += ', pearson_1={:.4f}'.format(torch.mean(pearson_1).item())
            outString += ', MSE_1={:.4f}'.format(mse_1.item())
            outString += ', r2_2={:.4f}'.format(torch.mean(r2_2).item())
            outString += ', pearson_2={:.4f}'.format(torch.mean(pearson_2).item())
            outString += ', MSE_2={:.4f}'.format(mse_2.item())
            outString += ', MI Loss={:.4f}'.format(mi_loss.item())
            outString += ', Prior Loss={:.4f}'.format(prior_loss.item())
            outString += ', Entropy Loss Global={:.4f}'.format(entropy_global.item())
            outString += ', Entropy_species={:.4f}'.format(entropy_species.item())
            outString += ', Entropy_protection={:.4f}'.format(protection_entropy.item())
            outString += ', Adverse Entropy={:.4f}'.format(adv_entropy.item())
            outString += ', Cosine Loss={:.4f}'.format(cosineLoss.item())
            outString += ', loss={:.4f}'.format(loss.item())
            outString += ', F1 latent={:.4f}'.format(f1_latent)
            outString += ', F1 latent Global={:.4f}'.format(f1_latent_global)
            outString +=', F1 Protection={:.4f}'.format(f1_protection)
            outString += ', Sensitivity={:.4f}'.format(sens)
            outString += ', Specificity={:.4f}'.format(spec)
            outString += ', Accuracy protect={:.4f}'.format(acc)
            outString += ', F1 Protection latent={:.4f}'.format(f1_protection_latent)
            outString += ', F1 basal={:.4f}'.format(f1_basal)
            # if e % model_params["adversary_steps"] == 0 and e>0:
            outString += ', F1 basal trained={:.4f}'.format(f1_basal_trained)
            outString += ', no_protected={:.2f}'.format(no_protected.item())
            # else:
            #    outString += ', F1 basal trained= %s'%f1_basal_trained
        # trainLossMEAN.append(np.mean(trainLoss))
        # trainLossSTD.append(np.std(trainLoss))
        if (e % 200 == 0):
            print2log(outString)
            # print2log(no_protected)
    print2log(outString)
    # T = np.array(range(len(trainLossMEAN)))
    # plt.plot(T, trainLossMEAN)
    # plt.xlim([0, len(T)])
    # curColor = plt.gca().lines[-1].get_color()
    # plt.fill_between(T,
    #                  np.array(trainLossMEAN) - np.array(trainLossSTD),
    #                  np.array(trainLossMEAN) + np.array(trainLossSTD),
    #                  color=curColor, alpha=0.2)
    # plt.plot([0, len(T)], np.array([1, 1]) * mLoss.item(), 'black', linestyle='--')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.yscale('log')
    # plt.show()
    decoder_1.eval()
    decoder_2.eval()
    encoder_1.eval()
    encoder_2.eval()
    prior_d.eval()
    local_d.eval()
    classifier.eval()
    species_classifier.eval()
    adverse_classifier.eval()
    # Vsp.eval()
    encoder_interm_1.eval()
    encoder_interm_2.eval()
    protection_classifier.eval()

    x_1 = xtest_human.float().to(device)
    x_2 = xtest_primates.float().to(device)
    x1_transformed = torch.tensor(pca_space_1.transform(xtest_human.numpy())).float().to(device)
    x2_transformed = torch.tensor(pca_space_2.transform(xtest_primates.numpy())).float().to(device)

    # z_species_1 = torch.cat((torch.ones(x_1.shape[0], 1),
    #                          torch.zeros(x_1.shape[0], 1)), 1).to(device)
    # z_species_2 = torch.cat((torch.zeros(x_2.shape[0], 1),
    #                          torch.ones(x_2.shape[0], 1)), 1).to(device)

    # z_latent_1 = encoder_1(x_1)
    # z_latent_2 = encoder_2(x_2)

    z_latent_base_1 = encoder_1(x_1)
    z_latent_base_2 = encoder_2(x_2)

    z_latent_1 = encoder_interm_1(z_latent_base_1)
    z_latent_2 = encoder_interm_2(z_latent_base_2)
    # z_latent_1 = Vsp(z_latent_base_1, z_species_1)
    # z_latent_2 = Vsp(z_latent_base_2, z_species_2)

    labels = classifier(torch.cat((z_latent_1, z_latent_2), 0))
    true_labels = torch.cat((ytest_human[:, 0],
                             ytest_primates[:, 0]), 0).long()
    _, predicted = torch.max(labels, 1)
    predicted = predicted.cpu().numpy()
    cf_matrix = confusion_matrix(true_labels.numpy(), predicted, labels=[0, 1])
    tn, fp, fn, tp = cf_matrix.ravel()
    class_acc = (tp + tn) / predicted.size
    f1 = 2 * tp / (2 * tp + fp + fn)
    # Global classification
    labels = classifier(torch.cat((z_latent_base_1, z_latent_base_2), 0))
    true_labels = torch.cat((ytest_human[:, 0],
                             ytest_primates[:, 0]), 0).long()
    _, predicted = torch.max(labels, 1)
    predicted = predicted.cpu().numpy()
    cf_matrix = confusion_matrix(true_labels.numpy(), predicted, labels=[0, 1])
    tn, fp, fn, tp = cf_matrix.ravel()
    class_acc_global = (tp + tn) / predicted.size
    f1_global = 2 * tp / (2 * tp + fp + fn)

    # Protection classification
    labels = protection_classifier(torch.cat((z_latent_base_1, z_latent_base_2), 0))
    true_labels = torch.cat((ytest_human[:, 1],
                             ytest_primates[:, 1]), 0).long().to(device)
    _, predicted = torch.max(labels, 1)
    predicted = predicted.cpu().numpy()
    cf_matrix = confusion_matrix(true_labels.cpu().numpy(), predicted, labels=[0, 1])
    tn, fp, fn, tp = cf_matrix.ravel()
    f1_protection = 2 * tp / (2 * tp + fp + fn)
    acc_protection = (tp + tn) / labels.shape[0]
    # Protection classification also in latent
    labels = protection_classifier(torch.cat((z_latent_1, z_latent_2), 0))
    true_labels = torch.cat((ytest_human[:, 1],
                             ytest_primates[:, 1]), 0).long().to(device)
    _, predicted = torch.max(labels, 1)
    predicted = predicted.cpu().numpy()
    cf_matrix = confusion_matrix(true_labels.cpu().numpy(), predicted, labels=[0, 1])
    tn, fp, fn, tp = cf_matrix.ravel()
    f1_protection_latent = 2 * tp / (2 * tp + fp + fn)

    # Species classifier
    labels = species_classifier(torch.cat((z_latent_1, z_latent_2), 0))
    true_labels = torch.cat((torch.ones(z_latent_1.shape[0]).view(z_latent_1.shape[0], 1),
                             torch.zeros(z_latent_2.shape[0]).view(z_latent_2.shape[0], 1)), 0).long()
    _, predicted = torch.max(labels, 1)
    predicted = predicted.cpu().numpy()
    cf_matrix = confusion_matrix(true_labels.numpy(), predicted, labels=[0, 1])
    tn, fp, fn, tp = cf_matrix.ravel()
    species_acc = (tp + tn) / predicted.size
    species_f1 = 2 * tp / (2 * tp + fp + fn)

    valF1.append(f1)
    valClassAcc.append(class_acc)
    valF1Species.append(species_f1)
    valAccSpecies.append(species_acc)
    valF1Protection.append(f1_protection)
    valAccProtection.append(acc_protection)
    valF1Global.append(f1_global)
    valClassAccGlobal.append(class_acc_global)

    print2log('Classification accuracy: %s' % class_acc)
    print2log('Classification F1 score: %s' % f1)
    print2log('Classification species F1 score: %s' % species_f1)
    print2log('Classification Protection F1 score: %s' % f1_protection)
    print2log('Classification F1 score in global: %s' % f1_global)
    print2log('Classification Protection F1 score in latent: %s' % f1_protection_latent)

    xhat_1 = decoder_1(z_latent_1)
    xhat_2 = decoder_2(z_latent_2)
    # xhat_1 = pre_decoder_1(z_latent_1)
    # xhat_2 = pre_decoder_2(z_latent_2)

    r2_1 = r_square(xhat_1.detach(), x_1.detach())
    pearson_1 = pearson_r(xhat_1.detach(), x_1.detach())
    r2_2 = r_square(xhat_2.detach(), x_2.detach())
    pearson_2 = pearson_r(xhat_2.detach(), x_2.detach())

    pearson_1_latent =  pearson_r(z_latent_1.detach(), x1_transformed.detach())
    pearson_2_latent = pearson_r(z_latent_2.detach(), x2_transformed.detach())


    valPear_1.append(torch.mean(pearson_1).item())
    valPear_2.append(torch.mean(pearson_2).item())
    valR2_1.append(torch.mean(r2_1).item())
    valR2_2.append(torch.mean(r2_2).item())
    print('R^2 human: %s'%torch.mean(r2_1).item())
    print2log('Pearson correlation human: %s' % torch.mean(pearson_1).item())
    print('R^2 primates: %s'%torch.mean(r2_2).item())
    print2log('Pearson correlation primates: %s' % torch.mean(pearson_2).item())
    valPear_1_latent.append(torch.mean(pearson_1_latent).item())
    valPear_2_latent.append(torch.mean(pearson_2_latent).item())

    pear_matrix_primates[i, :] = pearson_2.detach().cpu().numpy()
    pear_matrix_human[i, :] = pearson_1.detach().cpu().numpy()
    # pear_matrix_primates_latent[i, :] = pearson_2_latent.detach().cpu().numpy()
    # pear_matrix_human_latent[i, :] = pearson_1_latent.detach().cpu().numpy()
    df_z1 = pd.DataFrame(z_latent_1.detach().cpu().numpy())
    df_z1.columns = ['Z' + str(d + 1) for d in range(latent_dim)]
    df_pc1 = pd.DataFrame(x1_transformed.detach().cpu().numpy())
    df_pc1.columns = ['PC' + str(d + 1) for d in range(latent_dim)]
    df_z2 = pd.DataFrame(z_latent_2.detach().cpu().numpy())
    df_z2.columns = ['Z' + str(d + 1) for d in range(latent_dim)]
    df_pc2 = pd.DataFrame(x2_transformed.detach().cpu().numpy())
    df_pc2.columns = ['PC' + str(d + 1) for d in range(latent_dim)]
    df1 = pd.concat([df_z1,df_pc1],axis=1)
    df2 = pd.concat([df_z2, df_pc2], axis=1)
    corr_matrix_1 = df1.corr()
    corr_matrix_1 = corr_matrix_1.loc[df_pc1.columns.values,df_z1.columns.values]
    corr_matrix_1 = corr_matrix_1.values
    corr_matrix_2 = df2.corr()
    corr_matrix_2 = corr_matrix_2.loc[df_pc2.columns.values, df_z2.columns.values]
    corr_matrix_2 = corr_matrix_2.values
    pear_matrix_primates_latent[i, :, :] = corr_matrix_2
    pear_matrix_human_latent[i, :, :] = corr_matrix_1

    # Translate to other species
    # z_species_train_1 = torch.cat((torch.ones(xtrain_human.shape[0], 1),
    #                                torch.zeros(xtrain_human.shape[0], 1)), 1).to(device)
    # z_species_train_2 = torch.cat((torch.zeros(xtrain_primates.shape[0], 1),
    #                                torch.ones(xtrain_primates.shape[0], 1)), 1).to(device)
    # z1_train = Vsp(encoder_1(xtrain_human.float().to(device)),z_species_train_1)
    # z2_train = Vsp(encoder_2(xtrain_primates.float().to(device)),z_species_train_2)
    z1_train = encoder_interm_1(encoder_1(xtrain_human.float().to(device)))
    z2_train = encoder_interm_2(encoder_2(xtrain_primates.float().to(device)))

    knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
    knn.fit(torch.cat((z1_train, z2_train), 0).detach().cpu().numpy(), np.concatenate((np.ones(z1_train.shape[0]),
                                                                                 np.zeros(z2_train.shape[0])), 0))

    # z1_translated = Vsp(z_latent_base_2, 1 - z_species_2)
    # z2_translated = Vsp(z_latent_base_1, 1 - z_species_1)
    z1_translated = encoder_interm_1(z_latent_base_2)
    z2_translated = encoder_interm_2(z_latent_base_1)
    y_pred_translated = knn.predict(torch.cat((z1_translated, z2_translated), 0).detach().cpu().numpy())
    cf_matrix = confusion_matrix(np.concatenate((np.ones(z1_translated.shape[0]), np.zeros(z2_translated.shape[0])), 0),
                                 y_pred_translated, labels=[0, 1])
    tn, fp, fn, tp = cf_matrix.ravel()
    acc_translation = (tp + tn) / y_pred_translated.size
    rec_translation = tp / (tp + fn)
    prec_translation = tp / (tp + fp)
    f1_translation = 2 * tp / (2 * tp + fp + fn)

    species_labels = species_classifier(torch.cat((z1_translated, z2_translated), 0))
    species_true_labels = torch.cat((torch.ones(z1_translated.shape[0]),
                                     torch.zeros(z2_translated.shape[0])), 0).long().to(device)
    _, species_predicted = torch.max(species_labels, 1)
    species_predicted = species_predicted.cpu().numpy()
    cf_matrix = confusion_matrix(species_true_labels.cpu(), species_predicted, labels=[0, 1])
    tn, fp, fn, tp = cf_matrix.ravel()
    species_acc_trans = (tp + tn) / predicted.size
    species_f1_trans = 2 * tp / (2 * tp + fp + fn)

    print2log('F1 translation classifier: %s'%species_f1_trans)
    print2log('F1 translation KNN: %s' % f1_translation)

    valF1KNNTrans.append(f1_translation)
    valF1ClassTrans.append(species_f1_trans)

    torch.save(decoder_1, '../results_intermediate_encoders/models/10fold/decoder_human_%s.pt' % i)
    torch.save(decoder_2, '../results_intermediate_encoders/models/10fold/decoder_primates_%s.pt' % i)
    torch.save(prior_d, '../results_intermediate_encoders/models/10fold/priorDiscr_%s.pt' % i)
    torch.save(local_d, '../results_intermediate_encoders/models/10fold/localDiscr_%s.pt' % i)
    torch.save(encoder_1, '../results_intermediate_encoders/models/10fold/encoder_human_%s.pt' % i)
    torch.save(encoder_2, '../results_intermediate_encoders/models/10fold/encoder_primates_%s.pt' % i)
    torch.save(classifier, '../results_intermediate_encoders/models/10fold/classifier_%s.pt' % i)
    torch.save(species_classifier, '../results_intermediate_encoders/models/10fold/species_classifier_%s.pt' % i)
    # torch.save(Vsp, '../results_intermediate_encoders/models/Vspecies_%s.pt' % i)
    torch.save(encoder_interm_1,'../results_intermediate_encoders/models/10fold/encoder_intermediate_human_%s.pt' % i)
    torch.save(encoder_interm_2,'../results_intermediate_encoders/models/10fold/encoder_intermediate_primates_%s.pt' % i)
    torch.save(adverse_classifier, '../results_intermediate_encoders/models/10fold/classifier_adverse_%s.pt' % i)
    torch.save(protection_classifier,'../results_intermediate_encoders/models/10fold/protection_classifier_%s.pt' % i)

df_result = pd.DataFrame({'F1':valF1,'Accuracy':valClassAcc,'F1Species':valF1Species,'AccuracySpecies':valAccSpecies,
                          'F1_global': valF1Global, 'Accuracy_global': valClassAccGlobal,
                          'F1Protection':valF1Protection,'AccProtection':valAccProtection,
                          'recon_pear_primates':valPear_2 ,'recon_pear_human':valPear_1,
                          'recon_r2_primates':valR2_2,'recon_r2_human':valR2_1,
                          'latent_human_pear':valPear_1_latent,'latent_mouse_pear':valPear_2_latent,
                          'KNNTranslationF1':valF1KNNTrans,'ClassifierTranslationF1':valF1ClassTrans})

df_result.to_csv('../results_intermediate_encoders/10foldvalidation_wholeModel_'+str(latent_dim)+'dim'+str(NUM_EPOCHS)+'ep_serology.csv')
# df_result = pd.read_csv('../results_intermediate_encoders/10foldvalidation_wholeModel_'+str(latent_dim)+'dim'+str(NUM_EPOCHS)+'ep_serology.csv',index_col=0)

pear_matrix_primates = pd.DataFrame(pear_matrix_primates)
pear_matrix_primates.columns = primates_exprs.columns
# pear_matrix_primates = pd.read_csv('../results_intermediate_encoders/10foldvalidation_decoder_'+str(latent_dim)+'dim'+str(NUM_EPOCHS)+'ep_perFeature_primates.csv',index_col=0)
mu = pear_matrix_primates.mean(1).mean()
se = pear_matrix_primates.mean(1).std()/np.sqrt(model_params['no_folds'])
pm_symbol = "\u00B1"
text = f"r = {mu:.2f} {pm_symbol} {se:.2f}"
pear_matrix_primates.to_csv('../results_intermediate_encoders/10foldvalidation_decoder_'+str(latent_dim)+'dim'+str(NUM_EPOCHS)+'ep_perFeature_primates.csv')
pear_matrix_primates = pd.melt(pear_matrix_primates)
pear_matrix_primates.columns = ['feature','pearson']
grouped = pear_matrix_primates.groupby(['feature']).median().sort_values(by='pearson',ascending=False)
sns.set_theme(style="whitegrid",font_scale=3)
plt.figure(figsize=(18,12), dpi= 80)
ax = sns.boxplot(x="pearson", y="feature", data=pear_matrix_primates,order=grouped.index,orient='h')
plt.legend(loc='lower left')
plt.gca().set(title='Per feature performance of primate decoder in cross-validation',
              xlabel = 'pearson correlation',
              ylabel='features')
plt.yticks([])
# for ind, label in enumerate(ax.get_yticklabels()):
#     if ind % 5 == 0:  # every 10th label is kept
#         label.set_visible(True)
#     else:
#         label.set_visible(False)
plt.xlim(-1,1)
ax.text(-0.97, 80,text, fontsize=42,fontweight='bold',bbox=dict(edgecolor='black',facecolor='none'))
plt.savefig('../results_intermediate_encoders/perFeature_performance_decoder_'+str(latent_dim)+'dim'+str(NUM_EPOCHS)+'ep_primates.png', bbox_inches='tight',dpi=600)
pear_matrix_human = pd.DataFrame(pear_matrix_human)
pear_matrix_human.columns = human_exprs.columns
# pear_matrix_human = pd.read_csv('../results_intermediate_encoders/10foldvalidation_decoder_'+str(latent_dim)+'dim'+str(NUM_EPOCHS)+'ep_perFeature_human.csv',index_col=0)
mu = pear_matrix_human.mean(1).mean()
se = pear_matrix_human.mean(1).std()/np.sqrt(model_params['no_folds'])
pm_symbol = "\u00B1"
text = f"r = {mu:.2f} {pm_symbol} {se:.2f}"
pear_matrix_human.to_csv('../results_intermediate_encoders/10foldvalidation_decoder_'+str(latent_dim)+'dim'+str(NUM_EPOCHS)+'ep_perFeature_human.csv')
pear_matrix_human = pd.melt(pear_matrix_human)
pear_matrix_human.columns = ['feature','pearson']
grouped = pear_matrix_human.groupby(['feature']).median().sort_values(by='pearson',ascending=False)
sns.set_theme(style="whitegrid",font_scale=3)
plt.figure(figsize=(18,12), dpi= 80)
ax = sns.boxplot(x="pearson", y="feature", data=pear_matrix_human,order=grouped.index,orient='h')
# ax.yaxis.tick_right()
plt.legend(loc='lower left')
plt.gca().set(title='Per feature performance of human decoder in cross-validation',
              xlabel = 'pearson correlation',
              ylabel='features')
plt.xlim(-1,1)
# ax.yaxis.set_tick_params(labelsize = 5)
plt.yticks([])
ax.text(-0.5, 20,text, fontsize=42,fontweight='bold',bbox=dict(edgecolor='black',facecolor='none'))
plt.savefig('../results_intermediate_encoders/perFeature_performance_decoder_'+str(latent_dim)+'dim'+str(NUM_EPOCHS)+'ep_human.png', bbox_inches='tight',dpi=600)


pear_matrix_primates_latent = pear_matrix_primates_latent.mean(0)
pear_matrix_human_latent = pear_matrix_human_latent.mean(0)

pear_matrix_primates_latent = pd.DataFrame(pear_matrix_primates_latent)
# pear_matrix_primates_latent.columns = ['PC'+str(d+1) for d in range(latent_dim)]
pear_matrix_primates_latent.columns = ['Z'+str(d+1) for d in range(latent_dim)]
pear_matrix_primates_latent.index = ['PC'+str(d+1) for d in range(latent_dim)]
pear_matrix_primates_latent.to_csv('../results_intermediate_encoders/10foldvalidation_encoders_'+str(latent_dim)+'dim'+str(NUM_EPOCHS)+'ep_perFeature_primates.csv')
# pear_matrix_primates_latent = pd.melt(pear_matrix_primates_latent)
# pear_matrix_primates_latent.columns = ['PC','pearson']
# sns.set_theme(style="whitegrid")
plt.figure(figsize=(9,12), dpi= 80)
# ax = sns.boxplot(x="pearson", y="PC", data=pear_matrix_primates_latent,orient='h') #order=grouped.index
# ax.axhline(n2,ls='--',color='red')
# plt.legend(loc='lower left')
# plt.gca().set(title='Per principal component performance of primate encoder in 10-fold cross-validation',
#               xlabel = 'pearson correlation',
#               ylabel='PC')
# #plt.xlim(0,1)
# plt.savefig('../results_intermediate_encoders/perFeature_encoder_'+str(latent_dim)+'dim'+str(NUM_EPOCHS)+'ep_primates.png', bbox_inches='tight',dpi=600)
sns.clustermap(pear_matrix_primates_latent,cmap='RdBu_r', center=0,vmin=-0.75, vmax=0.75)
plt.savefig('../results_intermediate_encoders/perFeature_PCA_avgCorrelation_'+str(latent_dim)+'dim'+str(NUM_EPOCHS)+'ep_primates.png', bbox_inches='tight',dpi=600)
pear_matrix_human_latent = pd.DataFrame(pear_matrix_human_latent)
# pear_matrix_human_latent.columns = ['PC'+str(d+1) for d in range(latent_dim)]
pear_matrix_human_latent.columns = ['Z'+str(d+1) for d in range(latent_dim)]
pear_matrix_human_latent.index = ['PC'+str(d+1) for d in range(latent_dim)]
pear_matrix_human_latent.to_csv('../results_intermediate_encoders/10foldvalidation_encoders_'+str(latent_dim)+'dim'+str(NUM_EPOCHS)+'ep_perFeature_human.csv')
# pear_matrix_human_latent = pd.melt(pear_matrix_human_latent)
# pear_matrix_human_latent.columns = ['PC','pearson']
# sns.set_theme(style="whitegrid")
plt.figure(figsize=(9,12), dpi= 80)
# ax = sns.boxplot(x="pearson", y="PC", data=pear_matrix_human_latent,orient='h') #order=grouped.index
# ax.axhline(n1,ls='--',color='red')
# plt.legend(loc='lower left')
# plt.gca().set(title='Per principal component performance of human encoder in 10-fold cross-validation',
#               xlabel = 'pearson correlation',
#               ylabel='PC')
# #plt.xlim(0,1)
# plt.savefig('../results_intermediate_encoders/perFeature_encoder_'+str(latent_dim)+'dim'+str(NUM_EPOCHS)+'ep_human.png', bbox_inches='tight',dpi=600)
sns.clustermap(pear_matrix_human_latent,cmap='RdBu_r', center=0,vmin=-0.75, vmax=0.75)
plt.savefig('../results_intermediate_encoders/perFeature_PCA_avgCorrelation_'+str(latent_dim)+'dim'+str(NUM_EPOCHS)+'ep_human.png', bbox_inches='tight',dpi=600)

