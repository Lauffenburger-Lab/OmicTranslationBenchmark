import pickle
import torch
import torch.nn.functional as F
from trainingUtils import NBLoss,_convert_mean_disp_to_counts_logits #,MultipleOptimizer, MultipleScheduler, compute_kernel, compute_mmd
from models import Encoder,Decoder,GaussianDecoder,VAE,CellStateEncoder,\
                   CellStateDecoder, CellStateVAE,\
                   SimpleEncoder,LocalDiscriminator,PriorDiscriminator,\
                   EmbInfomax,MultiEncInfomax,Classifier,\
                   SpeciesCovariate,GaussianDecoder,VarDecoder,ElementWiseLinear
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,cross_validate,KFold
from sklearn.metrics import confusion_matrix,f1_score,r2_score
from torch.distributions import NegativeBinomial
from captum.attr import IntegratedGradients
from captum.attr import LayerConductance
from captum.attr import NeuronConductance

import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
print2log = logger.info

# initialize
folds = 10
max_iter = 1000
cell_types = ['Macrophage','Plasma cell','T cell','AT2','Fibroblast','B cell',
              'Myofibroblast','AT1','NK cell']
device = torch.device('cuda')

# Define translator model
# class TranslatorModel(torch.nn.Module):
#     def __init__(self, encoder,Vcov,decoder):
#         super(TranslatorModel, self).__init__()
#
#         self.encoder = encoder
#         self.Vcov = Vcov
#         self.decoder = decoder
#
#     def forward(self, x,z_species):
#         z = self.encoder(x)
#         z = self.Vcov(z,z_species)
#         y = self.decoder(z)
#         return y
class FullEncoder(torch.nn.Module):
    def __init__(self, encoder,Vcell,Vsp):
        super(TranslatorModel, self).__init__()

        self.encoder = encoder
        self.Vsp = Vsp
        self.Vcell = Vcell

    def forward(self, x,z_species,z_cell):
        z = self.encoder(x)
        z = self.Vsp(z,z_species)
        z = self.Vcell(z,z_cell)
        return z

for fold in range(folds):
    # xtrain_mouse = torch.load('10foldcrossval_lung/xtrain_mouse_%s.pt' % fold)
    # ytrain_mouse = torch.load('10foldcrossval_lung/ytrain_mouse_%s.pt' % fold)

    xtrain_human = torch.load('10foldcrossval_lung/xtrain_human_%s.pt' % fold)
    ytrain_human = torch.load('10foldcrossval_lung/ytrain_human_%s.pt' % fold)

    # gene_size_mouse = xtrain_mouse.shape[1]
    gene_size_human = xtrain_human.shape[1]

    # Load trained models
    encoder_human = torch.load('models/encoder_human_%s.pth' % fold)
    # encoder_mouse = torch.load('models/encoder_mouse_%s.pth' % fold)
    # decoder_human = torch.load('models/decoder_human_%s.pth' % fold)
    # decoder_mouse = torch.load('models/decoder_mouse_%s.pth' % fold)
    Vsp = torch.load('models/Vspecies_%s.pt' % fold)
    Vcell = torch.load('models/Vcell_%s.pt' % fold)
    classifier = torch.load('models/classifier_disease_%s.pth' % fold)

    encoder_human.eval()
    # encoder_mouse.eval()
    # decoder_human.eval()
    # decoder_mouse.eval()
    Vcell.eval()
    Vsp.eval()
    classifier.eval()

    full_encoder = FullEncoder(encoder_human,Vcell,Vsp)
    full_encoder.eval()

    Y_human = ytrain_human[:, 0]
    # Y_mouse = ytrain_mouse[:, 0]
    X_human = xtrain_human.to(device)
    # X_mouse = xtrain_mouse.to(device)
    conds_human = ytrain_human[:, 2]
    # conds_mouse = ytrain_mouse[:, 2]
    z_species_1 = torch.cat((torch.ones(X_human.shape[0], 1),
                             torch.zeros(X_human.shape[0], 1)), 1).to(device)
    z_species_2 = torch.cat((torch.zeros(X_mouse.shape[0], 1),
                             torch.ones(X_mouse.shape[0], 1)), 1).to(device)
    z_cell_1 = torch.cat((1 * (conds_human == 1).unsqueeze(1), 1 * (conds_human == 2).unsqueeze(1),
                          1 * (conds_human == 3).unsqueeze(1), 1 * (conds_human == 4).unsqueeze(1),
                          1 * (conds_human == 5).unsqueeze(1)), 1).float().to(device)
    z_base_1 = encoder_human(X_human).detach()
    z1 = Vsp(z_base_1, z_species_1).detach()
    z1 = Vcell(z1, z_cell_1).detach()

    print2log('Start classification importance for model %s'%i)
    # Classifier importance
    ig = IntegratedGradients(classifier)
    # z_base_1.requires_grad_()
    attr1 = ig.attribute(z1,target=1,n_steps=1000, return_convergence_delta=False)
    attr1 = attr1.detach().cpu().numpy()
    attr2 = ig.attribute(z1,target=0,n_steps=1000, return_convergence_delta=False)
    attr2 = attr2.detach().cpu().numpy()
    df1 = pd.DataFrame(attr1)
    df1.columns = ['z'+str(i) for i in range(z1.shape[1])]
    df1.to_csv('importance/important_scores_to_classify_human_protection_%s.csv'%i)
    df2 = pd.DataFrame(attr2)
    df2.columns = ['z'+str(i) for i in range(z1.shape[1])]
    df2.to_csv('importance/important_scores_to_classify_human_nonprotection_%s.csv'%i)

    print2log('Finished model %s' % i)

#### Find importance for encoding features per specific cell-type
for fold in range(folds):
    # xtrain_mouse = torch.load('10foldcrossval_lung/xtrain_mouse_%s.pt' % fold)
    # ytrain_mouse = torch.load('10foldcrossval_lung/ytrain_mouse_%s.pt' % fold)

    xtrain_human = torch.load('10foldcrossval_lung/xtrain_human_%s.pt' % fold)
    ytrain_human = torch.load('10foldcrossval_lung/ytrain_human_%s.pt' % fold)

    # gene_size_mouse = xtrain_mouse.shape[1]
    gene_size_human = xtrain_human.shape[1]

    # Load trained models
    encoder_human = torch.load('models/encoder_human_%s.pth' % fold)
    # encoder_mouse = torch.load('models/encoder_mouse_%s.pth' % fold)
    # decoder_human = torch.load('models/decoder_human_%s.pth' % fold)
    # decoder_mouse = torch.load('models/decoder_mouse_%s.pth' % fold)
    Vsp = torch.load('models/Vspecies_%s.pt' % fold)
    Vcell = torch.load('models/Vcell_%s.pt' % fold)
    classifier = torch.load('models/classifier_disease_%s.pth' % fold)

    encoder_human.eval()
    # encoder_mouse.eval()
    # decoder_human.eval()
    # decoder_mouse.eval()
    Vcell.eval()
    Vsp.eval()
    classifier.eval()

    full_encoder = FullEncoder(encoder_human,Vcell,Vsp)
    full_encoder.eval()
    print2log('Start human feature importance for model %s'%i)
    # Per output latent variable input importance translation captum
    # 1st dimesion input
    # 2nd dimesion output
    ig = IntegratedGradients(full_encoder)
    hid_dim = z1.shape[1]
    scores_human = torch.zeros((gene_size_human, hid_dim)).to(device)
    for z in range(hid_dim):
        # encoder_1.zero_grad()
        attr = ig.attribute(xtrain_human, target=z, n_steps=1000, return_convergence_delta=False)
        scores_human[:, z] = torch.mean(attr, 0)
        # if z % 2 == 0 :
        #     print2log(z)
    df_human = pd.DataFrame(scores_human.cpu().numpy())
    df_human.columns = ['z' + str(i) for i in range(model_params['latent_dim1'])]
    df_human.index = human_exprs.columns
    df_human.to_csv('importance/important_scores_human_features_%s.csv'%i)

    print2log('Finished cell type %s in model %s'%(cell,i))

# ### Perform importance calculation for translation
# ### Load features of interest
# interesting_feats = pd.read_csv('../importance_results_cpa/interesting_features.csv',index_col=0)
#
# # Define translator model
# class TranslatorModel(torch.nn.Module):
#     def __init__(self, encoder,Vcov,decoder):
#         super(TranslatorModel, self).__init__()
#
#         self.encoder = encoder
#         self.Vcov = Vcov
#         self.decoder = decoder
#
#     def forward(self, x,z_species):
#         z = self.encoder(x)
#         z = self.Vcov(z,z_species)
#         y = self.decoder(z)
#         return y
#
# print2log('Begin finding features to translate from primates to human')
# for i in range(model_params["no_folds"]):
#     gene_size_primates = xtrain_primates.shape[1]
#     gene_size_human = xtrain_human.shape[1]
#
#     decoder_1 = torch.load( '../results/models/10fold/decoder_human_%s.pt' % i)
#     # decoder_2 = torch.load( '../results/models/10fold/decoder_primates_%s.pt' % i)
#     # encoder_1 = torch.load('../results/models/10fold/encoder_human_%s.pt' % i)
#     encoder_2 = torch.load('../results/models/10fold/encoder_primates_%s.pt' % i)
#     Vsp = torch.load('../results/models/10fold/Vspecies_%s.pt' % i).to(device)
#
#     translator =TranslatorModel(encoder_2,Vsp,decoder_1).to(device)
#
#     # encoder_1.eval()
#     encoder_2.eval()
#     decoder_1.eval()
#     Vsp.eval()
#     # decoder_2.eval()
#     translator.eval()
#
#     # Per output latent variable input importance translation captum
#     # 1st dimesion input
#     # 2nd dimesion output
#     ig = IntegratedGradients(translator)
#     scores_primates = torch.zeros((gene_size_primates, len(interesting_human_feats_inds))).to(device)
#     ii = 0
#     for feat in interesting_human_feats_inds:
#         # encoder_1.zero_grad()
#         attr, _ = ig.attribute((xtrain_primates,z_species_1), target=feat, n_steps=2000, return_convergence_delta=False)
#         scores_primates[:, ii] = torch.mean(attr, 0)
#         ii = ii + 1
#         # if z % 2 == 0 :
#         #     print2log(z)
#     df_primates = pd.DataFrame(scores_primates.cpu().numpy())
#     df_primates.columns = interesting_feats[interesting_feats['species']=='human'].feature
#     df_primates.index = primates_exprs.columns
#     df_primates.to_csv('importance/important_scores_mouse_to_human_%s.csv' % i)
#     print2log('Finished model %s' % i)