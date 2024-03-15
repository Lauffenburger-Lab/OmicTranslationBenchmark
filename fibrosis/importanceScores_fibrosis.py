import torch
import torch.nn.functional as F
from models import Encoder,Decoder,GaussianDecoder,VAE,CellStateEncoder,\
                   CellStateDecoder, CellStateVAE,\
                   SimpleEncoder,LocalDiscriminator,PriorDiscriminator,\
                   EmbInfomax,MultiEncInfomax,Classifier,\
                   SpeciesCovariate,GaussianDecoder,VarDecoder,ElementWiseLinear
import math
import numpy as np
import pandas as pd
from captum.attr import IntegratedGradients
from captum.attr import LayerConductance
from captum.attr import NeuronConductance
from trainingUtils import NBLoss,_convert_mean_disp_to_counts_logits,compute_kernel, compute_mmd
from torch.distributions import NegativeBinomial
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

#Load all human data and artificially inflate
X_human = torch.load('xtrain_human_lung_all.pt').float().cpu()
ytrain_human = torch.load('ytrain_human_lung_all.pt').float().cpu()
conds_human = ytrain_human[:, 1]
X_mouse = torch.load('xtrain_mouse_lung_all.pt').float().cpu()
ytrain_mouse = torch.load('ytrain_mouse_lung_all.pt').float().cpu()
conds_mouse = ytrain_mouse[:, 1]
human_genes = pd.read_csv('human_genes.csv')
mouse_genes = pd.read_csv('mouse_genes.csv')

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
#class FullEncoder(torch.nn.Module):
#    def __init__(self, encoder,Vcell,Vsp):
#        super(FullEncoder, self).__init__()
#
#        self.encoder = encoder
#        self.Vsp = Vsp
#        self.Vcell = Vcell
#
#    def forward(self, x,z_species,z_cell):
#        z = self.encoder(x)
#        z = self.Vsp(z,z_species)
#        z = self.Vcell(z,z_cell)
#        return z

def getSamples(N, batchSize):
    order = np.arange(N)
    outList = []
    while len(order)>0:
        outList.append(order[0:batchSize])
        order = order[batchSize:]
    return outList
gene_size_mouse = X_mouse.shape[1]
gene_size_human = X_human.shape[1]
# for fold in range(folds):
#     human_latent = pd.read_csv('human lung fibrosis/preds/embeddings/latent_human4translation_%s.csv' % fold,index_col=0)
#     z1 = torch.tensor(human_latent.iloc[:,0:512].values).to(device).float()
#     # Load trained models
#     classifier = torch.load('models/classifier_disease_%s.pth' % fold)
#     classifier.eval()
#
#     print2log('Start classification importance for model %s'%fold)
#     # Classifier importance
#     ig = IntegratedGradients(classifier)
#     # z_base_1.requires_grad_()
#     attr1 = ig.attribute(z1,target=1,n_steps=100, return_convergence_delta=False)
#     attr1 = attr1.detach().cpu().numpy()
#     attr2 = ig.attribute(z1,target=0,n_steps=100, return_convergence_delta=False)
#     attr2 = attr2.detach().cpu().numpy()
#     df1 = pd.DataFrame(attr1)
#     df1.columns = ['z'+str(i) for i in range(z1.shape[1])]
#     df1.to_csv('importance/important_scores_to_classify_human_fibrosis_%s.csv'%fold)
#     df2 = pd.DataFrame(attr2)
#     df2.columns = ['z'+str(i) for i in range(z1.shape[1])]
#     df2.to_csv('importance/important_scores_to_classify_human_nonfibrosis_%s.csv'%fold)
#
#     print2log('Finished model %s'%fold)

#### Find importance for encoding features per specific cell-type
# for fold in range(folds):
#     human_latent = pd.read_csv('human lung fibrosis/preds/embeddings/latent_human4translation_%s.csv' % fold,index_col=0)
#
#     # Load trained models
#     encoder_human = torch.load('models/encoder_human_%s.pth' % fold)
#     Vsp = torch.load('models/Vspecies_%s.pt' % fold)
#     Vcell = torch.load('models/Vcell_%s.pt' % fold)
#     encoder_human.eval()
#     Vcell.eval()
#     Vsp.eval()
#     full_encoder = FullEncoder(encoder_human,Vcell,Vsp)
#     full_encoder.eval()
#
#     for cell in cell_types:
#         inds = np.where(human_latent.specific_human_cell.values == cell)[0]
#         x = X_human[inds,:]#.to(device)
#
#         z_species_1 = torch.cat((torch.ones(x.shape[0], 1),
#                                  torch.zeros(x.shape[0], 1)), 1)#.to(device)
#         z_cell_1 = torch.cat((1 * (conds_human == 1).unsqueeze(1), 1 * (conds_human == 2).unsqueeze(1),
#                               1 * (conds_human == 3).unsqueeze(1), 1 * (conds_human == 4).unsqueeze(1),
#                               1 * (conds_human == 5).unsqueeze(1)), 1).float()
#         z_cell_1 = z_cell_1[inds,:]#.to(device)
#
#         print2log('Start human feature importance of %s for model %s'%(cell,fold))
#         # Per output latent variable input importance translation captum
#         # 1st dimesion input
#         # 2nd dimesion output
#         ig = IntegratedGradients(full_encoder)
#         hid_dim = 512
#         scores_human = torch.zeros((gene_size_human, hid_dim)).double()
#         loader = getSamples(x.shape[0], 256)
#         for z in range(hid_dim):
#             all_attr = torch.zeros((x.shape[0],gene_size_human)).double()
#             for samps in loader:
#                 attr, _, _ = ig.attribute((x[samps,:].to(device),z_species_1[samps,:].to(device),z_cell_1[samps,:].to(device)), target=z, n_steps=100, return_convergence_delta=False)
#                 all_attr[samps,:] = attr.detach().cpu()
#             scores_human[:, z] = torch.mean(all_attr, 0).detach().cpu()
#             if z % 50 == 0:
#                 print2log('Finished latent dimension %s'%z)
#         df_human = pd.DataFrame(scores_human.cpu().numpy())
#         df_human.columns = ['z' + str(i) for i in range(512)]
#         # df_human.index = human_exprs.columns
#         df_human.to_csv('importance/%s_important_genes_human_%s.csv'%(cell,fold))
#
#         print2log('Finished %s in model %s'%(cell,fold))

### Perform importance calculation for translation
interesting_feats = pd.read_csv('filtered_human_fibrosis_important_genes.csv')

# Define translator model
class TranslatorModel(torch.nn.Module):
    def __init__(self, encoder,Vs,Vc,decoder):
        super(TranslatorModel, self).__init__()
        self.encoder = encoder
        self.Vs = Vs
        self.Vc = Vc
        self.decoder = decoder
    def forward(self, x,z_species,z_cell):
        z = self.encoder(x)
        z = self.Vs(z,z_species)
        z = self.Vc(z, z_cell)
        mu,_ = self.decoder(z)
        #counts, logits = _convert_mean_disp_to_counts_logits(
        #    torch.clamp(
        #        mu,
        #        min=1e-4,
        #        max=1e4,
        #    ),
        #    torch.clamp(
        #        disp,
        #        min=1e-4,
        #        max=1e4,
        #    )
        #)
        #distr = NegativeBinomial(
        #    total_count=counts,
        #    logits=logits
        #)
        #nb_sample = distr.sample()
        return mu#nb_sample
for fold in range(folds):
    mouse_latent = pd.read_csv('human lung fibrosis/preds/embeddings/latent_mouse4translation_%s.csv' % fold,
                               index_col=0)
    human_latent = pd.read_csv('human lung fibrosis/preds/embeddings/latent_human4translation_%s.csv' % fold,
                               index_col=0)
    # Load trained models
    encoder_mouse = torch.load('models/encoder_mouse_%s.pth' % fold)
    Vsp = torch.load('models/Vspecies_%s.pt' % fold)
    Vcell = torch.load('models/Vcell_%s.pt' % fold)
    decoder_human = torch.load('models/decoder_human_%s.pth' % fold)
    encoder_mouse.eval()
    decoder_human.eval()
    Vcell.eval()
    Vsp.eval()
    translator = TranslatorModel(encoder_mouse,Vsp,Vcell,decoder_human)
    translator.eval()

    for cell in cell_types:
        inds = np.where(mouse_latent.specific_mouse_cell.values == cell)[0]
        inds_h = np.where(human_latent.specific_human_cell.values == cell)[0]
        human_genes_of_interest = interesting_feats[interesting_feats['cell_type']==cell]
        human_genes_of_interest = np.unique(human_genes_of_interest.feature)
        human_feat_inds = np.where(np.isin(human_genes.gene.values,human_genes_of_interest))[0]
        xh = X_human[inds_h,:]
        xh = xh[:,human_feat_inds]#.to(device)
        x = X_mouse[inds, :]  # .to(device)

        z_species_2 = torch.cat((torch.zeros(x.shape[0], 1),
                                 torch.ones(x.shape[0], 1)), 1)#.to(device)
        z_cell_2 = torch.cat((1 * (conds_mouse == 1).unsqueeze(1), 1 * (conds_mouse == 2).unsqueeze(1),
                              1 * (conds_mouse == 3).unsqueeze(1), 1 * (conds_mouse == 4).unsqueeze(1),
                              1 * (conds_mouse == 5).unsqueeze(1)), 1).float()
        z_cell_2 = z_cell_2[inds,:]#.to(device)

        print2log('Start mouse feature importance of %s for model %s'%(cell,fold))
        # Per output latent variable input importance translation captum
        # 1st dimesion input
        # 2nd dimesion output
        ig = IntegratedGradients(translator)
        hid_dim = human_feat_inds.shape[0]
        scores_mouse = torch.zeros((gene_size_mouse, hid_dim)).double()
        loader = getSamples(x.shape[0], 128)
        counter = 0
        for z in human_feat_inds:
            all_attr = torch.zeros((x.shape[0],gene_size_mouse)).double()
            for samps in loader:
                #y = translator(x[samps,:].to(device),z_species_2[samps,:].to(device),z_cell_2[samps,:].to(device))
                #print2log(y.shape)
                #print2log(z)
                #print2log(x[samps,:].shape)
                #print2log(z_species_2[samps,:].shape)
                #print2log(z_cell_2[samps,:].shape)
                #print2log((x[samps,:].to(device),z_species_2[samps,:].to(device),z_cell_2[samps,:].to(device)))
                attr, _, _ = ig.attribute((x[samps,:].to(device),z_species_2[samps,:].to(device),z_cell_2[samps,:].to(device)), target=int(z), n_steps=100, return_convergence_delta=False)
                all_attr[samps,:] = attr.detach().cpu()
            scores_mouse[:, counter] = torch.mean(all_attr, 0).detach().cpu()
            counter+=1
            #if z % 50 == 0:
            print2log('Finished latent dimension %s'%z)
        df_mouse = pd.DataFrame(scores_mouse.cpu().numpy())
        df_mouse.columns = human_genes_of_interest
        df_mouse.index = mouse_genes.gene.values
        df_mouse.to_csv('importance/%s_important_translational_genes_mouse_%s.csv'%(cell,fold))

        print2log('Finished %s in model %s'%(cell,fold))
