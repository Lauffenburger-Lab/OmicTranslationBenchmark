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
import logging
from captum.attr import IntegratedGradients
from captum.attr import LayerConductance
from captum.attr import NeuronConductance


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
print2log = logger.info

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
cmap = pd.read_csv('cmap_all_genes_q1_tas03.csv',index_col = 0)

gene_size = len(cmap.columns)
samples = cmap.index.values

sampleInfo_1 = pd.read_csv('10fold_validation_spit/alldata/pc3_unpaired.csv',index_col=0)
sampleInfo_2 = pd.read_csv('10fold_validation_spit/alldata/ha1e_unpaired.csv',index_col=0)
sampleInfo_paired = pd.read_csv('10fold_validation_spit/alldata/paired_pc3_ha1e.csv',index_col=0)


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

# Create a train generators
def getSamples(N, batchSize):
    order = np.random.permutation(N)
    outList = []
    while len(order)>0:
        outList.append(order[0:batchSize])
        order = order[batchSize:]
    return outList

def compute_gradients(output, input):
    grads = torch.autograd.grad(output, input, create_graph=True)
    grads = grads[0].pow(2).mean()
    return grads


class_criterion = torch.nn.CrossEntropyLoss()
NUM_EPOCHS= model_params['epochs']
bs_1 = model_params['batch_size_1']
bs_2 =  model_params['batch_size_2']
bs_paired =  model_params['batch_size_paired']


encoder_1 = SimpleEncoder(gene_size, model_params['encoder_1_hiddens'], model_params['latent_dim'],
                          dropRate=model_params['dropout_encoder'],
                          activation=model_params['encoder_activation']).to(device)
encoder_2 = SimpleEncoder(gene_size, model_params['encoder_2_hiddens'], model_params['latent_dim'],
                          dropRate=model_params['dropout_encoder'],
                          activation=model_params['encoder_activation']).to(device)


classifier = Classifier(in_channel=model_params['latent_dim'],
                        hidden_layers=model_params['state_class_hidden'],
                        num_classes=model_params['no_states'],
                        drop_in=model_params['state_class_drop_in'],
                        drop=model_params['state_class_drop']).to(device)

Vsp = SpeciesCovariate(2, model_params['latent_dim'], dropRate=model_params['V_dropout']).to(device)

encoder_1=torch.load('trained_models/alldata_cpa_encoder_pc3.pth')
encoder_2=torch.load('trained_models/alldata_cpa_encoder_ha1e.pth')
classifier = torch.load('trained_models/alldata_cpa_classifier_pc3_ha1e.pth')
Vsp = torch.load('trained_models/alldata_cpa_Vsp_pc3_ha1e.pth')

encoder_1.eval()
encoder_2.eval()
classifier.eval()
Vsp.eval()

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


### Classifier importance
print2log('Classifier impoprtance')
df_z1 = pd.read_csv('trained_embs_all/AllEmbs_CPA_pc3.csv',index_col=0).drop_duplicates()
df_z2 = pd.read_csv('trained_embs_all/AllEmbs_CPA_ha1e.csv',index_col=0).drop_duplicates()
df_latents = pd.concat((df_z1,df_z2),axis=0)
z_1 = torch.tensor(df_z1.values).to(device)
z_2 = torch.tensor(df_z2.values).to(device)
z = torch.cat((z_1,z_2),0).float()
ig = IntegratedGradients(classifier)
z.requires_grad_()
attr1 = ig.attribute(z,target=1,n_steps=100, return_convergence_delta=False)
attr1 = attr1.detach().cpu().numpy()
attr2 = ig.attribute(z,target=0,n_steps=100, return_convergence_delta=False)
attr2 = attr2.detach().cpu().numpy()
df1 = pd.DataFrame(attr1)
df1.index = df_latents.index
df1.columns = ['z'+str(i) for i in range(model_params['latent_dim'])]
df1.to_csv('Importance_results/important_scores_to_classify_as_pc3_cpa.csv')
df2 = pd.DataFrame(attr2)
df2.index = df_latents.index
df2.columns = ['z'+str(i) for i in range(model_params['latent_dim'])]
df2.to_csv('Importance_results/important_scores_to_classify_as_ha1e_cpa.csv')


# x_1_equivalent = x_1[0:paired_inds,:]
# x_2_equivalent = x_2[0:paired_inds,:]

class LatentEncode(torch.nn.Module):
    def __init__(self, encoder,V):
        super(LatentEncode, self).__init__()

        self.encoder = encoder
        self.V = V

    def forward(self, x,z_base):

        z = self.encoder(x)
        z = self.V(z, z_base)

        return z

#fullEncoder = LatentEncode(encoder_1,Vsp).to(device)
#fullEncoder.eval()
#ig = IntegratedGradients(fullEncoder)
ig = IntegratedGradients(encoder_1)

print2log('Importance to encode into composed latent space')
# Per output latent variable input importance translation captum
# 1st dimesion input
# 2nd dimesion output
hid_dim = model_params['latent_dim']
scores = torch.zeros ((gene_size,hid_dim)).to(device)
for z in range(hid_dim):
    #encoder_1.zero_grad()
    #attr, delta = ig.attribute((x_1,z_species_1),target=z,n_steps=100,return_convergence_delta=True)
    attr, delta = ig.attribute(x_1,target=z,n_steps=100,return_convergence_delta=True)
    #if z == 0:
    #    print2log(attr[0].shape)
    #scores[:,z] = torch.mean(attr[0],0)
    scores[:,z] = torch.mean(attr,0)
print2log(scores)

#plt.figure(figsize=(10,5), dpi= 80)
#sns.set(font_scale=2)
#sns.distplot(scores.flatten().cpu().numpy())
#plt.xlabel('Importance score')
#plt.ylabel('Density')
#plt.title('Distribution of importance scores of the input features')
#plt.savefig('pc3_important_scores_to_encode.png', bbox_inches='tight')


df = pd.DataFrame(scores.cpu().numpy())
df.columns = ['z'+str(i) for i in range(model_params['latent_dim'])]
df.index = cmap.columns
df.to_csv('Importance_results/important_scores_pc3_to_encode.csv')
