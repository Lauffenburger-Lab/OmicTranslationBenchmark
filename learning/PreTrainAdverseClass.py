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
from sklearn.metrics import confusion_matrix,f1_score
from sklearn.neighbors import KNeighborsClassifier
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


# ### Load data
data = pd.read_csv('all_data_scaled.csv',index_col=0)

X = torch.tensor(data.iloc[:,:-3].values).float()
# X = 1./(1+np.exp(-X))
Y = torch.tensor(data.iloc[:,-3:].values).float()
N = X.shape[0]

# ### Train model

model_params = {'encoder_hiddens':[4096,2048],
                'latent_dim': 1024, # itan 512
                'dropout_encoder':0.1,
                'encoder_activation':torch.nn.ELU(),
                'adv_class_hidden':[512,256,128],
                'adv_class_drop_in':0.3,
                'adv_class_drop':0.2,
                'no_adv_class':2,
                'encoding_lr':0.001,
                'adv_lr':0.001,
                'schedule_step_enc':25,
                'gamma_enc':0.8,
                'batch_size':1024,
                'epochs':500,
                'prior_beta':1.0,
                'adv_class_reg':1e-04,
                'enc_l2_reg':0.001}

class_criterion = torch.nn.CrossEntropyLoss()
bs= model_params['batch_size']
NUM_EPOCHS= model_params['epochs']
gene_size = X.shape[1]

# In[ ]:

valAcc = []
valF1 = []
valPrec = []
valRec = []

encoder = SimpleEncoder(gene_size,model_params['encoder_hiddens'],model_params['latent_dim'],
                              dropRate=model_params['dropout_encoder'],
                              activation=model_params['encoder_activation'],
                              normalizeOutput=True).to(device)
prior_d = PriorDiscriminator(model_params['latent_dim']).to(device)
local_d = LocalDiscriminator(model_params['latent_dim'],model_params['latent_dim']).to(device)

classifier = Classifier(in_channel=model_params['latent_dim'],
                        hidden_layers=model_params['adv_class_hidden'],
                        num_classes=model_params['no_adv_class'],
                        drop_in=model_params['adv_class_drop_in'],
                        drop=model_params['adv_class_drop']).to(device)
    
allParams = list(encoder.parameters())
allParams = allParams + list(prior_d.parameters()) + list(local_d.parameters())
allParams = allParams + list(classifier.parameters())
optimizer = torch.optim.Adam(allParams, lr= model_params['encoding_lr'], weight_decay=0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size=model_params['schedule_step_enc'],
                                            gamma=model_params['gamma_enc'])
trainLoss = []
train_eps = []
for e in range(0, NUM_EPOCHS):
    encoder.train()
    prior_d.train()
    local_d.train()
    classifier.train()

    trainloader = getSamples(N, bs)
    iteration = 1

    for dataIndex in trainloader:
        dataIn = X[dataIndex]
        true_categories = Y[dataIndex]
        ind_humans = torch.where(true_categories[:,1]==1)[0]
        ind_mouse = torch.where(true_categories[:,1]!=1)[0]
        X_human = dataIn[ind_humans,:].to(device)
        X_mouse = dataIn[ind_mouse,:].to(device)
        Y_human = true_categories[ind_humans,0]
        Y_mouse = true_categories[ind_mouse,0]
        conds_human = true_categories[ind_humans,2]
        conds_mouse = true_categories[ind_mouse,2]

        conditions_disease = np.concatenate((Y_human.numpy(),Y_mouse.numpy()))
        size = conditions_disease.size
        conditions_disease = conditions_disease.reshape(size,1)
        conditions_disease = conditions_disease == conditions_disease.transpose()
        conditions_disease = conditions_disease*1

        conditions_cell = np.concatenate((conds_human.numpy(),conds_mouse.numpy()))
        size = conditions_cell.size
        conditions_cell = conditions_cell.reshape(size,1)
        conditions_cell = conditions_cell == conditions_cell.transpose()
        conditions_cell = conditions_cell*1

        conditions = np.multiply(conditions_disease,conditions_cell)
        mask = torch.tensor(conditions).to(device).detach()
        pos_mask = mask
        neg_mask = 1 - mask
        log_2 = math.log(2.)

        optimizer.zero_grad()

        z_1 = encoder(X_human)
        z_2 = encoder(X_mouse)
        latent_vectors = torch.cat((z_1, z_2), 0)

        z_un = local_d(latent_vectors)
        res_un = torch.matmul(z_un, z_un.t())

        p_samples = res_un * pos_mask.float()
        q_samples = res_un * neg_mask.float()
        Ep = log_2 - F.softplus(- p_samples)
        Eq = F.softplus(-q_samples) + q_samples - log_2
        Ep = (Ep * pos_mask.float()).sum() / pos_mask.float().sum()
        Eq = (Eq * neg_mask.float()).sum() / neg_mask.float().sum()
        mi_loss = Eq - Ep
        prior = torch.rand_like(latent_vectors)
        term_a = torch.log(prior_d(prior)).mean()
        term_b = torch.log(1.0 - prior_d(latent_vectors)).mean()
        prior_loss = -(term_a + term_b) * model_params['prior_beta']

        labels = classifier(latent_vectors)
        true_labels = torch.cat((torch.ones(z_1.shape[0]),
                                 torch.zeros(z_2.shape[0])),0).long().to(device)
        entropy = class_criterion(labels,true_labels)
        _, predicted = torch.max(labels, 1)
        predicted = predicted.cpu().numpy()
        cf_matrix = confusion_matrix(true_labels.cpu(),predicted)
        tn, fp, fn, tp = cf_matrix.ravel()
        acc = (tp + tn) / predicted.size
        f1 = 2*tp/(2*tp+fp+fn)

        loss = mi_loss + prior_loss + entropy +\
               classifier.L2Regularization(model_params['adv_class_reg']) +\
               encoder.L2Regularization(model_params['enc_l2_reg'])
        loss.backward()
        optimizer.step()

    scheduler.step()
    outString = 'Epoch={:.0f}/{:.0f}'.format(fold + 1, e + 1, NUM_EPOCHS)
    outString += ', MI Loss={:.4f}'.format(mi_loss.item())
    outString += ', Prior Loss={:.4f}'.format(prior_loss.item())
    outString += ', Entropy Loss={:.4f}'.format(entropy.item())
    outString += ', loss={:.4f}'.format(loss.item())
    outString += ', Accuracy_latent={:.4f}'.format(acc)
    outString += ', F1_latent={:.4f}'.format(f1)
    if ((e%10==0 and e>0) or e==1):
        print2log(outString)
print2log(outString)

torch.save(encoder_human,'master_encoder.pth')
torch.save(prior_d,'master_prior_d.pth')
torch.save(local_d,'master_local_d.pth')
torch.save(classifier,'pretrained_classifier.pth')