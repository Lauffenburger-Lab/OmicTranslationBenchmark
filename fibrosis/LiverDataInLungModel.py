import pickle
import torch
import torch.nn.functional as F
from models import Encoder,Decoder,GaussianDecoder,VAE,CellStateEncoder,\
                   CellStateDecoder, CellStateVAE,\
                   SimpleEncoder,LocalDiscriminator,PriorDiscriminator,\
                   EmbInfomax,MultiEncInfomax,Classifier,\
                   SpeciesCovariate,GaussianDecoder
import math
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split,cross_validate,KFold
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from evaluationUtils import r_square,get_cindex,pearson_r,pseudoAccuracy
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

### Load data
print2log('Start loading mouse')
mouse_df_frame = pd.read_csv('data/all_mouse_lung.csv',index_col=0)
mouse_df_frame = mouse_df_frame.iloc[:,:-3]
mouse_columns = mouse_df_frame.columns.values
#print2log(mouse_df_frame.shape)
mouse_df = pd.read_csv('all_mouse_scaled.csv',index_col=0)
Ym = torch.tensor(mouse_df.iloc[:,-3:].values)
#print2log(mouse_df.shape)
mouse_df = mouse_df.iloc[:,:-3]
mouse_df_frame = torch.zeros(len(mouse_df),mouse_df_frame.shape[1])
mouse_df_frame = pd.DataFrame(mouse_df_frame.numpy())
mouse_df_frame.columns = mouse_columns
liver_columns = np.intersect1d(mouse_df.columns,mouse_columns)
mouse_df = mouse_df.loc[:,liver_columns]
mouse_df_frame.loc[:,liver_columns] = mouse_df.values
Xm = torch.tensor(mouse_df_frame.values)
print2log('Start saving mouse')
torch.save(Xm,'data/liver_mouse_genes.pt')
torch.save(Ym,'data/liver_mouse_labels.pt')
# Xm = torch.load('data/liver_mouse_genes.pt')
#print2log(Xm.shape)
# Ym = torch.load('data/liver_mouse_labels.pt')

print2log('Start loading human')
human_df_frame = pd.read_csv('data/all_human_lung.csv',index_col=0)
human_df_frame = human_df_frame.iloc[:,:-3]
human_columns = human_df_frame.columns.values
#print2log(human_df_frame.shape)
human_df = pd.read_csv('data/human_liver_fibrosis_immune_only.csv',index_col=0)
Yh = torch.tensor(human_df.iloc[:,-3:].values)
#print2log(human_df.shape)
human_df = human_df.iloc[:,:-3]
human_df_frame = torch.zeros(len(human_df),human_df_frame.shape[1])
human_df_frame = pd.DataFrame(human_df_frame.numpy())
human_df_frame.columns = human_columns
liver_columns = np.intersect1d(human_df.columns,human_columns)
human_df = human_df.loc[:,liver_columns]
human_df_frame.loc[:,liver_columns] = human_df.values
Xh = torch.tensor(human_df_frame.values)
Xh = torch.log1p(Xh)
print2log('Start saving human')
print2log(Xh.shape)
torch.save(Xh,'data/liver_human_genes.pt')
torch.save(Yh,'data/liver_human_labels.pt')
#Xh = torch.load('data/liver_human_genes.pt')
#Yh = torch.load('data/liver_human_labels.pt')

### Train model
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

bs= model_params['batch_size']
k_folds=model_params['no_folds']
NUM_EPOCHS= model_params['epochs']


print2log('Start predictions:')
for i in range(k_folds):
    fold = i 
    encoder_human = torch.load('models/encoder_human_%s.pth' % fold).to(device)
    encoder_mouse = torch.load('models/encoder_mouse_%s.pth'%fold).to(device)
    classifier = torch.load('models/classifier_disease_%s.pth'%fold).to(device)
    species_classifier = torch.load('models/species_classifier_%s.pt'%fold).to(device)
    cell_classifier = torch.load('models/classifier_cell_%s.pth' % fold).to(device)
    Vsp = torch.load('models/Vspecies_%s.pt'%fold).to(device)
    Vcell = torch.load('models/Vcell_%s.pt'%fold).to(device)

    encoder_human.eval()
    encoder_mouse.eval()
    classifier.eval()
    species_classifier.eval()
    cell_classifier.eval()
    Vcell.eval()
    Vsp.eval()

    # Translate to other species
    z_species_train_1 = torch.cat((torch.ones(Xh.shape[0], 1),
                                  torch.zeros(Xh.shape[0], 1)), 1).to(device)
    z_species_train_2 = torch.cat((torch.zeros(Xm.shape[0], 1),
                                   torch.ones(Xm.shape[0], 1)), 1).to(device)
    z_cell_train_1 = torch.cat((1 * (Yh[:, 2] == 1).unsqueeze(1), 1 * (Yh[:, 2] == 2).unsqueeze(1),
                               1 * (Yh[:, 2] == 3).unsqueeze(1), 1 * (Yh[:, 2] == 4).unsqueeze(1),
                               1 * (Yh[:, 2] == 5).unsqueeze(1)), 1).float().to(device)
    z_cell_train_2 = torch.cat((1 * (Ym[:, 2] == 1).unsqueeze(1), 1 * (Ym[:, 2] == 2).unsqueeze(1),
                                1 * (Ym[:, 2] == 3).unsqueeze(1), 1 * (Ym[:, 2] == 4).unsqueeze(1),
                                1 * (Ym[:, 2] == 5).unsqueeze(1)), 1).float().to(device)
    z_train_base_1 = torch.zeros(Xh.shape[0], model_params['latent_dim'])
    z_train_base_2 = torch.zeros(Xm.shape[0], model_params['latent_dim'])
    z1_train = torch.zeros(Xh.shape[0], model_params['latent_dim'])
    z2_train = torch.zeros(Xm.shape[0], model_params['latent_dim'])
    y_cell_mouse = torch.zeros(Xm.shape[0])
    y_cell_human = torch.zeros(Xh.shape[0])
    y_species_mouse = torch.zeros(Xm.shape[0])
    y_species_human = torch.zeros(Xh.shape[0])
    y_mouse = torch.zeros(Xm.shape[0])
    y_human = torch.zeros(Xh.shape[0])
    val_bs_1 = 512
    val_bs_2 = 512
    ii = 0
    jj = 0
    while (ii < Xh.shape[0]):
       if (ii + val_bs_1 > Xh.shape[0]):
           z_train_base_1[ii:, :] = encoder_human(Xh[ii:, :].to(device)).cpu()
           z1_train[ii:, :] = Vsp(z_train_base_1[ii:, :].to(device),
                                  z_species_train_1[ii:, :]).cpu()
           z1_train[ii:, :] = Vcell(z1_train[ii:, :].to(device), z_cell_train_1[ii:, :]).cpu()
           _,y_cell_human[ii:] = torch.max(cell_classifier(z1_train[ii:, :].to(device)).cpu(),1)
           _,y_species_human[ii:] = torch.max(species_classifier(z1_train[ii:, :].to(device)).cpu(),1)
           _,y_human[ii:] = torch.max(classifier(z1_train[ii:, :].to(device)).cpu(),1)
           # z1_train[ii:,:] = encoder_interm_1(z_train_base_1[ii:,:].to(device)).cpu()
       else:
           z_train_base_1[ii:(ii + val_bs_1), :] = encoder_human(Xh[ii:(ii + val_bs_1), :].to(device)).cpu()
           z1_train[ii:(ii + val_bs_1), :] = Vsp(z_train_base_1[ii:(ii + val_bs_1), :].to(device),
                                                 z_species_train_1[ii:(ii + val_bs_1), :]).cpu()
           z1_train[ii:(ii + val_bs_1), :] = Vcell(z1_train[ii:(ii + val_bs_1), :].to(device),
                                                   z_cell_train_1[ii:(ii + val_bs_1), :]).cpu()
           _,y_cell_human[ii:(ii + val_bs_1)] = torch.max(cell_classifier(z1_train[ii:(ii + val_bs_1), :].to(device)).cpu(),1)
           _,y_species_human[ii:(ii + val_bs_1)] = torch.max(species_classifier(z1_train[ii:(ii + val_bs_1), :].to(device)).cpu(),1)
           _,y_human[ii:(ii + val_bs_1)] = torch.max(classifier(z1_train[ii:(ii + val_bs_1), :].to(device)).cpu(),1)
           # z1_train[ii:(ii+val_bs_1),:] = encoder_interm_1(z_train_base_1[ii:(ii+val_bs_1),:].to(device)).cpu()
       ii = ii + val_bs_1

    while (jj < Xm.shape[0]):
        if (jj + val_bs_2 > Xm.shape[0]):
            z_train_base_2[jj:, :] = encoder_mouse(Xm[jj:, :].to(device)).cpu()
            z2_train[jj:, :] = Vsp(z_train_base_2[jj:, :].to(device),
                                   z_species_train_2[jj:, :]).cpu()
            z2_train[jj:, :] = Vcell(z2_train[jj:, :].to(device), z_cell_train_2[jj:, :]).cpu()
            _,y_cell_mouse[jj:] = torch.max(cell_classifier(z2_train[jj:, :].to(device)).cpu(),1)
            _,y_species_mouse[jj:] = torch.max(species_classifier(z2_train[jj:, :].to(device)).cpu(),1)
            _,y_mouse[jj:] = torch.max(classifier(z2_train[jj:, :].to(device)).cpu(),1)
            # z2_train[jj:,:] = encoder_interm_2(z_train_base_2[jj:,:].to(device)).cpu()
        else:
            z_train_base_2[jj:(jj + val_bs_2), :] = encoder_mouse(Xm[jj:(jj + val_bs_2), :].to(device)).cpu()
            z2_train[jj:(jj + val_bs_2), :] = Vsp(z_train_base_2[jj:(jj + val_bs_2), :].to(device),
                                                  z_species_train_2[jj:(jj + val_bs_2), :]).cpu()
            z2_train[jj:(jj + val_bs_2), :] = Vcell(z2_train[jj:(jj + val_bs_2), :].to(device),
                                                    z_cell_train_2[jj:(jj + val_bs_2), :]).cpu()
            _,y_cell_mouse[jj:(jj + val_bs_2)] = torch.max(cell_classifier(z2_train[jj:(jj + val_bs_2), :].to(device)).cpu(),1)
            _,y_species_mouse[jj:(jj + val_bs_2)] = torch.max(species_classifier(z2_train[jj:(jj + val_bs_2), :].to(device)).cpu(),1)
            _,y_mouse[jj:(jj + val_bs_2)] = torch.max(classifier(z2_train[jj:(jj + val_bs_2), :].to(device)).cpu(),1)
            # z2_train[jj:(jj+val_bs_2),:] = encoder_interm_2(z_train_base_2[jj:(jj+val_bs_2),:].to(device)).cpu()
        jj = jj + val_bs_2

    df_disease_mouse = pd.DataFrame(torch.cat((y_mouse.unsqueeze(1),Ym[:,0:1]),1).detach().numpy())
    df_disease_mouse.columns = ['fibrosis_pred','fibrosis_true']
    df_cell_mouse = pd.DataFrame(torch.cat((y_cell_mouse.unsqueeze(1), Ym[:, 2:3]), 1).detach().numpy())
    df_cell_mouse.columns = ['cell_pred', 'cell_true']
    df_species_mouse = pd.DataFrame(torch.cat((y_species_mouse.unsqueeze(1), Ym[:, 1:2]), 1).detach().numpy())
    df_species_mouse.columns = ['species_pred', 'species_true']
    df_latent_mouse = pd.DataFrame(z2_train.detach().numpy())
    df_latent_mouse.columns = ['z'+str(z) for z in range(model_params['latent_dim'])]
    df_mouse = pd.concat([df_latent_mouse, df_disease_mouse,df_species_mouse,df_cell_mouse], axis=1)
    df_mouse.index = mouse_df.index
    df_mouse.to_csv('results/embs/LiverEmbs_mouse_%s.csv'%fold)
    df_latent_basal_mouse = pd.DataFrame(z_train_base_2.detach().numpy())
    df_latent_basal_mouse.columns = ['z' + str(z) for z in range(model_params['latent_dim'])]
    df_latent_basal_mouse.index = mouse_df.index
    df_latent_basal_mouse.to_csv('results/embs/LiverEmbsBasal_mouse_%s.csv'%fold)

    df_disease_human = pd.DataFrame(torch.cat((y_human.unsqueeze(1), Yh[:, 0:1]), 1).detach().numpy())
    df_disease_human.columns = ['fibrosis_pred', 'fibrosis_true']
    df_cell_human = pd.DataFrame(torch.cat((y_cell_human.unsqueeze(1), Yh[:, 2:3]), 1).detach().numpy())
    df_cell_human.columns = ['cell_pred', 'cell_true']
    df_species_human = pd.DataFrame(torch.cat((y_species_human.unsqueeze(1), Yh[:, 1:2]), 1).detach().numpy())
    df_species_human.columns = ['species_pred', 'species_true']
    df_latent_human = pd.DataFrame(z1_train.detach().numpy())
    df_latent_human.columns = ['z' + str(z) for z in range(model_params['latent_dim'])]
    df_human = pd.concat([df_latent_human, df_disease_human, df_species_human, df_cell_human], axis=1)
    df_human.index = human_df.index
    df_human.to_csv('results/embs/LiverEmbs_human_%s.csv'%fold)
    df_latent_basal_human = pd.DataFrame(z_train_base_1.detach().numpy())
    df_latent_basal_human.columns = ['z' + str(z) for z in range(model_params['latent_dim'])]
    df_latent_basal_human.index = human_df.index
    df_latent_basal_human.to_csv('results/embs/LiverEmbsBasal_human_%s.csv'%fold)

    print2log('Finished %s'%i)
