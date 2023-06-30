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
pre_encoder_human = torch.load('pre_models/encoder_human_6.pth')
pre_encoder_mouse = torch.load('pre_models/encoder_mouse_6.pth')
pre_prior_d = torch.load('pre_models/prior_d_6.pth')
pre_local_d = torch.load('pre_models/local_d_6.pth')
#pre_prior_d_2 = torch.load('pre_models/prior2_d_7.pth')
#pre_local_d_2 = torch.load('pre_models/local2_d_7.pth')
pre_classifier = torch.load('pre_models/classifier_disease_6.pth')
pre_species_classifier = torch.load('pre_models/species_classifier_6.pt')
pre_cell_classifier = torch.load('pre_models/classifier_cell_6.pth')
#pre_encoder_interm_1 = torch.load('pre_models/encoder_interm_human_7.pt')
#pre_encoder_interm_2 = torch.load('pre_models/encoder_interm_mouse_7.pt')
pre_adverse_classifier = torch.load('pre_models/classifier_adverse_6.pt')
pre_cell_adverse_classifier = torch.load('pre_models/classifier_cell_adverse_6.pt')
pre_Vsp = torch.load('pre_models/Vspecies_6.pt')
pre_Vcell = torch.load('pre_models/Vcell_6.pt')


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
    results = pd.read_csv('results/10foldvalidationResults_lungs.csv',index_col=0)
    valAcc = results['Accuracy'].tolist()
    valF1 = results['F1'].tolist()
    valPrec = results['Precision'].tolist()
    valRec = results['Recall'].tolist()
    valF1basal = results['BasalF1'].tolist()
    valAccTrans = results['TranslationAccuracy'].tolist()
    valF1Trans = results['TranslationF1'].tolist()
    valF1KNN = []
    valPrecKNN = []
    valRecKNN = []
    valAccKNN = []
    valF1Species = results['F1 Species'].tolist()
    valAccSpecies =results['Accuracy Species'].tolist()
    valF1SpeciesTrans = results['F1 species traslation'].tolist()
    valAccSpeciesTrans = results['Accuracy species translation'].tolist()
    valF1CellTrans = results['F1 cell-type species traslation'].tolist()
    valF1Cell = results['F1 Cell'].tolist()
    mean_score_human_val = results['r2_mu_human'].tolist()
    mean_score_mouse_val = results['r2_mu_mouse'].tolist()
    var_score_human_val = results['r2_var_human'].tolist()
    var_score_mouse_val = results['r2_var_mouse'].tolist()
    mean_score_trans_1to2 = results['r2_mu_human_to_mouse'].tolist()
    mean_score_trans_2to1 = results['r2_mu_mouse_to_human'].tolist()
    var_score_trans_1to2 = results['r2_var_human_to_mouse'].tolist()
    var_score_trans_2to1 = results['r2_var_mouse_to_human'].tolist()

#pretrained_classifier = torch.load('pretrained_classifier.pth')

for fold in range(begin_fold,model_params['no_folds']):
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

    gene_size_mouse = xtrain_mouse.shape[1]
    gene_size_human = xtrain_human.shape[1]

    N_2 = ytrain_mouse.shape[0]
    N_1 = ytrain_human.shape[0]

    N = N_1
    if N_2 > N:
        N = N_2
    
    # Network
    encoder_human = torch.nn.Sequential(ElementWiseLinear(gene_size_human),
                                        SimpleEncoder(gene_size_human,
                                        model_params['encoder_1_hiddens'],
                                        model_params['latent_dim'],
                                        dropRate=model_params['dropout_encoder'],
                                        activation=model_params['encoder_activation'],
                                        normalizeOutput=False)).to(device)
    encoder_mouse = torch.nn.Sequential(ElementWiseLinear(gene_size_mouse),
                                        SimpleEncoder(gene_size_mouse,
                                        model_params['encoder_2_hiddens'],
                                        model_params['latent_dim'],
                                        dropRate=model_params['dropout_encoder'],
                                        activation=model_params['encoder_activation'],
                                        normalizeOutput=False)).to(device)
    #MultipleElementWise_human = torch.nn.Sequential(ElementWiseLinear(gene_size_human),
    #                                                ElementWiseLinear(gene_size_human),
    #                                                ElementWiseLinear(gene_size_human)).to(device)
    #MultipleElementWise_mouse = torch.nn.Sequential(ElementWiseLinear(gene_size_mouse),
    #                                                ElementWiseLinear(gene_size_mouse),
    #                                                ElementWiseLinear(gene_size_mouse)).to(device)

    decoder_human = VarDecoder(model_params['latent_dim'],model_params['decoder_1_hiddens'],gene_size_human,
                           dropRate=model_params['dropout_decoder'],
                           activation=model_params['decoder_activation']).to(device)
    decoder_mouse = VarDecoder(model_params['latent_dim'],model_params['decoder_2_hiddens'],gene_size_mouse,
                           dropRate=model_params['dropout_decoder'],
                           activation=model_params['decoder_activation']).to(device)
    #distribution_mu_human = torch.nn.Linear(model_params['final_dec_1'],gene_size_human,bias=False).to(device)
    #distribution_var_human = torch.nn.Linear(model_params['final_dec_1'],gene_size_human,bias=False).to(device)
    #encoder_human = SimpleEncoder(gene_size_human,model_params['encoder_1_hiddens'],model_params['latent_dim'],
    #                              dropRate=model_params['dropout_encoder'], 
    #                              activation=model_params['encoder_activation'],
    #                              normalizeOutput=True).to(device)
    #encoder_mouse = SimpleEncoder(gene_size_mouse,model_params['encoder_2_hiddens'],model_params['latent_dim'],
    #                              dropRate=model_params['dropout_encoder'], 
    #                              activation=model_params['encoder_activation'],
    #                              normalizeOutput=True).to(device)
    prior_d = PriorDiscriminator(model_params['latent_dim']).to(device)
    local_d = LocalDiscriminator(model_params['latent_dim'],model_params['latent_dim']).to(device)
    
    classifier = Classifier(in_channel=model_params['latent_dim'],
                            hidden_layers=model_params['state_class_hidden'],
                            num_classes=model_params['no_states'],
                            drop_in=model_params['state_class_drop_in'],
                            drop=model_params['state_class_drop']).to(device)
    adverse_classifier = Classifier(in_channel=model_params['latent_dim'],
                                    hidden_layers=model_params['adv_class_hidden'],
                                    num_classes=model_params['no_adv_class'],
                                    drop_in=model_params['adv_class_drop_in'],
                                    drop=model_params['adv_class_drop']).to(device)
    cell_adverse_classifier = Classifier(in_channel=model_params['latent_dim'],
                                    hidden_layers=model_params['adv_class_hidden'],
                                    num_classes=model_params['no_cell_types'],
                                    drop_in=model_params['adv_class_drop_in'],
                                    drop=model_params['adv_class_drop']).to(device)
    #adverse_classifier.load_state_dict(pretrained_classifier.state_dict())
    
    species_classifier = Classifier(in_channel=model_params['latent_dim'],
                            hidden_layers=model_params['state_class_hidden'],
                            num_classes=2,
                            drop_in=model_params['state_class_drop_in'],
                            drop=model_params['state_class_drop']).to(device)

    cell_classifier = Classifier(in_channel=model_params['latent_dim'],
                                 hidden_layers=model_params['cell_class_hidden'],
                                 num_classes=model_params['no_cell_types'],
                                 drop_in=model_params['cell_class_drop_in'],
                                 drop=model_params['cell_class_drop']).to(device)

    Vsp = SpeciesCovariate(2,model_params['latent_dim'],dropRate=model_params['V_dropout']).to(device)
    Vcell = SpeciesCovariate(model_params['no_cell_types'],model_params['latent_dim'],dropRate=model_params['V_dropout']).to(device)

    #encoder_interm_1 = SimpleEncoder(model_params['latent_dim'],
    #                                 model_params['intermediateEncoder1'],
    #                                 model_params['intermediate_latent'],
    #                                 dropRate=model_params['intermediate_dropout'],
    #                                 activation=model_params['encoder_activation']).to(device)

    #encoder_interm_2 = SimpleEncoder(model_params['latent_dim'],
    #                                 model_params['intermediateEncoder2'],
    #                                 model_params['intermediate_latent'],
    #                                 dropRate=model_params['intermediate_dropout'], 
    #                                 activation=model_params['encoder_activation']).to(device)

    #prior_d_2 = PriorDiscriminator(model_params['intermediate_latent']).to(device)
    #local_d_2 = LocalDiscriminator(model_params['intermediate_latent'],model_params['intermediate_latent']).to(device)

    encoder_human.load_state_dict(pre_encoder_human.state_dict())
    encoder_mouse.load_state_dict(pre_encoder_mouse.state_dict())
    prior_d.load_state_dict(pre_prior_d.state_dict())
    local_d.load_state_dict(pre_local_d.state_dict())
    #prior_d_2.load_state_dict(pre_prior_d_2.state_dict())
    #local_d_2.load_state_dict(pre_local_d_2.state_dict())
    classifier.load_state_dict(pre_classifier.state_dict())
    species_classifier.load_state_dict(pre_species_classifier.state_dict())
    cell_classifier.load_state_dict(pre_cell_classifier.state_dict())
    #encoder_interm_1.load_state_dict(pre_encoder_interm_1.state_dict())
    #encoder_interm_2.load_state_dict(pre_encoder_interm_2.state_dict())
    adverse_classifier.load_state_dict(pre_adverse_classifier.state_dict())
    cell_adverse_classifier.load_state_dict(pre_cell_adverse_classifier.state_dict())
    Vsp.load_state_dict(pre_Vsp.state_dict())
    Vcell.load_state_dict(pre_Vcell.state_dict())


    allParams = list(encoder_human.parameters()) + list(encoder_mouse.parameters())
    allParams = list(decoder_human.parameters()) + list(decoder_mouse.parameters())
    allParams = allParams + list(prior_d.parameters()) + list(local_d.parameters())
    allParams = allParams + list(classifier.parameters())
    allParams = allParams + list(Vsp.parameters())
    allParams = allParams + list(species_classifier.parameters())
    allParams = allParams + list(cell_classifier.parameters())
    allParams = allParams + list(Vcell.parameters())
    #allParams = allParams + list(encoder_interm_1.parameters()) + list(encoder_interm_2.parameters())
    #allParams = allParams + list(prior_d_2.parameters()) + list(local_d_2.parameters())
    #allParams = allParams + list(MultipleElementWise_human.parameters()) + list(MultipleElementWise_mouse.parameters())
    optimizer = torch.optim.Adam(allParams, lr= model_params['encoding_lr'], weight_decay=0)
    optimizer_adv = torch.optim.Adam(list(adverse_classifier.parameters())+ list(cell_adverse_classifier.parameters()), lr= model_params['adv_lr'], weight_decay=0)
    if model_params['schedule_step_adv'] is not None:
        scheduler_adv = torch.optim.lr_scheduler.StepLR(optimizer_adv,
                                                        step_size=model_params['schedule_step_adv'],
                                                        gamma=model_params['gamma_adv'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=model_params['schedule_step_enc'],
                                                gamma=model_params['gamma_enc'])
    trainLoss = []
    train_eps = []
    f1_basal_trained = None 
    for e in range(0, NUM_EPOCHS):
        encoder_human.train()
        encoder_mouse.train()
        decoder_human.train()
        decoder_mouse.train()
        prior_d.train()
        local_d.train()
        classifier.train()
        adverse_classifier.train()
        species_classifier.train()
        cell_classifier.train()
        cell_adverse_classifier.train()
        Vcell.train()
        Vsp.train()
        #encoder_interm_1.train()
        #encoder_interm_2.train()
        #prior_d_2.train()
        #local_d_2.train()
        #MultipleElementWise_human.train()
        #MultipleElementWise_mouse.train()

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

            z_species_1 = torch.cat((torch.ones(X_human.shape[0],1),
                                     torch.zeros(X_human.shape[0],1)),1).to(device)
            z_species_2 = torch.cat((torch.zeros(X_mouse.shape[0],1),
                                     torch.ones(X_mouse.shape[0],1)),1).to(device)
            z_cell_1 = torch.cat((1*(conds_human==1).unsqueeze(1),1*(conds_human==2).unsqueeze(1),
                                  1*(conds_human==3).unsqueeze(1),1*(conds_human==4).unsqueeze(1),1*(conds_human==5).unsqueeze(1)),1).float().to(device)
            z_cell_2 = torch.cat((1*(conds_mouse==1).unsqueeze(1),1*(conds_mouse==2).unsqueeze(1),
                                  1*(conds_mouse==3).unsqueeze(1),1*(conds_mouse==4).unsqueeze(1),1*(conds_mouse==5).unsqueeze(1)),1).float().to(device)

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

            conditions_species = np.concatenate((np.ones(X_human.shape[0]),1.0+np.ones(X_mouse.shape[0])))
            size = conditions_species.size
            conditions_species = conditions_species.reshape(size,1)
            conditions_species = conditions_species == conditions_species.transpose()
            conditions_species = conditions_species*1
            mask_species = torch.tensor(conditions_species).to(device).detach()

            #conditions = np.multiply(conditions_disease,conditions_cell)
            #mask = torch.tensor(conditions).to(device).detach()
            mask2 = torch.tensor(np.multiply(conditions_disease,conditions_cell)).to(device).detach()
            #mask_cell = torch.tensor(conditions_cell).to(device).detach()
            mask = torch.tensor(conditions_disease).to(device).detach()
            pos_mask = mask
            neg_mask = 1 - mask
            pos_mask2 = mask2
            neg_mask2 = 1 - mask2
            log_2 = math.log(2.)

            optimizer.zero_grad()
            optimizer_adv.zero_grad()

            if (e % model_params["adversary_steps"]  == 0 ):
                z_base_1 = encoder_human(torch.log1p(X_human))
                z_base_2 = encoder_mouse(torch.log1p(X_mouse))
                #z_base_1 = encoder_human(MultipleElementWise_human(torch.log1p(X_human)))
                #z_base_2 = encoder_mouse(MultipleElementWise_mouse(torch.log1p(X_mouse)))
                latent_base_vectors = torch.cat((z_base_1, z_base_2), 0)
                labels_adv = adverse_classifier(latent_base_vectors)
                true_labels = torch.cat((torch.ones(z_base_1.shape[0]),
                                     torch.zeros(z_base_2.shape[0])),0).long().to(device)
                adv_entropy = class_criterion(labels_adv,true_labels)
                adversary_species_penalty = compute_gradients(labels_adv.sum(), latent_base_vectors)

                cell_labels_adv = cell_adverse_classifier(latent_base_vectors)
                cell_true_labels = torch.cat((conds_human,
                                              conds_mouse), 0).long().to(device) - 1
                adv_cell_entropy = class_criterion(cell_labels_adv,cell_true_labels)
                adversary_cell_penalty = compute_gradients(cell_labels_adv.sum(), latent_base_vectors)

                loss_adv = adv_entropy + model_params['adv_penalnty'] * adversary_species_penalty +\
                           adv_cell_entropy + model_params['adv_penalnty'] * adversary_cell_penalty
                loss_adv.backward()
                optimizer_adv.step()

                _, predicted = torch.max(labels_adv, 1)
                predicted = predicted.cpu().numpy()
                cf_matrix = confusion_matrix(true_labels.cpu(),predicted)
                tn, fp, fn, tp = cf_matrix.ravel()
                f1_basal_trained = 2*tp/(2*tp+fp+fn)

                _, cell_predicted = torch.max(cell_labels_adv, 1)
                cell_predicted = cell_predicted.cpu().numpy()
                cell_basal_f1 = f1_score(cell_true_labels.cpu(), cell_predicted, average='micro')
            else:
                #optimizer.zero_grad()
                z_base_1 = encoder_human(torch.log1p(X_human))
                z_base_2 = encoder_mouse(torch.log1p(X_mouse))
                latent_base_vectors = torch.cat((z_base_1, z_base_2), 0)

                z_un = local_d(latent_base_vectors)
                res_un = torch.matmul(z_un, z_un.t())

                z_1 = Vsp(z_base_1,z_species_1) 
                z_1 = Vcell(z_1,z_cell_1)
                z_2 = Vsp(z_base_2,z_species_2) 
                z_2 = Vcell(z_2,z_cell_2)
                #z_1 = encoder_interm_1(z_base_1)
                #z_2 = encoder_interm_2(z_base_2)
                latent_vectors = torch.cat((z_1, z_2), 0)
                #z_un_interm = local_d_2(latent_vectors)
                #res_un_interm = torch.matmul(z_un_interm, z_un_interm.t())

                gene_means_1 , gene_vars_1 = decoder_human(z_1)
                reconstruction_loss_1 = recon_criterion(gene_means_1, X_human, gene_vars_1)

                gene_means_2 , gene_vars_2 = decoder_mouse(z_2)
                reconstruction_loss_2 = recon_criterion(gene_means_2, X_mouse, gene_vars_2)

                silimalityLoss = torch.sum(torch.cdist(latent_base_vectors,latent_base_vectors) * mask.float())/(mask.float().sum())
                w1 = latent_base_vectors.norm(p=2, dim=1, keepdim=True)
                w2 = latent_base_vectors.norm(p=2, dim=1, keepdim=True)
                cosineLoss = torch.mm(latent_base_vectors, latent_base_vectors.t()) / (w1 * w2.t()).clamp(min=1e-6)
                cosineLossRand =  torch.sum(cosineLoss * (1.0-mask_species.float()))/(1.0-mask_species.float()).sum()
                cosineLossSpecies =  torch.sum(cosineLoss * mask_species.float())/mask_species.float().sum()
                sA_squared_base = torch.sum(torch.square(cosineLoss -cosineLossSpecies) * mask_species.float())/mask_species.float().sum()
                sB_squared_base = torch.sum(torch.square(cosineLoss - cosineLossRand) * (1.0-mask_species.float()))/((1.0-mask_species.float()).sum())
                s_base = ((torch.sum(mask_species.float()) - 1.0) * sA_squared_base + (torch.sum(1.0-mask_species.float()) - 1.0) * sB_squared_base)/(torch.sum(mask_species.float()) + torch.sum(1.0-mask_species.float()) -2)
                s_base = torch.sqrt(s_base).detach()
                cohen_base = (cosineLossSpecies- cosineLossRand)/s_base
                cohen_base =cohen_base.detach()
                cosineLoss = torch.sum(cosineLoss * mask.float())/mask.float().sum()
                w1_latent = latent_vectors.norm(p=2, dim=1, keepdim=True)
                w2_latent = latent_vectors.norm(p=2, dim=1, keepdim=True)
                cosineLoss_latent = torch.mm(latent_vectors, latent_vectors.t()) / (w1_latent * w2_latent.t()).clamp(min=1e-6)
                cosineLossA = torch.sum(cosineLoss_latent * mask_species.float())/mask_species.float().sum()
                cosineLossB = torch.sum(cosineLoss_latent * (1.0-mask_species.float()))/(1.0-mask_species.float()).sum()
                sA_squared = torch.sum(torch.square(cosineLoss_latent - cosineLossA) * mask_species.float())/mask_species.float().sum()
                sB_squared = torch.sum(torch.square(cosineLoss_latent - cosineLossB) * (1.0-mask_species.float()))/((1.0-mask_species.float()).sum())
                s = ((torch.sum(mask_species.float()) - 1.0) * sA_squared + (torch.sum(1.0-mask_species.float()) - 1.0) * sB_squared)/(torch.sum(mask_species.float()) + torch.sum(1.0-mask_species.float()) -2)
                s = torch.sqrt(s).detach()
                cohen = (cosineLossA - cosineLossB)/s
                cohen = cohen.detach()

                p_samples = res_un * pos_mask.float()
                q_samples = res_un * neg_mask.float()

                Ep = log_2 - F.softplus(- p_samples)
                Eq = F.softplus(-q_samples) + q_samples - log_2

                Ep = (Ep * pos_mask.float()).sum() / pos_mask.float().sum()
                Eq = (Eq * neg_mask.float()).sum() / neg_mask.float().sum()
                mi_loss = Eq - Ep

                #prior = torch.rand_like(torch.cat((z_1, z_2), 0))
                prior = torch.rand_like(latent_base_vectors)

                term_a = torch.log(prior_d(prior)).mean()
                #term_b = torch.log(1.0 - prior_d(torch.cat((z_1, z_2), 0))).mean()
                term_b = torch.log(1.0 - prior_d(latent_base_vectors)).mean()
                prior_loss = -(term_a + term_b) * model_params['prior_beta']

                #p_samples_interm = res_un_interm * pos_mask2.float()
                #q_samples_interm = res_un_interm * neg_mask2.float()
                #Ep_interm = log_2 - F.softplus(- p_samples_interm)
                #Eq_interm = F.softplus(-q_samples_interm) + q_samples_interm - log_2
                #Ep_interm = (Ep_interm * pos_mask2.float()).sum() / pos_mask2.float().sum()
                #Eq_interm = (Eq_interm * neg_mask2.float()).sum() / neg_mask2.float().sum()
                #mi_loss_interm = Eq_interm - Ep_interm
                #prior_interm = torch.rand_like(latent_vectors)
                #term_a_interm = torch.log(prior_d_2(prior_interm)).mean()
                #term_b_interm = torch.log(1.0 - prior_d_2(latent_vectors)).mean()
                #prior_loss_interm = -(term_a_interm + term_b_interm) * model_params['prior_beta']

                # Classification loss
                labels = classifier(latent_vectors)
                true_labels = torch.cat((Y_human,Y_mouse),0).long().to(device)
                entropy = class_criterion(labels,true_labels)
                _, predicted = torch.max(labels, 1)
                predicted = predicted.cpu().numpy()
                cf_matrix = confusion_matrix(true_labels.cpu(),predicted)
                tn, fp, fn, tp = cf_matrix.ravel()
                acc = (tp+tn)/predicted.size
                f1 = 2*tp/(2*tp+fp+fn)

                # Species classification loss
                species_labels = species_classifier(latent_vectors)
                species_true_labels = torch.cat((torch.ones(Y_human.shape[0]),
                                                 torch.zeros(Y_mouse.shape[0])),0).long().to(device)
                species_entropy = class_criterion(species_labels,species_true_labels)
                _, species_predicted = torch.max(species_labels, 1)
                species_predicted = species_predicted.cpu().numpy()
                cf_matrix = confusion_matrix(species_true_labels.cpu(),species_predicted)
                tn, fp, fn, tp = cf_matrix.ravel()
                species_acc = (tp+tn)/species_predicted.size
                species_f1 = 2*tp/(2*tp+fp+fn)

                # Cell classification
                cell_labels = cell_classifier(latent_vectors)
                cell_true_labels = torch.cat((conds_human,
                                              conds_mouse), 0).long().to(device) - 1
                cell_entropy = class_criterion(cell_labels, cell_true_labels)
                _, cell_predicted = torch.max(cell_labels, 1)
                cell_predicted = cell_predicted.cpu().numpy()
                cell_f1 = f1_score(cell_true_labels.cpu(), cell_predicted, average='micro')

                # Remove signal from z_basal
                labels_adv = adverse_classifier(latent_base_vectors)
                true_labels = torch.cat((torch.ones(z_base_1.shape[0]),
                                         torch.zeros(z_base_2.shape[0])),0).long().to(device)
                adv_entropy = class_criterion(labels_adv,true_labels)
                _, predicted = torch.max(labels_adv, 1)
                predicted = predicted.cpu().numpy()
                cf_matrix = confusion_matrix(true_labels.cpu().numpy(),predicted)
                tn, fp, fn, tp = cf_matrix.ravel()
                f1_basal = 2*tp/(2*tp+fp+fn)

                cell_labels_adv = cell_adverse_classifier(latent_base_vectors)
                cell_true_labels_adv = torch.cat((conds_human,
                                              conds_mouse), 0).long().to(device) - 1
                adv_cell_entropy = class_criterion(cell_labels_adv,cell_true_labels_adv)
                _, cell_predicted_adv = torch.max(cell_labels_adv, 1)
                cell_predicted_adv = cell_predicted_adv.cpu().numpy()
                cell_basal_f1 = f1_score(cell_true_labels_adv.cpu(), cell_predicted_adv, average='micro')

                # + Vcell.Regularization(model_params['v_reg'])
                #model_params['similarity_reg'] * silimalityLoss -model_params['cosine_reg']*cosineLoss+\
                #Vsp.Regularization(model_params['v_reg']) + Vcell.Regularization(model_params['v_reg'])+\
                #model_params['cosine_reg']*(cosineLossSpecies-cosineLossA)
                #model_params['lambda_mi_loss']*mi_loss_interm+ prior_loss_interm +\
                #encoder_interm_1.L2Regularization(model_params['intermediate_reg']) + encoder_interm_2.L2Regularization(model_params['intermediate_reg'])

                loss =  model_params['reg_recon']*reconstruction_loss_1+ model_params['reg_recon']*reconstruction_loss_2 +\
                        model_params['similarity_reg'] * silimalityLoss -model_params['cosine_reg']*cosineLoss +\
                        model_params['lambda_mi_loss']*mi_loss + prior_loss  +\
                        model_params['reg_state'] * entropy - model_params['reg_adv']*adv_entropy +\
                        model_params['reg_state'] * species_entropy +\
                        model_params['reg_state'] * cell_entropy - model_params['reg_adv']*adv_cell_entropy+\
                        classifier.L2Regularization(model_params['state_class_reg']) + cell_classifier.L2Regularization(model_params['state_class_reg'])+\
                        encoder_human[1].L2Regularization(model_params['enc_l2_reg']) +\
                        encoder_mouse[1].L2Regularization(model_params['enc_l2_reg']) +\
                        decoder_human.L2Regularization(model_params['dec_l2_reg']) + \
                        decoder_mouse.L2Regularization(model_params['dec_l2_reg']) +\
                        model_params['enc_l2_reg'] * (torch.sum(torch.square(encoder_human[0].weight)) + torch.sum(torch.abs(encoder_human[0].bias))) +\
                        model_params['enc_l2_reg'] * (torch.sum(torch.square(encoder_mouse[0].weight)) + torch.sum(torch.abs(encoder_mouse[0].bias))) +\
                        Vsp.Regularization(model_params['v_reg']) + Vcell.Regularization(model_params['v_reg'])

                loss.backward()
                optimizer.step()
                #iteration += 1
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
                    #print2log(logits1)
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

        if model_params['schedule_step_adv'] is not None :
            scheduler_adv.step()
        if (e>0):
            scheduler.step()
            outString = 'Split {:.0f}: Epoch={:.0f}/{:.0f}'.format(fold + 1, e + 1, NUM_EPOCHS)
            outString += ', MI Loss={:.4f}'.format(mi_loss.item())
            outString += ', Prior Loss={:.4f}'.format(prior_loss.item())
            outString += ', Entropy Loss={:.4f}'.format(entropy.item())
            outString += ', Adverse Entropy={:.4f}'.format(adv_entropy.item())
            outString += ', loss={:.4f}'.format(loss.item())
            outString += ', Accuracy_latent={:.4f}'.format(acc)
            outString += ', F1_latent={:.4f}'.format(f1)
            outString += ', F1_species={:.4f}'.format(species_f1)
            #outString += ', F1_cell={:.4f}'.format(cell_f1)
            outString += ', F1 basal={:.4f}'.format(f1_basal)
            outString += ', NBloss_human={:.4f}'.format(reconstruction_loss_1.item())
            outString += ', NBloss_mouse={:.4f}'.format(reconstruction_loss_2.item())
            outString += ', mean_score_human={:.4f}'.format(mean_score_human)
            outString += ', var_score_human={:.4f}'.format(var_score_human)
            outString += ', mean_score_mouse={:.4f}'.format(mean_score_mouse)
            outString += ', var_score_mouse={:.4f}'.format(var_score_mouse)
            outString += ', Cosine similarity={:.4f}'.format(cosineLoss.item())
            outString += ', Latent Cohen`s d={:.4f}'.format(cohen.item())
            outString += ', Base Cohen`s d={:.4f}'.format(cohen_base.item())            
            if f1_basal_trained is not None:
                outString += ', F1 basal trained={:.4f}'.format(f1_basal_trained)
        if ((e%10==0 and e>0) or e==1):
            print2log(outString)
        if e % model_params['adversary_steps'] !=0:
            trainLoss.append(loss.item())
            train_eps.append(e)
    print2log(outString)

    encoder_human.eval()
    encoder_mouse.eval()
    prior_d.eval()
    local_d.eval()
    classifier.eval()
    adverse_classifier.eval()
    species_classifier.eval()
    decoder_human.eval()
    decoder_mouse.eval()
    cell_classifier.eval()
    cell_adverse_classifier.eval()
    Vcell.eval()
    Vsp.eval()
    #encoder_interm_1.eval()
    #encoder_interm_2.eval()
    #prior_d_2.eval()
    #local_d_2.eval()

    X_human = xtest_human.to(device)
    X_mouse = xtest_mouse.to(device)
    Y_human = ytest_human[:,0]
    Y_mouse = ytest_mouse[:,0]
    conds_human = ytest_human[:, 2]
    conds_mouse = ytest_mouse[:, 2]

    z_species_1 = torch.cat((torch.ones(X_human.shape[0],1),
                             torch.zeros(X_human.shape[0],1)),1).to(device)
    z_species_2 = torch.cat((torch.zeros(X_mouse.shape[0],1),
                             torch.ones(X_mouse.shape[0],1)),1).to(device)

    z_cell_1 = torch.cat((1*(conds_human==1).unsqueeze(1),1*(conds_human==2).unsqueeze(1),
                          1*(conds_human==3).unsqueeze(1),1*(conds_human==4).unsqueeze(1),1*(conds_human==5).unsqueeze(1)),1).float().to(device)
    z_cell_2 = torch.cat((1*(conds_mouse==1).unsqueeze(1),1*(conds_mouse==2).unsqueeze(1),
                          1*(conds_mouse==3).unsqueeze(1),1*(conds_mouse==4).unsqueeze(1),1*(conds_mouse==5).unsqueeze(1)),1).float().to(device)

    z_latent_base_1 = encoder_human(torch.log1p(X_human))
    z_latent_base_2 = encoder_mouse(torch.log1p(X_mouse))

    z1 = Vsp(z_latent_base_1,z_species_1) 
    z1 = Vcell(z1,z_cell_1)
    z2 = Vsp(z_latent_base_2,z_species_2) 
    z2 = Vcell(z2,z_cell_2)
    #z1 = encoder_interm_1(z_latent_base_1)
    #z2 = encoder_interm_2(z_latent_base_2)

    #gene_reconstructions_1 = decoder_human(z1)
    gene_means_1 , gene_vars_1 = decoder_human(z1)
    #dim_1 = gene_reconstructions_1.size(1) // 2
    #gene_means_1 = gene_reconstructions_1[:, :dim_1]
    #gene_vars_1 = gene_reconstructions_1[:, dim_1:]
    #gene_reconstructions_2 = decoder_mouse(z2)
    gene_means_2 , gene_vars_2 = decoder_mouse(z2)
    #dim_2 = gene_reconstructions_2.size(1) // 2
    #gene_means_2 = gene_reconstructions_2[:, :dim_2]
    #gene_vars_2 = gene_reconstructions_2[:, dim_2:]

    # Classification
    test_out = classifier(torch.cat((z1, z2), 0))
    true_labels = torch.cat((Y_human,Y_mouse),0).long().to(device)
    _, predicted = torch.max(test_out, 1)
    predicted = predicted.cpu().numpy()
    cf_matrix = confusion_matrix(true_labels.cpu().numpy(),predicted)
    tn, fp, fn, tp = cf_matrix.ravel()
    acc = (tp+tn)/predicted.size
    rec = tp/(tp+fn)
    prec = tp/(tp+fp)
    f1 = 2*tp/(2*tp+fp+fn)

    labels_adv = adverse_classifier(torch.cat((z_latent_base_1, z_latent_base_2), 0))
    true_labels = torch.cat((torch.ones(z_latent_base_1.shape[0]),
                                     torch.zeros(z_latent_base_2.shape[0])),0).long().to(device)
    _, predicted = torch.max(labels_adv, 1)
    predicted = predicted.cpu().numpy()
    cf_matrix = confusion_matrix(true_labels.cpu().numpy(),predicted)
    tn, fp, fn, tp = cf_matrix.ravel()
    f1_basal = 2*tp/(2*tp+fp+fn)

    # Species classification
    species_labels = species_classifier(torch.cat((z1, z2), 0))
    species_true_labels = torch.cat((torch.ones(Y_human.shape[0]),
        torch.zeros(Y_mouse.shape[0])),0).long().to(device)
    _, species_predicted = torch.max(species_labels, 1)
    species_predicted = species_predicted.cpu().numpy()
    cf_matrix = confusion_matrix(species_true_labels.cpu(),species_predicted)
    tn, fp, fn, tp = cf_matrix.ravel()
    species_acc = (tp+tn)/predicted.size
    species_f1 = 2*tp/(2*tp+fp+fn)

    # Cell classification
    cell_labels = cell_classifier(torch.cat((z1, z2), 0))
    cell_true_labels = torch.cat((conds_human,
        conds_mouse), 0).long().to(device) - 1
    _, cell_predicted = torch.max(cell_labels, 1)
    cell_predicted = cell_predicted.cpu().numpy()
    cell_f1 = f1_score(cell_true_labels.cpu(), cell_predicted, average='micro')

    valF1Cell.append(cell_f1)

    valF1Species.append(species_f1)
    valAccSpecies.append(species_acc)

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

    # Translate to other species
    z_species_train_1 = torch.cat((torch.ones(xtrain_human.shape[0],1),
                             torch.zeros(xtrain_human.shape[0],1)),1).to(device)
    z_species_train_2 = torch.cat((torch.zeros(xtrain_mouse.shape[0],1),
                             torch.ones(xtrain_mouse.shape[0],1)),1).to(device)
    z_cell_train_1 = torch.cat((1*(ytrain_human[:,2]==1).unsqueeze(1),1*(ytrain_human[:,2]==2).unsqueeze(1),
                                1*(ytrain_human[:,2]==3).unsqueeze(1),1*(ytrain_human[:,2]==4).unsqueeze(1),1*(ytrain_human[:,2]==5).unsqueeze(1)),1).float().to(device)
    z_cell_train_2 = torch.cat((1*(ytrain_mouse[:,2]==1).unsqueeze(1),1*(ytrain_mouse[:,2]==2).unsqueeze(1),
                                1*(ytrain_mouse[:,2]==3).unsqueeze(1),1*(ytrain_mouse[:,2]==4).unsqueeze(1),1*(ytrain_mouse[:,2]==5).unsqueeze(1)),1).float().to(device)
    z_train_base_1 = torch.zeros(xtrain_human.shape[0],model_params['latent_dim'])
    z_train_base_2 = torch.zeros(xtrain_mouse.shape[0],model_params['latent_dim'])
    z1_train = torch.zeros(xtrain_human.shape[0],model_params['latent_dim'])
    z2_train = torch.zeros(xtrain_mouse.shape[0],model_params['latent_dim'])
    val_bs_1 = 512
    val_bs_2 = 512
    ii=0
    jj=0
    while (ii<xtrain_human.shape[0]):
        if (ii+val_bs_1>xtrain_human.shape[0]):
            z_train_base_1[ii:,:] =  encoder_human(torch.log1p(xtrain_human[ii:,:]).to(device)).cpu()
            z1_train[ii:,:] = Vsp(z_train_base_1[ii:,:].to(device),
                                  z_species_train_1[ii:,:]).cpu()
            z1_train[ii:,:]  = Vcell(z1_train[ii:,:].to(device),z_cell_train_1[ii:,:]).cpu()
            #z1_train[ii:,:] = encoder_interm_1(z_train_base_1[ii:,:].to(device)).cpu()
        else:
            z_train_base_1[ii:(ii+val_bs_1),:] =  encoder_human(torch.log1p(xtrain_human[ii:(ii+val_bs_1),:]).to(device)).cpu()
            z1_train[ii:(ii+val_bs_1),:]  = Vsp(z_train_base_1[ii:(ii+val_bs_1),:].to(device),
                                                z_species_train_1[ii:(ii+val_bs_1),:]).cpu()
            z1_train[ii:(ii+val_bs_1),:]  = Vcell(z1_train[ii:(ii+val_bs_1),:].to(device),z_cell_train_1[ii:(ii+val_bs_1),:]).cpu()
            #z1_train[ii:(ii+val_bs_1),:] = encoder_interm_1(z_train_base_1[ii:(ii+val_bs_1),:].to(device)).cpu()
        ii = ii + val_bs_1

    while (jj<xtrain_mouse.shape[0]):
        if (jj+val_bs_2>xtrain_mouse.shape[0]):
            z_train_base_2[jj:,:] =  encoder_mouse(torch.log1p(xtrain_mouse[jj:,:]).to(device)).cpu()
            z2_train[jj:,:] = Vsp(z_train_base_2[jj:,:].to(device),
                                  z_species_train_2[jj:,:]).cpu()
            z2_train[jj:,:]  = Vcell(z2_train[jj:,:].to(device),z_cell_train_2[jj:,:]).cpu()
            #z2_train[jj:,:] = encoder_interm_2(z_train_base_2[jj:,:].to(device)).cpu()
        else:
            z_train_base_2[jj:(jj+val_bs_2),:] =  encoder_mouse(torch.log1p(xtrain_mouse[jj:(jj+val_bs_2),:]).to(device)).cpu()
            z2_train[jj:(jj+val_bs_2),:] = Vsp(z_train_base_2[jj:(jj+val_bs_2),:].to(device),
                                               z_species_train_2[jj:(jj+val_bs_2),:]).cpu()
            z2_train[jj:(jj+val_bs_2),:]  = Vcell(z2_train[jj:(jj+val_bs_2),:].to(device),z_cell_train_2[jj:(jj+val_bs_2),:]).cpu()
            #z2_train[jj:(jj+val_bs_2),:] = encoder_interm_2(z_train_base_2[jj:(jj+val_bs_2),:].to(device)).cpu()
        jj = jj + val_bs_2

    knn = KNeighborsClassifier(n_neighbors=5,metric = 'cosine')
    knn.fit(torch.cat((z1_train,z2_train),0).detach().numpy(), np.concatenate((np.ones(z1_train.shape[0]),
                                                                     np.zeros(z2_train.shape[0])),0))

    z1_translated = Vsp(z_latent_base_2,1 - z_species_2) 
    z1_translated = Vcell(z1_translated,z_cell_2)
    #z1_translated = encoder_interm_1(z_latent_base_2)
    z2_translated = Vsp(z_latent_base_1,1 - z_species_1) 
    z2_translated = Vcell(z2_translated,z_cell_1)
    #z2_translated = encoder_interm_2(z_latent_base_1)
    y_pred_translated = knn.predict(torch.cat((z1_translated,z2_translated),0).detach().cpu().numpy())
    cf_matrix = confusion_matrix(np.concatenate((np.ones(z1_translated.shape[0]),np.zeros(z2_translated.shape[0])),0),
                                 y_pred_translated)
    tn, fp, fn, tp = cf_matrix.ravel()
    acc_translation = (tp+tn)/y_pred_translated.size
    rec_translation = tp/(tp+fn)
    prec_translation = tp/(tp+fp)
    f1_translation = 2*tp/(2*tp+fp+fn)

    species_labels = species_classifier(torch.cat((z1_translated,z2_translated),0))
    species_true_labels = torch.cat((torch.ones(z1_translated.shape[0]),
        torch.zeros(z2_translated.shape[0])),0).long().to(device)
    _, species_predicted = torch.max(species_labels, 1)
    species_predicted = species_predicted.cpu().numpy()
    cf_matrix = confusion_matrix(species_true_labels.cpu(),species_predicted)
    tn, fp, fn, tp = cf_matrix.ravel()
    species_acc_trans = (tp+tn)/predicted.size
    species_f1_trans = 2*tp/(2*tp+fp+fn)

    #gene_reconstructions_1 = decoder_human(z1_translated)
    y_mu_1 , y_var_1 = decoder_human(z1_translated)
    #dim_1 = gene_reconstructions_1.size(1) // 2
    #y_mu_1 = gene_reconstructions_1[:, :dim_1]
    #y_var_1 = gene_reconstructions_1[:, dim_1:]

    #gene_reconstructions_2 = decoder_mouse(z2_translated)
    y_mu_2 , y_var_2 = decoder_mouse(z2_translated)
    #dim_2 = gene_reconstructions_2.size(1) // 2
    #y_mu_2 = gene_reconstructions_2[:, :dim_2]
    #y_var_2 = gene_reconstructions_2[:, dim_2:]

    if model_params['loss_ae'] == 'nb':
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
        nb_sample = dist1.sample().cpu().numpy()
        yp_m1 = nb_sample.mean(0)
        yp_v1 = nb_sample.var(0)

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
        nb_sample = dist2.sample().cpu().numpy()
        yp_m2 = nb_sample.mean(0)
        yp_v2 = nb_sample.var(0)
    else:
        # predicted means and variances
        yp_m1 = y_mu_1.mean(0)
        yp_v1 = y_var_2.mean(0)
        yp_m2 = y_mu_2.mean(0)
        yp_v2 = y_var_1.mean(0)
        # estimate metrics only for reasonably-sized drug/cell-type combos

    mean_score_trans_2to1.append(r2_score(yt_m1, yp_m1))
    var_score_trans_2to1.append(r2_score(yt_v1, yp_v1))
    mean_score_trans_1to2.append(r2_score(yt_m2, yp_m2))
    var_score_trans_1to2.append(r2_score(yt_v2, yp_v2))

    # Cell classification
    cell_labels = cell_classifier(torch.cat((z1_translated, z2_translated), 0))
    cell_true_labels = torch.cat((conds_human,
                                  conds_mouse), 0).long().to(device) - 1
    _, cell_predicted = torch.max(cell_labels, 1)
    cell_predicted = cell_predicted.cpu().numpy()
    cell_f1_trans = f1_score(cell_true_labels.cpu(), cell_predicted, average='micro')

    valF1CellTrans.append(cell_f1_trans)

    valF1SpeciesTrans.append(species_f1_trans)
    valAccSpeciesTrans.append(species_acc_trans)

    #knn performance in the validation set
    y_pred = knn.predict(torch.cat((z1,z2),0).detach().cpu().numpy())
    cf_matrix = confusion_matrix(np.concatenate((np.ones(z1.shape[0]),np.zeros(z2.shape[0])),0),
                                 y_pred)
    tn, fp, fn, tp = cf_matrix.ravel()
    acc_knn = (tp+tn)/y_pred.size
    rec_knn = tp/(tp+fn)
    prec_knn = tp/(tp+fp)
    f1_knn = 2*tp/(2*tp+fp+fn)

    outString = 'Validation-set performance: Fold={:.0f}'.format(fold)
    outString += ', Accuracy={:.4f}'.format(acc)
    outString += ', F1 score={:.4f}'.format(f1)
    outString += ', Precision={:.4f}'.format(prec)
    outString += ', Recall={:.4f}'.format(rec)
    outString += ', F1 score basal={:.4f}'.format(f1_basal)
    outString += ', Accuracy KNN={:.4f}'.format(acc_knn)
    outString += ', F1 score KNN={:.4f}'.format(f1_knn)
    outString += ', Accuracy translation={:.4f}'.format(acc_translation)
    outString += ', F1 translation={:.4f}'.format(f1_translation)
    outString += ', Accuracy species={:.4f}'.format(species_acc)
    outString += ', F1 species={:.4f}'.format(species_f1)
    outString += ', Accuracy species translation={:.4f}'.format(species_acc_trans)
    outString += ', F1 species translation={:.4f}'.format(species_f1_trans)
    outString += ', r2 mean score human={:.4f}'.format(mean_score_human_val[fold])
    outString += ', r2 mean score mouse={:.4f}'.format(mean_score_mouse_val[fold])
    outString += ', r2 var score human={:.4f}'.format(var_score_human_val[fold])
    outString += ', r2 var score mouse={:.4f}'.format(var_score_mouse_val[fold])
    outString += ', r2 mean score translation human to mouse={:.4f}'.format(r2_score(yt_m2, yp_m2))
    outString += ', r2 mean score translation mouse to human={:.4f}'.format(r2_score(yt_m1, yp_m1))
    outString += ', F1 cell={:.4f}'.format(cell_f1)
    outString += ', F1 cell translation={:.4f}'.format(cell_f1_trans)

    valAcc.append(acc)
    valF1.append(f1)
    valPrec.append(prec)
    valRec.append(rec)
    valF1basal.append(f1_basal)
    valAccTrans.append(acc_translation)
    valF1Trans.append(f1_translation)
    valF1KNN.append(f1_knn)
    valPrecKNN.append(prec_knn)
    valRecKNN.append(rec_knn)
    valAccKNN.append(acc_knn)

    print2log(outString)
    #plt.figure()
    #plt.plot(train_eps,trainLoss)
    #plt.ylim(0,1)
    #plt.yscale('log')
    #plt.show()

    torch.save(encoder_human,'models/encoder_human_%s.pth'%fold)
    torch.save(encoder_mouse,'models/encoder_mouse_%s.pth'%fold)
    torch.save(decoder_human, 'models/decoder_human_%s.pth' % fold)
    torch.save(decoder_mouse, 'models/decoder_mouse_%s.pth' % fold)
    torch.save(prior_d,'models/prior_d_%s.pth'%fold)
    torch.save(local_d,'models/local_d_%s.pth'%fold)
    #torch.save(prior_d_2,'models/prior2_d_%s.pth'%fold)
    #torch.save(local_d_2,'models/local2_d_%s.pth'%fold)
    torch.save(classifier,'models/classifier_disease_%s.pth'%fold)
    torch.save(Vsp,'models/Vspecies_%s.pt'%fold)
    torch.save(Vcell,'models/Vcell_%s.pt'%fold)
    torch.save(adverse_classifier,'models/classifier_adverse_%s.pt'%fold)
    torch.save(cell_adverse_classifier,'models/classifier_cell_adverse_%s.pt'%fold)
    torch.save(cell_classifier, 'models/classifier_cell_%s.pth' % fold)
    torch.save(species_classifier,'models/species_classifier_%s.pt'%fold)
    #torch.save(encoder_interm_1,'models/encoder_interm_human_%s.pt'%fold)
    #torch.save(encoder_interm_2,'models/encoder_interm_mouse_%s.pt'%fold)

    valEmbs_human  = pd.DataFrame(z1.detach().cpu().numpy())
    valEmbs_human['diagnosis'] = Y_human.detach().cpu().numpy()
    valEmbs_human['cell_type'] = ytest_human[:,2]
    valEmbs_mouse = pd.DataFrame(z2.detach().cpu().numpy())
    valEmbs_mouse['diagnosis'] = Y_mouse.detach().cpu().numpy()
    valEmbs_mouse['cell_type'] = ytest_mouse[:,2]

    valEmbs_human.to_csv('results/embs/validation/valEmbs_%s_human.csv'%fold)
    valEmbs_mouse.to_csv('results/embs/validation/valEmbs_%s_mouse.csv'%fold)

    valEmbs_base_human  = pd.DataFrame(z_latent_base_1.detach().cpu().numpy())
    valEmbs_base_human['diagnosis'] = Y_human.detach().cpu().numpy()
    valEmbs_base_human['cell_type'] = ytest_human[:,2]
    valEmbs_base_mouse = pd.DataFrame(z_latent_base_2.detach().cpu().numpy())
    valEmbs_base_mouse['diagnosis'] = Y_mouse.detach().cpu().numpy()
    valEmbs_base_mouse['cell_type'] = ytest_mouse[:,2]

    valEmbs_base_human.to_csv('results/embs/validation/valEmbs_base_%s_human.csv'%fold)
    valEmbs_base_mouse.to_csv('results/embs/validation/valEmbs_base_%s_mouse.csv'%fold)

    trainEmbs_human  = pd.DataFrame(z1_train.detach().cpu().numpy())
    trainEmbs_human['diagnosis'] = ytrain_human[:,0].detach().cpu().numpy()
    trainEmbs_human['cell_type'] = ytrain_human[:,2]
    trainEmbs_mouse = pd.DataFrame(z2_train.detach().cpu().numpy())
    trainEmbs_mouse['diagnosis'] = ytrain_mouse[:,0].detach().cpu().numpy()
    trainEmbs_mouse['cell_type'] = ytrain_mouse[:,2]

    trainEmbs_human.to_csv('results/embs/train/trainEmbs_%s_human.csv'%fold)
    trainEmbs_mouse.to_csv('results/embs/train/trainEmbs_%s_mouse.csv'%fold)

    trainEmbs_base_human  = pd.DataFrame(z_train_base_1.detach().cpu().numpy())
    trainEmbs_base_human['diagnosis'] = ytrain_human[:,0].detach().cpu().numpy()
    trainEmbs_base_human['cell_type'] = ytrain_human[:,2]
    trainEmbs_base_mouse = pd.DataFrame(z_train_base_2.detach().cpu().numpy())
    trainEmbs_base_mouse['diagnosis'] = ytrain_mouse[:,0].cpu().numpy()
    trainEmbs_base_mouse['cell_type'] = ytrain_mouse[:,2]

    trainEmbs_base_human.to_csv('results/embs/train/trainEmbs_base_%s_human.csv'%fold)
    trainEmbs_base_mouse.to_csv('results/embs/train/trainEmbs_base_%s_mouse.csv'%fold)

    valPreds_mu_human = pd.DataFrame(gene_means_1.detach().cpu().numpy())
    valPreds_mu_human['diagnosis'] = Y_human.detach().cpu().numpy()
    valPreds_mu_human['cell_type'] = ytest_human[:, 2]
    valPreds_mu_mouse = pd.DataFrame(gene_means_2.detach().cpu().numpy())
    valPreds_mu_mouse['diagnosis'] = Y_mouse.detach().cpu().numpy()
    valPreds_mu_mouse['cell_type'] = ytest_mouse[:, 2]

    valPreds_mu_human.to_csv('results/preds/validation/valPreds_mu_%s_human.csv' % fold)
    valPreds_mu_mouse.to_csv('results/preds/validation/valPreds_mu_%s_mouse.csv' % fold)

    valPreds_var_human = pd.DataFrame(gene_vars_1.detach().cpu().numpy())
    valPreds_var_human['diagnosis'] = Y_human.detach().cpu().numpy()
    valPreds_var_human['cell_type'] = ytest_human[:, 2]
    valPreds_var_mouse = pd.DataFrame(gene_vars_2.detach().cpu().numpy())
    valPreds_var_mouse['diagnosis'] = Y_mouse.detach().cpu().numpy()
    valPreds_var_mouse['cell_type'] = ytest_mouse[:, 2]

    valPreds_var_human.to_csv('results/preds/validation/valPreds_var_%s_human.csv' % fold)
    valPreds_var_mouse.to_csv('results/preds/validation/valPreds_var_%s_mouse.csv' % fold)


    # In[ ]:
    results = pd.DataFrame({'Accuracy':valAcc,'F1':valF1,'Precision':valPrec,'Recall':valRec,
                        'BasalF1':valF1basal,'TranslationAccuracy':valAccTrans,'TranslationF1':valF1Trans,'F1 cell-type species traslation':valF1CellTrans,'F1 Cell':valF1Cell,
                        'Accuracy Species':valAccSpecies,'F1 Species':valF1Species,'Accuracy species translation':valAccSpeciesTrans,'F1 species traslation':valF1SpeciesTrans,
                        'r2_mu_human':mean_score_human_val,'r2_mu_mouse':mean_score_mouse_val,
                        'r2_var_human':var_score_human_val,'r2_var_mouse':var_score_mouse_val,
                        'r2_mu_human_to_mouse':mean_score_trans_1to2,'r2_mu_mouse_to_human':mean_score_trans_2to1,
                        'r2_var_human_to_mouse':var_score_trans_1to2,'r2_var_mouse_to_human':var_score_trans_2to1})
    results.to_csv('results/10foldvalidationResults_lungs.csv')
print2log(results)
