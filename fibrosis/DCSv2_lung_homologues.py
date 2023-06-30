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
# pre_encoder_human = torch.load(pre_models/encoder_human_homologues_6.pth')
# pre_encoder_mouse = torch.load(pre_models/encoder_mouse_homologues_6.pth')
# pre_prior_d = torch.load(pre_models/prior_d_homologues_6.pth')
# pre_local_d = torch.load(pre_models/local_d_homologues_6.pth')
# pre_classifier = torch.load(pre_models/classifier_disease_homologues_6.pth')
# pre_species_classifier = torch.load(pre_models/species_classifier_homologues_6.pt')
# pre_cell_classifier = torch.load(pre_models/classifier_cell_homologues_6.pth')
# pre_adverse_classifier = torch.load(pre_models/classifier_adverse_homologues_6.pt')
# pre_cell_adverse_classifier = torch.load(pre_models/classifier_cell_adverse_homologues_6.pt')
# pre_Vsp = torch.load(pre_models/Vspecies_homologues_6.pt')
# pre_Vcell = torch.load(pre_models/Vcell_homologues_6.pt')


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
    results = pd.read_csv('results/10foldvalidationResults_lungs_DCS_homologues.csv',index_col=0)
    mean_score_human_val = results['r2_mu_human'].tolist()
    mean_score_mouse_val = results['r2_mu_mouse'].tolist()
    var_score_human_val = results['r2_var_human'].tolist()
    var_score_mouse_val = results['r2_var_mouse'].tolist()
    mean_score_trans_1to2 = results['r2_mu_human_to_mouse'].tolist()
    mean_score_trans_2to1 = results['r2_mu_mouse_to_human'].tolist()
    var_score_trans_1to2 = results['r2_var_human_to_mouse'].tolist()
    var_score_trans_2to1 = results['r2_var_mouse_to_human'].tolist()

#pretrained_classifier = torch.load('pretrained_classifier.pth')
# Load the homologues map
homologues_map = pd.read_csv('results/HumanMouseHomologuesMap.csv',index_col=0)

for fold in range(begin_fold,model_params['no_folds']):
    pre_master_encoder = torch.load(pre_models/master_encoder_homologues_%s.pth'% fold)

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

    # keep only hologues
    xtrain_mouse = xtrain_mouse[:, homologues_map.mouse_id.values]
    xtest_mouse = xtest_mouse[:, homologues_map.mouse_id.values]
    xtrain_human = xtrain_human[:, homologues_map.human_id.values]
    xtest_human = xtest_human[:, homologues_map.human_id.values]

    gene_size_mouse = xtrain_mouse.shape[1]
    gene_size_human = xtrain_human.shape[1]

    #print2log(gene_size_human)
    #print2log(pre_encoder_human)

    N_2 = ytrain_mouse.shape[0]
    N_1 = ytrain_human.shape[0]

    N = N_1
    if N_2 > N:
        N = N_2
    
    # Network
    master_encoder = torch.nn.Sequential(ElementWiseLinear(gene_size_human),
                                         SimpleEncoder(gene_size_human,
                                                       model_params['encoder_1_hiddens'],
                                                       model_params['latent_dim'],
                                                       dropRate=model_params['dropout_encoder'],
                                                       activation=model_params['encoder_activation'],
                                                       normalizeOutput=False)).to(device)
    master_encoder.load_state_dict(pre_master_encoder.state_dict())
    decoder_human = VarDecoder(model_params['latent_dim'],model_params['decoder_1_hiddens'],gene_size_human,
                           dropRate=model_params['dropout_decoder'],
                           activation=model_params['decoder_activation']).to(device)
    decoder_mouse = VarDecoder(model_params['latent_dim'],model_params['decoder_2_hiddens'],gene_size_mouse,
                           dropRate=model_params['dropout_decoder'],
                           activation=model_params['decoder_activation']).to(device)


    allParams = list(master_encoder.parameters())
    allParams = list(decoder_human.parameters()) + list(decoder_mouse.parameters())
    optimizer = torch.optim.Adam(allParams, lr= model_params['encoding_lr'], weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=model_params['schedule_step_enc'],
                                                gamma=model_params['gamma_enc'])
    trainLoss = []
    train_eps = []
    f1_basal_trained = None
    print2log('Begin training model %s'%fold)
    for e in range(0, NUM_EPOCHS):
        master_encoder.train()
        decoder_human.train()
        decoder_mouse.train()

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

            z_1 = master_encoder(torch.log1p(X_human))
            z_2 = master_encoder(torch.log1p(X_mouse))
            latent_vectors = torch.cat((z_1, z_2), 0)

            gene_means_1 , gene_vars_1 = decoder_human(z_1)
            reconstruction_loss_1 = recon_criterion(gene_means_1, X_human, gene_vars_1)

            gene_means_2 , gene_vars_2 = decoder_mouse(z_2)
            reconstruction_loss_2 = recon_criterion(gene_means_2, X_mouse, gene_vars_2)

            silimalityLoss = torch.sum(torch.cdist(latent_vectors,latent_vectors) * mask.float())/(mask.float().sum())
            w1 = latent_vectors.norm(p=2, dim=1, keepdim=True)
            w2 = latent_vectors.norm(p=2, dim=1, keepdim=True)
            cosineLoss = torch.mm(latent_vectors, latent_vectors.t()) / (w1 * w2.t()).clamp(min=1e-6)
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

            L1Loss = 1e-7 * (torch.mean(torch.sum(torch.abs(z_1), dim=1)) + torch.mean(torch.sum(torch.abs(z_2), dim=1)))

            loss =  model_params['reg_recon']*reconstruction_loss_1+ model_params['reg_recon']*reconstruction_loss_2 +\
                    model_params['similarity_reg'] * silimalityLoss -model_params['cosine_reg']*cosineLoss +\
                    master_encoder[1].L2Regularization(model_params['enc_l2_reg']) +\
                    decoder_human.L2Regularization(model_params['dec_l2_reg']) + \
                    decoder_mouse.L2Regularization(model_params['dec_l2_reg']) +\
                    model_params['enc_l2_reg'] * (torch.sum(torch.square(master_encoder[0].weight)) + torch.sum(torch.abs(master_encoder[0].bias))) + \
                    L1Loss

            loss.backward()
            optimizer.step()
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
                # print2log(logits1)
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

        scheduler.step()
        outString = 'Split {:.0f}: Epoch={:.0f}/{:.0f}'.format(fold + 1, e + 1, NUM_EPOCHS)
        outString += ', loss={:.4f}'.format(loss.item())
        outString += ', NBloss_human={:.4f}'.format(reconstruction_loss_1.item())
        outString += ', NBloss_mouse={:.4f}'.format(reconstruction_loss_2.item())
        outString += ', mean_score_human={:.4f}'.format(mean_score_human)
        outString += ', var_score_human={:.4f}'.format(var_score_human)
        outString += ', mean_score_mouse={:.4f}'.format(mean_score_mouse)
        outString += ', var_score_mouse={:.4f}'.format(var_score_mouse)
        outString += ', Cosine similarity={:.4f}'.format(cosineLoss.item())
        outString += ', Latent Cohen`s d={:.4f}'.format(cohen.item())
        outString += ', Base Cohen`s d={:.4f}'.format(cohen_base.item())
        if e % 20 == 0:
            print2log(outString)
    print2log(outString)

    master_encoder.eval()
    decoder_human.eval()
    decoder_mouse.eval()

    X_human = xtest_human.to(device)
    X_mouse = xtest_mouse.to(device)
    Y_human = ytest_human[:,0]
    Y_mouse = ytest_mouse[:,0]
    conds_human = ytest_human[:, 2]
    conds_mouse = ytest_mouse[:, 2]

    z1 = master_encoder(torch.log1p(X_human))
    z2 = master_encoder(torch.log1p(X_mouse))

    gene_means_1 , gene_vars_1 = decoder_human(z1)
    gene_means_2 , gene_vars_2 = decoder_mouse(z2)

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

    y_mu_1 , y_var_1 = decoder_human(z2)
    y_mu_2 , y_var_2 = decoder_mouse(z1)

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

    outString = 'Validation-set performance: Fold={:.0f}'.format(fold)
    outString += ', r2 mean score human={:.4f}'.format(mean_score_human_val[fold])
    outString += ', r2 mean score mouse={:.4f}'.format(mean_score_mouse_val[fold])
    outString += ', r2 var score human={:.4f}'.format(var_score_human_val[fold])
    outString += ', r2 var score mouse={:.4f}'.format(var_score_mouse_val[fold])
    outString += ', r2 mean score translation human to mouse={:.4f}'.format(r2_score(yt_m2, yp_m2))
    outString += ', r2 mean score translation mouse to human={:.4f}'.format(r2_score(yt_m1, yp_m1))

    print2log(outString)
    #plt.figure()
    #plt.plot(train_eps,trainLoss)
    #plt.ylim(0,1)
    #plt.yscale('log')
    #plt.show()

    torch.save(master_encoder,'models/retrained_master_encoder_homologues_%s.pth'%fold)
    torch.save(decoder_human, 'models/DCS_decoder_human_homologues_%s.pth' % fold)
    torch.save(decoder_mouse, 'models/DCS_decoder_mouse_homologues_%s.pth' % fold)


    # In[ ]:
    results = pd.DataFrame({'r2_mu_human':mean_score_human_val,'r2_mu_mouse':mean_score_mouse_val,
                        'r2_var_human':var_score_human_val,'r2_var_mouse':var_score_mouse_val,
                        'r2_mu_human_to_mouse':mean_score_trans_1to2,'r2_mu_mouse_to_human':mean_score_trans_2to1,
                        'r2_var_human_to_mouse':var_score_trans_1to2,'r2_var_mouse_to_human':var_score_trans_2to1})
    results.to_csv('results/10foldvalidationResults_lungs_DCS_homologues.csv')
print2log(results)
