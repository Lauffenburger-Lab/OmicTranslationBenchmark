import torch
import torch.nn.functional as F
from models import SimpleEncoder,LocalDiscriminator
import math
import numpy as np
import pandas as pd

device = torch.device('cuda')
print(torch.cuda.is_available())
print(device)

# Read data
cmap = pd.read_csv('../preprocessing/preprocessed_data/all_cmap_landmarks.csv',index_col = 0)
gene_size = len(cmap.columns)
dmi = []
for i in range(10):
    # Network
    encoder_1 = torch.load('../results/MI_results/models/CPA_approach/encoder_a375_%s.pt'%i)
    encoder_2 = torch.load('../results/MI_results/models/CPA_approach/encoder_ht29_%s.pt'%i)
    local_d = torch.load('../results/MI_results/models/CPA_approach/localDiscr_%s.pt'%i)

    trainInfo_paired = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/train_paired_%s.csv'%i,index_col=0)
    trainInfo_1 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/train_a375_%s.csv'%i,index_col=0)
    trainInfo_2 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/train_ht29_%s.csv'%i,index_col=0)

    valInfo_paired = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/val_paired_%s.csv'%i,index_col=0)
    valInfo_1 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/val_a375_%s.csv'%i,index_col=0)
    valInfo_2 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/val_ht29_%s.csv'%i,index_col=0)

    encoder_1.eval()
    encoder_2.eval()
    local_d.eval()

    xval_1 = torch.tensor(np.concatenate((cmap.loc[valInfo_paired['sig_id.x']].values,
                                                      cmap.loc[valInfo_1.sig_id].values))).float().to(device)
    xval_2 = torch.tensor(np.concatenate((cmap.loc[valInfo_paired['sig_id.y']].values,
                                                      cmap.loc[valInfo_2.sig_id].values))).float().to(device)

    xtrain_1 = torch.tensor(np.concatenate((cmap.loc[trainInfo_paired['sig_id.x']].values,
                                                  cmap.loc[trainInfo_1.sig_id].values))).float().to(device)
    xtrain_2 = torch.tensor(np.concatenate((cmap.loc[trainInfo_paired['sig_id.y']].values,
                                                      cmap.loc[trainInfo_2.sig_id].values))).float().to(device)

    x_1 = torch.cat((xtrain_1,xval_1),0)
    x_2 = torch.cat((xtrain_2, xval_2), 0)

    conditions = np.concatenate((trainInfo_paired.conditionId.values,
                                 trainInfo_1.conditionId.values,
                                 valInfo_paired.conditionId.values,
                                 valInfo_1.conditionId.values,
                                 trainInfo_paired.conditionId.values,
                                 trainInfo_2.conditionId.values,
                                 valInfo_paired.conditionId.values,
                                 valInfo_2.conditionId.values))
    size = conditions.size
    conditions = conditions.reshape(size, 1)
    conditions = conditions == conditions.transpose()
    conditions = conditions * 1
    mask = torch.tensor(conditions).to(device).detach()
    pos_mask = mask
    neg_mask = 1 - mask
    log_2 = math.log(2.)

    z_latent_1 = encoder_1(x_1)
    z_latent_2 = encoder_2(x_2)
    latent_base_vectors = torch.cat((z_latent_1, z_latent_2), 0)

    z_un = local_d(latent_base_vectors)
    res_un = torch.matmul(z_un, z_un.t())

    MI =  log_2 - F.softplus(- res_un)
    p_samples = res_un * pos_mask.float()
    q_samples = res_un * neg_mask.float()
    Ep = log_2 - F.softplus(- p_samples)
    Eq = F.softplus(-q_samples) + q_samples - log_2
    ### Save MI estimations ###
    positive_mi = pd.DataFrame(Ep.detach().cpu().numpy())
    positive_mi.index = list(trainInfo_paired['sig_id.x']) + list(trainInfo_1.sig_id) + list(
        valInfo_paired['sig_id.x']) + list(
        valInfo_1.sig_id) + list(trainInfo_paired['sig_id.y']) + list(trainInfo_2.sig_id) + list(
        valInfo_paired['sig_id.y']) + list(valInfo_2.sig_id)
    positive_mi.columns = list(trainInfo_paired['sig_id.x']) + list(trainInfo_1.sig_id) + list(
        valInfo_paired['sig_id.x']) + list(
        valInfo_1.sig_id) + list(trainInfo_paired['sig_id.y']) + list(trainInfo_2.sig_id) + list(
        valInfo_paired['sig_id.y']) + list(valInfo_2.sig_id)
    positive_mi.to_csv('../results/MI_results/mi_estimations/CPA/estimated_mi_positives_ht29_a375_fold%s.csv' % i)

    negative_mi = pd.DataFrame(Eq.detach().cpu().numpy())
    negative_mi.index = list(trainInfo_paired['sig_id.x']) + list(trainInfo_1.sig_id) + list(
        valInfo_paired['sig_id.x']) + list(
        valInfo_1.sig_id) + list(trainInfo_paired['sig_id.y']) + list(trainInfo_2.sig_id) + list(
        valInfo_paired['sig_id.y']) + list(valInfo_2.sig_id)
    negative_mi.columns = list(trainInfo_paired['sig_id.x']) + list(trainInfo_1.sig_id) + list(
        valInfo_paired['sig_id.x']) + list(
        valInfo_1.sig_id) + list(trainInfo_paired['sig_id.y']) + list(trainInfo_2.sig_id) + list(
        valInfo_paired['sig_id.y']) + list(valInfo_2.sig_id)
    negative_mi.to_csv('../results/MI_results/mi_estimations/CPA/estimated_mi_negatives_ht29_a375_fold%s.csv' % i)
    
    Ep = (Ep * pos_mask.float()).sum() / pos_mask.float().sum()
    Eq = (Eq * neg_mask.float()).sum() / neg_mask.float().sum()
    # Ep = Ep * pos_mask.float()
    # Eq = Eq * neg_mask.float()
    DMI = Ep - Eq
    dmi.append(DMI.item())

    ### Save MI estimations ###
    mi = pd.DataFrame(MI.detach().cpu().numpy())
    mi.index = list(trainInfo_paired['sig_id.x']) + list(trainInfo_1.sig_id) + list(valInfo_paired['sig_id.x']) + list(
        valInfo_1.sig_id) + list(trainInfo_paired['sig_id.y']) + list(trainInfo_2.sig_id) + list(valInfo_paired['sig_id.y']) + list(valInfo_2.sig_id)
    mi.columns = list(trainInfo_paired['sig_id.x']) + list(trainInfo_1.sig_id) + list(valInfo_paired['sig_id.x']) + list(
        valInfo_1.sig_id) + list(trainInfo_paired['sig_id.y']) + list(trainInfo_2.sig_id) + list(valInfo_paired['sig_id.y']) + list(valInfo_2.sig_id)
    mi.to_csv('../results/MI_results/mi_estimations/CPA/estimated_mi_ht29_a375_fold%s.csv' % i)

    # dmi = pd.DataFrame(DMI.detach().cpu().numpy())
    # #dmi['fold'] = i
    # dmi.to_csv('../results/MI_results/mi_estimations/CPA/DeltaMI_ht29_a375_fold%s.csv'%i)

dmi = pd.DataFrame({'DMI':dmi})
dmi['fold'] = [x for x in range(10)]
dmi.to_csv('../results/MI_results/mi_estimations/CPA/DeltaMI_ht29_a375.csv')

