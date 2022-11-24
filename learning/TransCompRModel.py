import torch
import torch.nn.functional as F
import math
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score,confusion_matrix
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
from models import Decoder
from evaluationUtils import r_square,get_cindex,pearson_r,pseudoAccuracy
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
print2log = logger.info

parser = argparse.ArgumentParser(prog='TransCompR approaches simulation')
parser.add_argument('--gene_space', action='store', default='landmarks')
parser.add_argument('--filter_pcs', action='store', default=True)
args = parser.parse_args()
gene_space = args.gene_space
filter_pcs = args.filter_pcs

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

### Load cell-line data
print2log('Loading data...')
cmap = pd.read_csv('../preprocessing/preprocessed_data/cmap_all_genes_q1_tas03.csv',index_col = 0)
if gene_space=='landmarks':
    lands = pd.read_csv('../preprocessing/preprocessed_data/cmap_landmarks_HT29_A375.csv', index_col=0)
    lands = lands.columns
    cmap = cmap.loc[:, lands]
gene_size = len(cmap.columns)
samples = cmap.index.values

paired = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/alldata/paired_pc3_ha1e.csv',index_col=None)
data_1 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/alldata/pc3_unpaired.csv',index_col=None)
data_2 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/alldata/ha1e_unpaired.csv',index_col=None)

cmap1 = cmap.loc[np.concatenate((paired['sig_id.x'].values,data_1['sig_id'].values)),:]
cmap2 = cmap.loc[np.concatenate((paired['sig_id.y'].values,data_2['sig_id'].values)),:]

### Perform PCA for each each cell-line
print2log('Building PCA space for each cell-line...')
pca = PCA() #n_components=292
pca_space_1 = pca.fit(cmap1)
pca_space_2 = pca.fit(cmap2)
if filter_pcs==True:
    exp_var_pca1 = pca_space_1.explained_variance_ratio_
    cum_sum_eigenvalues_1 = np.cumsum(exp_var_pca1)
    nComps1 = np.min(np.where(cum_sum_eigenvalues_1>=0.99)[0])
    pca1 = PCA(n_components=nComps1)
    pca_space_1 = pca1.fit(cmap1)
    exp_var_pca2 = pca_space_2.explained_variance_ratio_
    cum_sum_eigenvalues_2 = np.cumsum(exp_var_pca2)
    nComps2 = np.min(np.where(cum_sum_eigenvalues_2 >= 0.99)[0])
    pca2 = PCA(n_components=nComps2)
    pca_space_2 = pca2.fit(cmap2)
else:
    nComps1 = pca_transformed_1.n_components_
    nComps2 = pca_transformed_2.n_components_
# nComps1 = 292
# nComps2 = 292

pca_transformed_1 = pca_space_1.transform(cmap1)
pca_transformed_2 = pca_space_2.transform(cmap2)

print2log('Begin TransCompR modeling...')
### Train TransCompR model with Decoder architecture
if gene_space=='landmarks':
    model_params = {'decoder_1_hiddens': [384,640],
                    'decoder_2_hiddens': [384,640], #640, 768
                    'dropout_decoder': 0.2,
                    'decoder_activation': torch.nn.ELU(),
                    'lr': 0.001,
                    'schedule_step': 300,
                    'gamma': 0.8,
                    'batch_size_1': 250, #178,
                    'batch_size_2':  160, #154,
                    'batch_size_paired': 75, #90,
                    'epochs': 1000,
                    'no_folds': 10,
                    'dec_l2_reg': 0.01,
                    'autoencoder_wd': 0}
else:
    model_params = {'decoder_1_hiddens': [2048, 4096],
                    'decoder_2_hiddens': [2048, 4096],
                    'dropout_decoder': 0.2,
                    'decoder_activation': torch.nn.ELU(),
                    'lr': 0.001,
                    'schedule_step': 300,
                    'gamma': 0.8,
                    'batch_size_1': 250, #178,
                    'batch_size_2': 160, #154,
                    'batch_size_paired': 75, #90,
                    'epochs': 1000,
                    'no_folds': 10,
                    'dec_l2_reg': 0.01,
                    'autoencoder_wd': 0}

NUM_EPOCHS= model_params['epochs']
bs_1 = model_params['batch_size_1']
bs_2 =  model_params['batch_size_2']
bs_paired =  model_params['batch_size_paired']

print2log('Training Decoder architecture to predict GeX...')
valR2 = []
valPear = []
valSpear = []
valAccuracy = []

valPear_1 = []
valSpear_1 = []
valAccuracy_1 = []

valPear_2 = []
valSpear_2 = []
valAccuracy_2 = []

print2log('Train decoder for cell-line 2 to translate cell-line 1')
for i in range(model_params["no_folds"]):
    # Network
    decoder_2 = Decoder(nComps2, model_params['decoder_2_hiddens'], gene_size,
                        dropRate=model_params['dropout_decoder'],
                        activation=model_params['decoder_activation']).to(device)

    trainInfo_paired = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/train_paired_%s.csv' % i, index_col=0)
    trainInfo_2 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/train_ha1e_%s.csv' % i, index_col=0)

    N_paired = len(trainInfo_paired)
    N = len(trainInfo_2)

    allParams = list(decoder_2.parameters())
    optimizer = torch.optim.Adam(allParams, lr=model_params['lr'], weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=model_params['schedule_step'],
                                                gamma=model_params['gamma'])
    for e in range(0, NUM_EPOCHS):
        decoder_2.train()
        trainloader_2 = getSamples(N, bs_2)
        len_2 = len(trainloader_2)
        trainloader_paired = getSamples(N_paired, bs_paired)
        len_paired = len(trainloader_paired)

        lens = [len_2, len_paired]
        maxLen = np.max(lens)
        if maxLen > lens[0]:
            trainloader_suppl = getSamples(N, bs_2)
            for jj in range(maxLen - lens[0]):
                trainloader_2.insert(jj, trainloader_suppl[jj])
        if maxLen > lens[1]:
            trainloader_suppl = getSamples(N_paired, bs_paired)
            for jj in range(maxLen - lens[1]):
                trainloader_paired.insert(jj, trainloader_suppl[jj])

        for j in range(maxLen):
            dataIndex_2 = trainloader_2[j]
            dataIndex_paired = trainloader_paired[j]

            df_pairs = trainInfo_paired.iloc[dataIndex_paired, :]
            df_2 = trainInfo_2.iloc[dataIndex_2, :]
            X_1 = cmap.loc[df_pairs['sig_id.x']].values
            X_2 = np.concatenate((cmap.loc[df_pairs['sig_id.y']].values,cmap.loc[df_2.sig_id].values))
            X1_transformed = pca_space_2.transform(cmap.loc[df_pairs['sig_id.x']])
            X2_transformed = pca_space_2.transform(pd.concat((cmap.loc[df_pairs['sig_id.y']],cmap.loc[df_2.sig_id])))
            X_2 = torch.tensor(np.concatenate((cmap.loc[df_pairs['sig_id.y']].values,X_2))).float().to(device)
            z = torch.tensor(np.concatenate((X1_transformed,X2_transformed))).float().to(device)
            optimizer.zero_grad()


            y_pred_2 = decoder_2(z)
            fitLoss = torch.mean(torch.sum((y_pred_2 - X_2) ** 2, dim=1))
            L2Loss = decoder_2.L2Regularization(model_params['dec_l2_reg'])
            loss = fitLoss + L2Loss

            loss.backward()
            optimizer.step()

            pearson = pearson_r(y_pred_2.detach().flatten(), X_2.detach().flatten())
            r2 = r_square(y_pred_2.detach().flatten(), X_2.detach().flatten())
            mse = torch.mean(torch.mean((y_pred_2.detach() - X_2.detach()) ** 2, dim=1))

        scheduler.step()
        outString = 'Split {:.0f}: Epoch={:.0f}/{:.0f}'.format(i + 1, e + 1, NUM_EPOCHS)
        outString += ', r2={:.4f}'.format(r2.item())
        outString += ', pearson={:.4f}'.format(pearson.item())
        outString += ', MSE={:.4f}'.format(mse.item())
        outString += ', loss={:.4f}'.format(loss.item())
        if (e % 200 == 0):
            print2log(outString)
    print2log(outString)
    torch.save(decoder_2, '../results/TransCompR_results/models/decoder_ha1e_%s.pt' % i)

print2log('Train decoder for cell-line 1 to translate cell-line 2')
for i in range(model_params["no_folds"]):
    # Network
    decoder_1 = Decoder(nComps1, model_params['decoder_1_hiddens'], gene_size,
                        dropRate=model_params['dropout_decoder'],
                        activation=model_params['decoder_activation']).to(device)

    trainInfo_paired = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/train_paired_%s.csv' % i, index_col=0)
    trainInfo_1 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/train_pc3_%s.csv' % i, index_col=0)

    N_paired = len(trainInfo_paired)
    N = len(trainInfo_1)

    allParams = list(decoder_1.parameters())
    optimizer = torch.optim.Adam(allParams, lr=model_params['lr'], weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=model_params['schedule_step'],
                                                gamma=model_params['gamma'])
    for e in range(0, NUM_EPOCHS):
        decoder_1.train()
        trainloader_1 = getSamples(N, bs_1)
        len_1 = len(trainloader_1)
        trainloader_paired = getSamples(N_paired, bs_paired)
        len_paired = len(trainloader_paired)

        lens = [len_1, len_paired]
        maxLen = np.max(lens)
        if maxLen > lens[0]:
            trainloader_suppl = getSamples(N, bs_1)
            for jj in range(maxLen - lens[0]):
                trainloader_1.insert(jj, trainloader_suppl[jj])
        if maxLen > lens[1]:
            trainloader_suppl = getSamples(N_paired, bs_paired)
            for jj in range(maxLen - lens[1]):
                trainloader_paired.insert(jj, trainloader_suppl[jj])

        for j in range(maxLen):
            dataIndex_1 = trainloader_1[j]
            dataIndex_paired = trainloader_paired[j]

            df_pairs = trainInfo_paired.iloc[dataIndex_paired, :]
            df_1 = trainInfo_1.iloc[dataIndex_1, :]
            X_2 = cmap.loc[df_pairs['sig_id.y']].values
            X_1 = np.concatenate((cmap.loc[df_pairs['sig_id.x']].values,cmap.loc[df_1.sig_id].values))
            X2_transformed = pca_space_1.transform(cmap.loc[df_pairs['sig_id.y']])
            X1_transformed = pca_space_1.transform(pd.concat((cmap.loc[df_pairs['sig_id.x']],cmap.loc[df_1.sig_id])))
            X_1 = torch.tensor(np.concatenate((cmap.loc[df_pairs['sig_id.x']].values,X_1))).float().to(device)
            z = torch.tensor(np.concatenate((X2_transformed,X1_transformed))).float().to(device)
            optimizer.zero_grad()


            y_pred_1 = decoder_1(z)
            fitLoss = torch.mean(torch.sum((y_pred_1 - X_1) ** 2, dim=1))
            L2Loss = decoder_1.L2Regularization(model_params['dec_l2_reg'])
            loss = fitLoss + L2Loss

            loss.backward()
            optimizer.step()

            pearson = pearson_r(y_pred_1.detach().flatten(), X_1.detach().flatten())
            r2 = r_square(y_pred_1.detach().flatten(), X_1.detach().flatten())
            mse = torch.mean(torch.mean((y_pred_1.detach() - X_1.detach()) ** 2, dim=1))

        scheduler.step()
        outString = 'Split {:.0f}: Epoch={:.0f}/{:.0f}'.format(i + 1, e + 1, NUM_EPOCHS)
        outString += ', r2={:.4f}'.format(r2.item())
        outString += ', pearson={:.4f}'.format(pearson.item())
        outString += ', MSE={:.4f}'.format(mse.item())
        outString += ', loss={:.4f}'.format(loss.item())
        if (e % 200 == 0):
            print2log(outString)
    print2log(outString)
    torch.save(decoder_1, '../results/TransCompR_results/models/decoder_pc3_%s.pt' % i)

print('Evaluate translation using decoders')
for i in range(model_params["no_folds"]):
    decoder_1 = torch.load('../results/TransCompR_results/models/decoder_pc3_%s.pt' % i)
    decoder_2 = torch.load('../results/TransCompR_results/models/decoder_ha1e_%s.pt' % i)
    trainInfo_paired = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/train_paired_%s.csv' % i, index_col=0)
    trainInfo_1 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/train_pc3_%s.csv' % i, index_col=0)
    trainInfo_2 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/train_ha1e_%s.csv' % i, index_col=0)
    valInfo_paired = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/val_paired_%s.csv' % i, index_col=0)
    valInfo_1 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/val_pc3_%s.csv' % i, index_col=0)
    valInfo_2 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/val_ha1e_%s.csv' % i, index_col=0)
    decoder_1.eval()
    decoder_2.eval()
    
    x_1 = cmap.loc[valInfo_paired['sig_id.x']].values
    x_2 = cmap.loc[valInfo_paired['sig_id.y']].values
    x1_transformed = torch.tensor(pca_space_2.transform(cmap.loc[valInfo_paired['sig_id.x']])).float().to(device)
    x2_transformed = torch.tensor(pca_space_1.transform(cmap.loc[valInfo_paired['sig_id.y']])).float().to(device)
    x_1 = torch.tensor(cmap.loc[valInfo_paired['sig_id.x']].values).float().to(device)
    x_2 = torch.tensor(cmap.loc[valInfo_paired['sig_id.y']].values).float().to(device)
    xhat_2 = decoder_2(x1_transformed)
    xhat_1 = decoder_1(x2_transformed)

    x1_all = torch.tensor(np.concatenate((cmap.loc[valInfo_paired['sig_id.x']].values,cmap.loc[valInfo_1['sig_id']].values))).float().to(device)
    x2_all = torch.tensor(np.concatenate((cmap.loc[valInfo_paired['sig_id.y']].values,cmap.loc[valInfo_2['sig_id']].values))).float().to(device)
    ypred_2 = decoder_2(torch.tensor(pca_space_2.transform(pd.concat((cmap.loc[valInfo_paired['sig_id.y']], cmap.loc[valInfo_2['sig_id']])))).float().to(device))
    ypred_1 = decoder_1(torch.tensor(pca_space_1.transform(pd.concat((cmap.loc[valInfo_paired['sig_id.x']], cmap.loc[valInfo_1['sig_id']])))).float().to(device))

    pearson_2 = pearson_r(ypred_2.detach().flatten(),x2_all.detach().flatten())
    rhos = []
    for jj in range(ypred_2.shape[0]):
        rho, p = spearmanr(x2_all[jj, :].detach().cpu().numpy(), ypred_2[jj, :].detach().cpu().numpy())
        rhos.append(rho)
    valSpear_2.append(np.mean(rhos))
    acc = pseudoAccuracy(x2_all.detach().cpu(), ypred_2.detach().cpu(), eps=1e-6)
    valAccuracy_2.append(np.mean(acc))

    pearson_1 = pearson_r(ypred_1.detach().flatten(), x1_all.detach().flatten())
    rhos = []
    for jj in range(ypred_1.shape[0]):
        rho, p = spearmanr(x1_all[jj, :].detach().cpu().numpy(), ypred_1[jj, :].detach().cpu().numpy())
        rhos.append(rho)
    valSpear_1.append(np.mean(rhos))
    acc = pseudoAccuracy(x1_all.detach().cpu(), ypred_1.detach().cpu(), eps=1e-6)
    valAccuracy_1.append(np.mean(acc))

    valPear_1.append(pearson_1.item())
    valPear_2.append(pearson_2.item())
    print2log('Pearson correlation 1: %s' % pearson_1.item())
    print2log('Spearman correlation 1: %s' % valSpear_1[i])
    print2log('Pseudo-Accuracy 1: %s' % valAccuracy_1[i])
    print2log('Pearson correlation 2: %s' % pearson_2.item())
    print2log('Spearman correlation 2: %s' % valSpear_2[i])
    print2log('Pseudo-Accuracy 2: %s' % valAccuracy_2[i])

    pearson_2 = pearson_r(xhat_2.detach().flatten(), x_2.detach().flatten())
    rhos = []
    for jj in range(xhat_2.shape[0]):
        rho, p = spearmanr(x_2[jj, :].detach().cpu().numpy(),
                           xhat_2[jj, :].detach().cpu().numpy())
        rhos.append(rho)
    rho_2 = np.mean(rhos)
    acc_2 = np.mean(pseudoAccuracy(x_2.detach().cpu(), xhat_2.detach().cpu(), eps=1e-6))
    print2log('Pearson correlation 1 to 2: %s' % pearson_2.item())
    print2log('Pseudo accuracy 1 to 2: %s' % acc_2)

    pearson_1 = pearson_r(xhat_1.detach().flatten(), x_1.detach().flatten())
    rhos = []
    for jj in range(xhat_1.shape[0]):
        rho, p = spearmanr(x_1[jj, :].detach().cpu().numpy(),
                           xhat_1[jj, :].detach().cpu().numpy())
        rhos.append(rho)
    rho_1 = np.mean(rhos)
    acc_1 = np.mean(pseudoAccuracy(x_1.detach().cpu(), xhat_1.detach().cpu(), eps=1e-6))
    print2log('Pearson correlation 2 to 1: %s' % pearson_1.item())
    print2log('Pseudo accuracy 2 to 1: %s' % acc_1)

    valPear.append([pearson_2.item(), pearson_1.item()])
    valSpear.append([rho_2, rho_1])
    valAccuracy.append([acc_2, acc_1])

print2log('Summarize validation results')
valPear = np.array(valPear)
valSpear = np.array(valSpear)
valAccuracy= np.array(valAccuracy)
print2log(np.mean(valPear,axis=0))

df_result = pd.DataFrame({'model_pearsonHA1E':valPear[:,0],'model_pearsonPC3':valPear[:,1],
                          'model_spearHA1E':valSpear[:,0],'model_spearPC3':valSpear[:,1],
                          'model_accHA1E':valAccuracy[:,0],'model_accPC3':valAccuracy[:,1],
                          'recon_pear_ha1e':valPear_2 ,'recon_pear_pc3':valPear_1,
                          'recon_spear_ha1e':valSpear_2 ,'recon_spear_pc3':valSpear_1,
                          'recon_acc_ha1e':valAccuracy_2 ,'recon_acc_pc3':valAccuracy_1})
df_result.to_csv('../results/TransCompR_results/'+gene_space+'_10foldvalidation_transcompr_decoders_1000ep512bs_pc3_ha1e.csv')

### Train model with just a transformation matrix
print2log('Training Matrix architecture to predict GeX...')
class MatrixKernel(torch.nn.Module):
    def __init__(self, in_channel, out_channel,dropRate=0.1):
        super(MatrixKernel, self).__init__()

        self.mat = torch.nn.Linear(in_channel, out_channel, bias=False)
        self.bn = torch.nn.BatchNorm1d(num_features=out_channel, momentum=0.6)
        self.dropout = torch.nn.Dropout(dropRate)
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x):
        y = self.dropout(x)
        y = self.mat(y)
        y = self.bn(y)
        return y

    def L2Regularization(self, L2):
        weightLoss = L2 * torch.sum((self.mat.weight)**2)
        #L2Loss = biasLoss + weightLoss
        return(weightLoss)

valR2 = []
valPear = []
valSpear = []
valAccuracy = []

valPear_1 = []
valSpear_1 = []
valAccuracy_1 = []

valPear_2 = []
valSpear_2 = []
valAccuracy_2 = []

print2log('Train Matrix for cell-line 2 to translate cell-line 1')
for i in range(model_params["no_folds"]):
    # Network
    decoder_2 = MatrixKernel(nComps2, gene_size).to(device)

    trainInfo_paired = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/train_paired_%s.csv' % i, index_col=0)
    trainInfo_2 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/train_ha1e_%s.csv' % i, index_col=0)

    N_paired = len(trainInfo_paired)
    N = len(trainInfo_2)

    allParams = list(decoder_2.parameters())
    optimizer = torch.optim.Adam(allParams, lr=model_params['lr'], weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=model_params['schedule_step'],
                                                gamma=model_params['gamma'])
    for e in range(0, NUM_EPOCHS):
        decoder_2.train()
        trainloader_2 = getSamples(N, bs_2)
        len_2 = len(trainloader_2)
        trainloader_paired = getSamples(N_paired, bs_paired)
        len_paired = len(trainloader_paired)

        lens = [len_2, len_paired]
        maxLen = np.max(lens)
        if maxLen > lens[0]:
            trainloader_suppl = getSamples(N, bs_2)
            for jj in range(maxLen - lens[0]):
                trainloader_2.insert(jj, trainloader_suppl[jj])
        if maxLen > lens[1]:
            trainloader_suppl = getSamples(N_paired, bs_paired)
            for jj in range(maxLen - lens[1]):
                trainloader_paired.insert(jj, trainloader_suppl[jj])

        for j in range(maxLen):
            dataIndex_2 = trainloader_2[j]
            dataIndex_paired = trainloader_paired[j]

            df_pairs = trainInfo_paired.iloc[dataIndex_paired, :]
            df_2 = trainInfo_2.iloc[dataIndex_2, :]
            X_1 = cmap.loc[df_pairs['sig_id.x']].values
            X_2 = np.concatenate((cmap.loc[df_pairs['sig_id.y']].values,cmap.loc[df_2.sig_id].values))
            X1_transformed = pca_space_2.transform(cmap.loc[df_pairs['sig_id.x']])
            X2_transformed = pca_space_2.transform(pd.concat((cmap.loc[df_pairs['sig_id.y']],cmap.loc[df_2.sig_id])))
            X_2 = torch.tensor(np.concatenate((cmap.loc[df_pairs['sig_id.y']].values,X_2))).float().to(device)
            z = torch.tensor(np.concatenate((X1_transformed,X2_transformed))).float().to(device)
            optimizer.zero_grad()


            y_pred_2 = decoder_2(z)
            fitLoss = torch.mean(torch.sum((y_pred_2 - X_2) ** 2, dim=1))
            L2Loss = decoder_2.L2Regularization(model_params['dec_l2_reg'])
            loss = fitLoss + L2Loss

            loss.backward()
            optimizer.step()

            pearson = pearson_r(y_pred_2.detach().flatten(), X_2.detach().flatten())
            r2 = r_square(y_pred_2.detach().flatten(), X_2.detach().flatten())
            mse = torch.mean(torch.mean((y_pred_2.detach() - X_2.detach()) ** 2, dim=1))

        scheduler.step()
        outString = 'Split {:.0f}: Epoch={:.0f}/{:.0f}'.format(i + 1, e + 1, NUM_EPOCHS)
        outString += ', r2={:.4f}'.format(r2.item())
        outString += ', pearson={:.4f}'.format(pearson.item())
        outString += ', MSE={:.4f}'.format(mse.item())
        outString += ', loss={:.4f}'.format(loss.item())
        if (e % 200 == 0):
            print2log(outString)
    print2log(outString)
    torch.save(decoder_2, '../results/TransCompR_results/models/MatrixKernel_ha1e_%s.pt' % i)

print2log('Train Matrix for cell-line 1 to translate cell-line 2')
for i in range(model_params["no_folds"]):
    # Network
    decoder_1 = MatrixKernel(nComps1, gene_size).to(device)

    trainInfo_paired = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/train_paired_%s.csv' % i, index_col=0)
    trainInfo_1 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/train_pc3_%s.csv' % i, index_col=0)

    N_paired = len(trainInfo_paired)
    N = len(trainInfo_1)

    allParams = list(decoder_1.parameters())
    optimizer = torch.optim.Adam(allParams, lr=model_params['lr'], weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=model_params['schedule_step'],
                                                gamma=model_params['gamma'])
    for e in range(0, NUM_EPOCHS):
        decoder_1.train()
        trainloader_1 = getSamples(N, bs_1)
        len_1 = len(trainloader_1)
        trainloader_paired = getSamples(N_paired, bs_paired)
        len_paired = len(trainloader_paired)

        lens = [len_1, len_paired]
        maxLen = np.max(lens)
        if maxLen > lens[0]:
            trainloader_suppl = getSamples(N, bs_1)
            for jj in range(maxLen - lens[0]):
                trainloader_1.insert(jj, trainloader_suppl[jj])
        if maxLen > lens[1]:
            trainloader_suppl = getSamples(N_paired, bs_paired)
            for jj in range(maxLen - lens[1]):
                trainloader_paired.insert(jj, trainloader_suppl[jj])

        for j in range(maxLen):
            dataIndex_1 = trainloader_1[j]
            dataIndex_paired = trainloader_paired[j]

            df_pairs = trainInfo_paired.iloc[dataIndex_paired, :]
            df_1 = trainInfo_1.iloc[dataIndex_1, :]
            X_2 = cmap.loc[df_pairs['sig_id.y']].values
            X_1 = np.concatenate((cmap.loc[df_pairs['sig_id.x']].values,cmap.loc[df_1.sig_id].values))
            X2_transformed = pca_space_1.transform(cmap.loc[df_pairs['sig_id.y']])
            X1_transformed = pca_space_1.transform(pd.concat((cmap.loc[df_pairs['sig_id.x']],cmap.loc[df_1.sig_id])))
            X_1 = torch.tensor(np.concatenate((cmap.loc[df_pairs['sig_id.x']].values,X_1))).float().to(device)
            z = torch.tensor(np.concatenate((X2_transformed,X1_transformed))).float().to(device)
            optimizer.zero_grad()


            y_pred_1 = decoder_1(z)
            fitLoss = torch.mean(torch.sum((y_pred_1 - X_1) ** 2, dim=1))
            L2Loss = decoder_1.L2Regularization(model_params['dec_l2_reg'])
            loss = fitLoss + L2Loss

            loss.backward()
            optimizer.step()

            pearson = pearson_r(y_pred_1.detach().flatten(), X_1.detach().flatten())
            r2 = r_square(y_pred_1.detach().flatten(), X_1.detach().flatten())
            mse = torch.mean(torch.mean((y_pred_1.detach() - X_1.detach()) ** 2, dim=1))

        scheduler.step()
        outString = 'Split {:.0f}: Epoch={:.0f}/{:.0f}'.format(i + 1, e + 1, NUM_EPOCHS)
        outString += ', r2={:.4f}'.format(r2.item())
        outString += ', pearson={:.4f}'.format(pearson.item())
        outString += ', MSE={:.4f}'.format(mse.item())
        outString += ', loss={:.4f}'.format(loss.item())
        if (e % 200 == 0):
            print2log(outString)
    print2log(outString)
    torch.save(decoder_1, '../results/TransCompR_results/models/MatrixKernel_pc3_%s.pt' % i)

print('Evaluate translation using matrix kernel')
for i in range(model_params["no_folds"]):
    decoder_1 = torch.load('../results/TransCompR_results/models/MatrixKernel_pc3_%s.pt' % i)
    decoder_2 = torch.load('../results/TransCompR_results/models/MatrixKernel_ha1e_%s.pt' % i)
    trainInfo_paired = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/train_paired_%s.csv' % i, index_col=0)
    trainInfo_1 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/train_pc3_%s.csv' % i, index_col=0)
    trainInfo_2 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/train_ha1e_%s.csv' % i, index_col=0)
    valInfo_paired = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/val_paired_%s.csv' % i, index_col=0)
    valInfo_1 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/val_pc3_%s.csv' % i, index_col=0)
    valInfo_2 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/val_ha1e_%s.csv' % i, index_col=0)
    decoder_1.eval()
    decoder_2.eval()
    
    x_1 = cmap.loc[valInfo_paired['sig_id.x']].values
    x_2 = cmap.loc[valInfo_paired['sig_id.y']].values
    x1_transformed = torch.tensor(pca_space_2.transform(cmap.loc[valInfo_paired['sig_id.x']])).float().to(device)
    x2_transformed = torch.tensor(pca_space_1.transform(cmap.loc[valInfo_paired['sig_id.y']])).float().to(device)
    x_1 = torch.tensor(cmap.loc[valInfo_paired['sig_id.x']].values).float().to(device)
    x_2 = torch.tensor(cmap.loc[valInfo_paired['sig_id.y']].values).float().to(device)
    xhat_2 = decoder_2(x1_transformed)
    xhat_1 = decoder_1(x2_transformed)

    x1_all = torch.tensor(np.concatenate((cmap.loc[valInfo_paired['sig_id.x']].values,cmap.loc[valInfo_1['sig_id']].values))).float().to(device)
    x2_all = torch.tensor(np.concatenate((cmap.loc[valInfo_paired['sig_id.y']].values,cmap.loc[valInfo_2['sig_id']].values))).float().to(device)
    ypred_2 = decoder_2(torch.tensor(pca_space_2.transform(pd.concat((cmap.loc[valInfo_paired['sig_id.y']], cmap.loc[valInfo_2['sig_id']])))).float().to(device))
    ypred_1 = decoder_1(torch.tensor(pca_space_1.transform(pd.concat((cmap.loc[valInfo_paired['sig_id.x']], cmap.loc[valInfo_1['sig_id']])))).float().to(device))

    pearson_2 = pearson_r(ypred_2.detach().flatten(),x2_all.detach().flatten())
    rhos = []
    for jj in range(ypred_2.shape[0]):
        rho, p = spearmanr(x2_all[jj, :].detach().cpu().numpy(), ypred_2[jj, :].detach().cpu().numpy())
        rhos.append(rho)
    valSpear_2.append(np.mean(rhos))
    acc = pseudoAccuracy(x2_all.detach().cpu(), ypred_2.detach().cpu(), eps=1e-6)
    valAccuracy_2.append(np.mean(acc))

    pearson_1 = pearson_r(ypred_1.detach().flatten(), x1_all.detach().flatten())
    rhos = []
    for jj in range(ypred_1.shape[0]):
        rho, p = spearmanr(x1_all[jj, :].detach().cpu().numpy(), ypred_1[jj, :].detach().cpu().numpy())
        rhos.append(rho)
    valSpear_1.append(np.mean(rhos))
    acc = pseudoAccuracy(x1_all.detach().cpu(), ypred_1.detach().cpu(), eps=1e-6)
    valAccuracy_1.append(np.mean(acc))

    valPear_1.append(pearson_1.item())
    valPear_2.append(pearson_2.item())
    print2log('Pearson correlation 1: %s' % pearson_1.item())
    print2log('Spearman correlation 1: %s' % valSpear_1[i])
    print2log('Pseudo-Accuracy 1: %s' % valAccuracy_1[i])
    print2log('Pearson correlation 2: %s' % pearson_2.item())
    print2log('Spearman correlation 2: %s' % valSpear_2[i])
    print2log('Pseudo-Accuracy 2: %s' % valAccuracy_2[i])

    pearson_2 = pearson_r(xhat_2.detach().flatten(), x_2.detach().flatten())
    rhos = []
    for jj in range(xhat_2.shape[0]):
        rho, p = spearmanr(x_2[jj, :].detach().cpu().numpy(),
                           xhat_2[jj, :].detach().cpu().numpy())
        rhos.append(rho)
    rho_2 = np.mean(rhos)
    acc_2 = np.mean(pseudoAccuracy(x_2.detach().cpu(), xhat_2.detach().cpu(), eps=1e-6))
    print2log('Pearson correlation 1 to 2: %s' % pearson_2.item())
    print2log('Pseudo accuracy 1 to 2: %s' % acc_2)

    pearson_1 = pearson_r(xhat_1.detach().flatten(), x_1.detach().flatten())
    rhos = []
    for jj in range(xhat_1.shape[0]):
        rho, p = spearmanr(x_1[jj, :].detach().cpu().numpy(),
                           xhat_1[jj, :].detach().cpu().numpy())
        rhos.append(rho)
    rho_1 = np.mean(rhos)
    acc_1 = np.mean(pseudoAccuracy(x_1.detach().cpu(), xhat_1.detach().cpu(), eps=1e-6))
    print2log('Pearson correlation 2 to 1: %s' % pearson_1.item())
    print2log('Pseudo accuracy 2 to 1: %s' % acc_1)

    valPear.append([pearson_2.item(), pearson_1.item()])
    valSpear.append([rho_2, rho_1])
    valAccuracy.append([acc_2, acc_1])

print2log('Summarize validation results')
valPear = np.array(valPear)
valSpear = np.array(valSpear)
valAccuracy= np.array(valAccuracy)
print2log(np.mean(valPear,axis=0))

df_result = pd.DataFrame({'model_pearsonHA1E':valPear[:,0],'model_pearsonPC3':valPear[:,1],
                          'model_spearHA1E':valSpear[:,0],'model_spearPC3':valSpear[:,1],
                          'model_accHA1E':valAccuracy[:,0],'model_accPC3':valAccuracy[:,1],
                          'recon_pear_ha1e':valPear_2 ,'recon_pear_pc3':valPear_1,
                          'recon_spear_ha1e':valSpear_2 ,'recon_spear_pc3':valSpear_1,
                          'recon_acc_ha1e':valAccuracy_2 ,'recon_acc_pc3':valAccuracy_1})
df_result.to_csv('../results/TransCompR_results/'+gene_space+'_10foldvalidation_transcompr_matrixKernel_1000ep512bs_pc3_ha1e.csv')