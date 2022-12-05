import torch
import torch.nn.functional as F
import math
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score,confusion_matrix
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
from trainingUtils import MultipleOptimizer, MultipleScheduler, compute_kernel, compute_mmd
from models import Decoder, SimpleEncoder,LocalDiscriminator,PriorDiscriminator,Classifier,SpeciesCovariate
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
# Gradient penalnty function
def compute_gradients(output, input):
    grads = torch.autograd.grad(output, input, create_graph=True)
    grads = grads[0].pow(2).mean()
    return grads
### Load cell-line data
print2log('Loading data...')
cmap = pd.read_csv('../preprocessing/preprocessed_data/cmap_all_genes_q1_tas03.csv',index_col = 0)
if gene_space=='landmarks':
    lands = pd.read_csv('../preprocessing/preprocessed_data/cmap_landmarks_HT29_A375.csv', index_col=0)
    lands = lands.columns
    cmap = cmap.loc[:, lands]
gene_size = len(cmap.columns)
samples = cmap.index.values

paired = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/alldata/paired_a375_ht29.csv',index_col=None)
data_1 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/alldata/a375_unpaired.csv',index_col=None)
data_2 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/alldata/ht29_unpaired.csv',index_col=None)

cmap1 = cmap.loc[np.concatenate((paired['sig_id.x'].values,data_1['sig_id'].values)),:]
cmap2 = cmap.loc[np.concatenate((paired['sig_id.y'].values,data_2['sig_id'].values)),:]

print2log('Begin TransCompR modeling...')
### Train TransCompR model with Decoder architecture
if gene_space=='landmarks':
    model_params = {'encoder_1_hiddens':[640,384],
                    'encoder_2_hiddens':[640,384],
                    'decoder_1_hiddens': [384,640],
                    'decoder_2_hiddens': [384,640], #640, 768
                    'latent_dim': 292,
                    'dropout_decoder': 0.2,
                    'dropout_encoder':0.1,
                    'encoder_activation':torch.nn.ELU(),
                    'decoder_activation':torch.nn.ELU(),
                    'V_dropout':0.25,
                    'state_class_hidden':[256,128,64],
                    'state_class_drop_in':0.5,
                    'state_class_drop':0.25,
                    'no_states':2,
                    'adv_class_hidden':[256,128,64],
                    'adv_class_drop_in':0.3,
                    'adv_class_drop':0.1,
                    'no_adv_class':2,
                    'encoding_lr':0.001,
                    'adv_lr':0.001,
                    'schedule_step_adv':200,
                    'gamma_adv':0.5,
                    'schedule_step_enc':200,
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
                    'effsize_reg': 100,
                    'cosine_loss': 10,
                    'adv_penalnty':100,
                    'reg_adv':1000,
                    'reg_classifier': 1000,
                    'similarity_reg' : 10,
                    'adversary_steps':4,
                    'autoencoder_wd': 0.,
                    'adversary_wd': 0.}
else:
    model_params = {'encoder_1_hiddens':[4096,2048],
                    'encoder_2_hiddens':[4096,2048],
                    'decoder_1_hiddens': [2048, 4096],
                    'decoder_2_hiddens': [2048, 4096],
                    'latent_dim': 1024,
                    'dropout_decoder': 0.2,
                    'dropout_encoder':0.1,
                    'encoder_activation':torch.nn.ELU(),
                    'decoder_activation':torch.nn.ELU(),
                    'V_dropout':0.25,
                    'state_class_hidden':[256,128,64],
                    'state_class_drop_in':0.5,
                    'state_class_drop':0.25,
                    'no_states':2,
                    'adv_class_hidden':[256,128,64],
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
                    'autoencoder_wd': 0.,
                    'adversary_wd': 0.}

### Perform PCA for each each cell-line
print2log('Building PCA space for each cell-line...')
pca = PCA(n_components=model_params['latent_dim'])
pca_space_1 = pca.fit(cmap1)
pca_space_2 = pca.fit(cmap2)
nComps1 = model_params['latent_dim']
nComps2 = model_params['latent_dim']
pca_transformed_1 = pca_space_1.transform(cmap1)
pca_transformed_2 = pca_space_2.transform(cmap2)

### Begin pre-training decoder
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

print2log('Train decoder for cell-line 2')
for i in range(model_params["no_folds"]):
    # Network
    decoder_2 = Decoder(nComps2, model_params['decoder_2_hiddens'], gene_size,
                        dropRate=model_params['dropout_decoder'],
                        activation=model_params['decoder_activation']).to(device)

    trainInfo_paired = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/train_paired_%s.csv' % i, index_col=0)
    trainInfo_2 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/train_ht29_%s.csv' % i, index_col=0)

    N_paired = len(trainInfo_paired)
    N = len(trainInfo_2)

    allParams = list(decoder_2.parameters())
    optimizer = torch.optim.Adam(allParams, lr=model_params['encoding_lr'], weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=model_params['schedule_step_enc'],
                                                gamma=model_params['gamma_enc'])
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
            X_2 = np.concatenate((cmap.loc[df_pairs['sig_id.y']].values,cmap.loc[df_2.sig_id].values))
            X2_transformed = pca_space_2.transform(pd.concat((cmap.loc[df_pairs['sig_id.y']],cmap.loc[df_2.sig_id])))
            X_2 = torch.tensor(X_2).float().to(device)
            z = torch.tensor(X2_transformed).float().to(device)
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
    torch.save(decoder_2, '../results/CPAmixedTransCompR_results/pretrained_models/decoder_ht29_%s.pt' % i)

print2log('Train decoder for cell-line 1')
for i in range(model_params["no_folds"]):
    # Network
    decoder_1 = Decoder(nComps1, model_params['decoder_1_hiddens'], gene_size,
                        dropRate=model_params['dropout_decoder'],
                        activation=model_params['decoder_activation']).to(device)

    trainInfo_paired = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/train_paired_%s.csv' % i, index_col=0)
    trainInfo_1 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/train_a375_%s.csv' % i, index_col=0)

    N_paired = len(trainInfo_paired)
    N = len(trainInfo_1)

    allParams = list(decoder_1.parameters())
    optimizer = torch.optim.Adam(allParams, lr=model_params['encoding_lr'], weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=model_params['schedule_step_enc'],
                                                gamma=model_params['gamma_enc'])
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
            X_1 = np.concatenate((cmap.loc[df_pairs['sig_id.x']].values,cmap.loc[df_1.sig_id].values))
            X1_transformed = pca_space_1.transform(pd.concat((cmap.loc[df_pairs['sig_id.x']],cmap.loc[df_1.sig_id])))
            X_1 = torch.tensor(X_1).float().to(device)
            z = torch.tensor(X1_transformed).float().to(device)
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
    torch.save(decoder_1, '../results/CPAmixedTransCompR_results/pretrained_models/decoder_a375_%s.pt' % i)

print2log('Evaluate translation using decoders')
for i in range(model_params["no_folds"]):
    decoder_1 = torch.load('../results/CPAmixedTransCompR_results/pretrained_models/decoder_a375_%s.pt' % i)
    decoder_2 = torch.load('../results/CPAmixedTransCompR_results/pretrained_models/decoder_ht29_%s.pt' % i)
    trainInfo_paired = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/train_paired_%s.csv' % i, index_col=0)
    trainInfo_1 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/train_a375_%s.csv' % i, index_col=0)
    trainInfo_2 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/train_ht29_%s.csv' % i, index_col=0)
    valInfo_paired = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/val_paired_%s.csv' % i, index_col=0)
    valInfo_1 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/val_a375_%s.csv' % i, index_col=0)
    valInfo_2 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/val_ht29_%s.csv' % i, index_col=0)
    decoder_1.eval()
    decoder_2.eval()

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

df_result = pd.DataFrame({'recon_pear_ht29':valPear_2 ,'recon_pear_a375':valPear_1,
                          'recon_spear_ht29':valSpear_2 ,'recon_spear_a375':valSpear_1,
                          'recon_acc_ht29':valAccuracy_2 ,'recon_acc_a375':valAccuracy_1})
df_result.to_csv('../results/CPAmixedTransCompR_results/'+gene_space+'_10foldvalidation_pretrained_decoders_1000ep512bs_a375_ht29.csv')

## Train encoders
print2log('Training encoder architecture to predict PCA...')
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
print2log('Train encoder for cell-line 2')
for i in range(model_params["no_folds"]):
    # Network
    encoder_2 = SimpleEncoder(gene_size, model_params['encoder_2_hiddens'], nComps2,
                        dropRate=model_params['dropout_encoder'],
                        activation=model_params['encoder_activation']).to(device)
    Vsp = SpeciesCovariate(2, model_params['latent_dim'], dropRate=model_params['V_dropout']).to(device)

    trainInfo_paired = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/train_paired_%s.csv' % i,
                                   index_col=0)
    trainInfo_2 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/train_ht29_%s.csv' % i,
                              index_col=0)

    N_paired = len(trainInfo_paired)
    N = len(trainInfo_2)

    allParams = list(encoder_2.parameters()) + list(Vsp.parameters())
    optimizer = torch.optim.Adam(allParams, lr=model_params['encoding_lr'], weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=model_params['schedule_step_enc'],
                                                gamma=model_params['gamma_enc'])
    for e in range(0, NUM_EPOCHS):
        encoder_2.train()
        Vsp.train()
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
            X_2 = np.concatenate((cmap.loc[df_pairs['sig_id.y']].values, cmap.loc[df_2.sig_id].values))
            X2_transformed = pca_space_2.transform(pd.concat((cmap.loc[df_pairs['sig_id.y']], cmap.loc[df_2.sig_id])))
            X_2 = torch.tensor(X_2).float().to(device)
            X2_transformed = torch.tensor(X2_transformed).float().to(device)
            z_species_2 = torch.cat((torch.zeros(X_2.shape[0], 1),
                                     torch.ones(X_2.shape[0], 1)), 1).to(device)
            optimizer.zero_grad()

            z_base_2 = encoder_2(X_2)
            y_pred_2 = Vsp(z_base_2, z_species_2)
            # y_pred_2 = encoder_2(X_2)
            fitLoss = torch.mean(torch.sum((y_pred_2 - X2_transformed) ** 2, dim=1))
            L2Loss = encoder_2.L2Regularization(model_params['dec_l2_reg'])
            loss = fitLoss + L2Loss

            loss.backward()
            optimizer.step()

            pearson = pearson_r(y_pred_2.detach().flatten(), X2_transformed.detach().flatten())
            r2 = r_square(y_pred_2.detach().flatten(), X2_transformed.detach().flatten())
            mse = torch.mean(torch.mean((y_pred_2.detach() - X2_transformed.detach()) ** 2, dim=1))

        scheduler.step()
        outString = 'Split {:.0f}: Epoch={:.0f}/{:.0f}'.format(i + 1, e + 1, NUM_EPOCHS)
        outString += ', r2={:.4f}'.format(r2.item())
        outString += ', pearson={:.4f}'.format(pearson.item())
        outString += ', MSE={:.4f}'.format(mse.item())
        outString += ', loss={:.4f}'.format(loss.item())
        if (e % 200 == 0):
            print2log(outString)
    print2log(outString)
    torch.save(encoder_2, '../results/CPAmixedTransCompR_results/pretrained_models/encoder_ht29_%s.pt' % i)
    torch.save(Vsp, '../results/CPAmixedTransCompR_results/pretrained_models/pre_trained_Vsp_%s.pt' % i)

print2log('Train encoder for cell-line 1')
for i in range(model_params["no_folds"]):
    # Network
    encoder_1 = SimpleEncoder(gene_size, model_params['encoder_1_hiddens'], nComps1,
                        dropRate=model_params['dropout_encoder'],
                        activation=model_params['encoder_activation']).to(device)
    Vsp = SpeciesCovariate(2, model_params['latent_dim'], dropRate=model_params['V_dropout']).to(device)
    pretrained_Vsp = torch.load('../results/CPAmixedTransCompR_results/pretrained_models/pre_trained_Vsp_%s.pt' % i)
    Vsp.load_state_dict(pretrained_Vsp.state_dict())

    trainInfo_paired = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/train_paired_%s.csv' % i,
                                   index_col=0)
    trainInfo_1 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/train_a375_%s.csv' % i,
                              index_col=0)

    N_paired = len(trainInfo_paired)
    N = len(trainInfo_1)

    allParams = list(encoder_1.parameters()) + list(Vsp.parameters())
    optimizer = torch.optim.Adam(allParams, lr=model_params['encoding_lr'], weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=model_params['schedule_step_enc'],
                                                gamma=model_params['gamma_enc'])
    for e in range(0, NUM_EPOCHS):
        encoder_1.train()
        Vsp.train()
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
            X_1 = np.concatenate((cmap.loc[df_pairs['sig_id.x']].values, cmap.loc[df_1.sig_id].values))
            X1_transformed = pca_space_1.transform(pd.concat((cmap.loc[df_pairs['sig_id.x']], cmap.loc[df_1.sig_id])))
            X_1 = torch.tensor(X_1).float().to(device)
            X1_transformed = torch.tensor(X1_transformed).float().to(device)
            z_species_1 = torch.cat((torch.ones(X_1.shape[0], 1),
                                     torch.zeros(X_1.shape[0], 1)), 1).to(device)
            optimizer.zero_grad()

            z_base_1 = encoder_1(X_1)
            y_pred_1 = Vsp(z_base_1, z_species_1)
            # y_pred_1 = encoder_1(X_1)
            fitLoss = torch.mean(torch.sum((y_pred_1 - X1_transformed) ** 2, dim=1))
            L2Loss = encoder_1.L2Regularization(model_params['dec_l2_reg'])
            loss = fitLoss + L2Loss

            loss.backward()
            optimizer.step()

            pearson = pearson_r(y_pred_1.detach().flatten(), X1_transformed.detach().flatten())
            r2 = r_square(y_pred_1.detach().flatten(), X1_transformed.detach().flatten())
            mse = torch.mean(torch.mean((y_pred_1.detach() - X1_transformed.detach()) ** 2, dim=1))

        scheduler.step()
        outString = 'Split {:.0f}: Epoch={:.0f}/{:.0f}'.format(i + 1, e + 1, NUM_EPOCHS)
        outString += ', r2={:.4f}'.format(r2.item())
        outString += ', pearson={:.4f}'.format(pearson.item())
        outString += ', MSE={:.4f}'.format(mse.item())
        outString += ', loss={:.4f}'.format(loss.item())
        if (e % 200 == 0):
            print2log(outString)
    print2log(outString)
    torch.save(encoder_1, '../results/CPAmixedTransCompR_results/pretrained_models/encoder_a375_%s.pt' % i)
    torch.save(Vsp,'../results/CPAmixedTransCompR_results/pretrained_models/pre_trained_Vsp_%s.pt' % i)

print2log('Evaluate translation using encoders')
for i in range(model_params["no_folds"]):
    encoder_1 = torch.load('../results/CPAmixedTransCompR_results/pretrained_models/encoder_a375_%s.pt' % i)
    encoder_2 = torch.load('../results/CPAmixedTransCompR_results/pretrained_models/encoder_ht29_%s.pt' % i)
    Vsp = torch.load('../results/CPAmixedTransCompR_results/pretrained_models/pre_trained_Vsp_%s.pt' % i)

    trainInfo_paired = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/train_paired_%s.csv' % i,
                                   index_col=0)
    trainInfo_1 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/train_a375_%s.csv' % i,
                              index_col=0)
    trainInfo_2 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/train_ht29_%s.csv' % i,
                              index_col=0)
    valInfo_paired = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/val_paired_%s.csv' % i,
                                 index_col=0)
    valInfo_1 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/val_a375_%s.csv' % i,
                            index_col=0)
    valInfo_2 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/val_ht29_%s.csv' % i,
                            index_col=0)
    encoder_1.eval()
    encoder_2.eval()
    Vsp.eval()

    x1_transformed = torch.tensor(pca_space_1.transform(pd.concat((cmap.loc[valInfo_paired['sig_id.x']], cmap.loc[valInfo_1['sig_id']])))).float().to(device)
    x2_transformed = torch.tensor(pca_space_2.transform(pd.concat((cmap.loc[valInfo_paired['sig_id.y']], cmap.loc[valInfo_2['sig_id']])))).float().to(device)

    x1_all = torch.tensor(
        np.concatenate((cmap.loc[valInfo_paired['sig_id.x']].values, cmap.loc[valInfo_1['sig_id']].values))).float().to(
        device)
    x2_all = torch.tensor(
        np.concatenate((cmap.loc[valInfo_paired['sig_id.y']].values, cmap.loc[valInfo_2['sig_id']].values))).float().to(
        device)
    z_species_1 = torch.cat((torch.ones(x1_all.shape[0], 1),
                             torch.zeros(x1_all.shape[0], 1)), 1).to(device)
    z_species_2 = torch.cat((torch.zeros(x2_all.shape[0], 1),
                             torch.ones(x2_all.shape[0], 1)), 1).to(device)
    z_base_1 = encoder_1(x1_all)
    ypred_1 = Vsp(z_base_1, z_species_1)
    z_base_2 = encoder_1(x2_all)
    ypred_2 = Vsp(z_base_2, z_species_2)
    # ypred_2 = encoder_2(x2_all)
    # ypred_1 = encoder_1(x1_all)

    pearson_2 = pearson_r(ypred_2.detach().flatten(), x2_transformed.detach().flatten())
    rhos = []
    for jj in range(ypred_2.shape[0]):
        rho, p = spearmanr(x2_transformed[jj, :].detach().cpu().numpy(), ypred_2[jj, :].detach().cpu().numpy())
        rhos.append(rho)
    valSpear_2.append(np.mean(rhos))
    acc = pseudoAccuracy(x2_transformed.detach().cpu(), ypred_2.detach().cpu(), eps=1e-6)
    valAccuracy_2.append(np.mean(acc))

    pearson_1 = pearson_r(ypred_1.detach().flatten(), x1_transformed.detach().flatten())
    rhos = []
    for jj in range(ypred_1.shape[0]):
        rho, p = spearmanr(x1_transformed[jj, :].detach().cpu().numpy(), ypred_1[jj, :].detach().cpu().numpy())
        rhos.append(rho)
    valSpear_1.append(np.mean(rhos))
    acc = pseudoAccuracy(x1_transformed.detach().cpu(), ypred_1.detach().cpu(), eps=1e-6)
    valAccuracy_1.append(np.mean(acc))

    valPear_1.append(pearson_1.item())
    valPear_2.append(pearson_2.item())
    print2log('Pearson correlation 1: %s' % pearson_1.item())
    print2log('Spearman correlation 1: %s' % valSpear_1[i])
    print2log('Pseudo-Accuracy 1: %s' % valAccuracy_1[i])
    print2log('Pearson correlation 2: %s' % pearson_2.item())
    print2log('Spearman correlation 2: %s' % valSpear_2[i])
    print2log('Pseudo-Accuracy 2: %s' % valAccuracy_2[i])

df_result = pd.DataFrame({'recon_pear_ht29':valPear_2 ,'recon_pear_a375':valPear_1,
                          'recon_spear_ht29':valSpear_2 ,'recon_spear_a375':valSpear_1,
                          'recon_acc_ht29':valAccuracy_2 ,'recon_acc_a375':valAccuracy_1})
df_result.to_csv('../results/CPAmixedTransCompR_results/'+gene_space+'_10foldvalidation_pretrained_encoders_1000ep512bs_a375_ht29.csv')

## Pre-train adverse classifier
print2log('Pre-train adverse classifier')
class_criterion = torch.nn.CrossEntropyLoss()
for i in range(model_params["no_folds"]):
    # Network
    pre_encoder_1 = torch.load('../results/CPAmixedTransCompR_results/pretrained_models/encoder_a375_%s.pt'%i)
    pre_encoder_2 = torch.load('../results/CPAmixedTransCompR_results/pretrained_models/encoder_ht29_%s.pt' % i)
    prior_d = PriorDiscriminator(model_params['latent_dim']).to(device)
    local_d = LocalDiscriminator(model_params['latent_dim'], model_params['latent_dim']).to(device)
    adverse_classifier = Classifier(in_channel=model_params['latent_dim'],
                                    hidden_layers=model_params['adv_class_hidden'],
                                    num_classes=model_params['no_adv_class'],
                                    drop_in=0.5,
                                    drop=0.3).to(device)

    trainInfo_paired = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/train_paired_%s.csv' % i, index_col=0)
    trainInfo_1 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/train_a375_%s.csv' % i, index_col=0)
    trainInfo_2 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/train_ht29_%s.csv' % i, index_col=0)

    valInfo_paired = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/val_paired_%s.csv' % i, index_col=0)
    valInfo_1 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/val_a375_%s.csv' % i, index_col=0)
    valInfo_2 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/val_ht29_%s.csv' % i, index_col=0)

    N_paired = len(trainInfo_paired)
    N_1 = len(trainInfo_1)
    N_2 = len(trainInfo_2)
    N = N_1
    if N_2 > N:
        N = N_2

    allParams = list(prior_d.parameters()) + list(local_d.parameters())
    allParams = allParams + list(adverse_classifier.parameters())
    optimizer_adv = torch.optim.Adam(allParams, lr=model_params['encoding_lr'], weight_decay=0)
    scheduler_adv = torch.optim.lr_scheduler.StepLR(optimizer_adv,
                                                    step_size=model_params['schedule_step_enc'],
                                                    gamma=model_params['gamma_enc'])
    for e in range(0, NUM_EPOCHS):
        pre_encoder_1.eval()
        pre_encoder_2.eval()
        prior_d.train()
        local_d.train()
        adverse_classifier.train()

        trainloader_1 = getSamples(N_1, bs_1)
        len_1 = len(trainloader_1)
        trainloader_2 = getSamples(N_2, bs_2)
        len_2 = len(trainloader_2)
        trainloader_paired = getSamples(N_paired, bs_paired)
        len_paired = len(trainloader_paired)

        lens = [len_1, len_2, len_paired]
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

        if maxLen > lens[2]:
            trainloader_suppl = getSamples(N_paired, bs_paired)
            for jj in range(maxLen - lens[2]):
                trainloader_paired.insert(jj, trainloader_suppl[jj])

        for j in range(maxLen):
            dataIndex_1 = trainloader_1[j]
            dataIndex_2 = trainloader_2[j]
            dataIndex_paired = trainloader_paired[j]

            df_pairs = trainInfo_paired.iloc[dataIndex_paired, :]
            df_1 = trainInfo_1.iloc[dataIndex_1, :]
            df_2 = trainInfo_2.iloc[dataIndex_2, :]
            paired_inds = len(df_pairs)

            X_1 = torch.tensor(np.concatenate((cmap.loc[df_pairs['sig_id.x']].values,
                                               cmap.loc[df_1.sig_id].values))).float().to(device)
            X_2 = torch.tensor(np.concatenate((cmap.loc[df_pairs['sig_id.y']].values,
                                               cmap.loc[df_2.sig_id].values))).float().to(device)

            conditions = np.concatenate((df_pairs.conditionId.values,
                                         df_1.conditionId.values,
                                         df_pairs.conditionId.values,
                                         df_2.conditionId.values))
            size = conditions.size
            conditions = conditions.reshape(size, 1)
            conditions = conditions == conditions.transpose()
            conditions = conditions * 1
            mask = torch.tensor(conditions).to(device).detach()
            pos_mask = mask
            neg_mask = 1 - mask
            log_2 = math.log(2.)
            optimizer_adv.zero_grad()

            z_base_1 = pre_encoder_1(X_1)
            z_base_2 = pre_encoder_2(X_2)

            latent_base_vectors = torch.cat((z_base_1, z_base_2), 0)
            z_un = local_d(latent_base_vectors)
            res_un = torch.matmul(z_un, z_un.t())

            p_samples = res_un * pos_mask.float()
            q_samples = res_un * neg_mask.float()
            Ep = log_2 - F.softplus(- p_samples)
            Eq = F.softplus(-q_samples) + q_samples - log_2
            Ep = (Ep * pos_mask.float()).sum() / pos_mask.float().sum()
            Eq = (Eq * neg_mask.float()).sum() / neg_mask.float().sum()
            mi_loss = Eq - Ep
            prior = torch.rand_like(latent_base_vectors)
            term_a = torch.log(prior_d(prior)).mean()
            term_b = torch.log(1.0 - prior_d(latent_base_vectors)).mean()
            prior_loss = -(term_a + term_b) * model_params['prior_beta']

            # Remove signal from z_basal
            labels_adv = adverse_classifier(latent_base_vectors)
            true_labels = torch.cat((torch.ones(z_base_1.shape[0]),
                                     torch.zeros(z_base_2.shape[0])), 0).long().to(device)
            adv_entropy = class_criterion(labels_adv, true_labels)
            _, predicted = torch.max(labels_adv, 1)
            predicted = predicted.cpu().numpy()
            cf_matrix = confusion_matrix(true_labels.cpu().numpy(), predicted)
            tn, fp, fn, tp = cf_matrix.ravel()
            f1_basal = 2 * tp / (2 * tp + fp + fn)

            loss = mi_loss + prior_loss + adv_entropy + adverse_classifier.L2Regularization(model_params['state_class_reg'])

            loss.backward()
            optimizer_adv.step()

        if (e >= 0):
            scheduler_adv.step()
            outString = 'Split {:.0f}: Epoch={:.0f}/{:.0f}'.format(i + 1, e + 1, NUM_EPOCHS)
            outString += ', MI Loss={:.4f}'.format(mi_loss.item())
            outString += ', Prior Loss={:.4f}'.format(prior_loss.item())
            outString += ', Adverse Entropy={:.4f}'.format(adv_entropy.item())
            outString += ', loss={:.4f}'.format(loss.item())
            outString += ', F1 basal={:.4f}'.format(f1_basal)
        if (e % 50 == 0):
            print2log(outString)
    print2log(outString)
    pre_encoder_1.eval()
    pre_encoder_2.eval()
    prior_d.eval()
    local_d.eval()
    adverse_classifier.eval()

    paired_val_inds = len(valInfo_paired)
    x_1 = torch.tensor(np.concatenate((cmap.loc[valInfo_paired['sig_id.x']].values,
                                       cmap.loc[valInfo_1.sig_id].values))).float().to(device)
    x_2 = torch.tensor(np.concatenate((cmap.loc[valInfo_paired['sig_id.y']].values,
                                       cmap.loc[valInfo_2.sig_id].values))).float().to(device)

    z_latent_base_1 = pre_encoder_1(x_1)
    z_latent_base_2 = pre_encoder_2(x_2)

    labels = adverse_classifier(torch.cat((z_latent_base_1, z_latent_base_2), 0))
    true_labels = torch.cat((torch.ones(z_latent_base_1.shape[0]).view(z_latent_base_1.shape[0], 1),
                             torch.zeros(z_latent_base_2.shape[0]).view(z_latent_base_2.shape[0], 1)), 0).long()
    _, predicted = torch.max(labels, 1)
    predicted = predicted.cpu().numpy()
    cf_matrix = confusion_matrix(true_labels.numpy(), predicted)
    tn, fp, fn, tp = cf_matrix.ravel()
    class_acc = (tp + tn) / predicted.size
    f1 = 2 * tp / (2 * tp + fp + fn)

    print2log('Classification accuracy: %s' % class_acc)
    print2log('Classification F1 score: %s' % f1)

    torch.save(adverse_classifier, '../results/CPAmixedTransCompR_results/pretrained_models/pre_trained_classifier_adverse_%s.pt' % i)


### Train whole translational model
print2log('Train translation model')
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

valPearDirect = []
valSpearDirect = []
valAccDirect = []

valF1 = []
valClassAcc = []

#Reduce epochs and sceduler step
NUM_EPOCHS = int(NUM_EPOCHS/2)
model_params['epochs'] = NUM_EPOCHS
model_params['schedule_step_enc'] = int(model_params['schedule_step_enc']/2)

pretrained_adv_class = torch.load('../results/CPAmixedTransCompR_results/pretrained_models/pre_trained_classifier_adverse_0.pt')
for i in range(model_params["no_folds"]):
    # Network
    pre_encoder_1 = torch.load('../results/CPAmixedTransCompR_results/pretrained_models/encoder_a375_%s.pt'%i)
    pre_encoder_2 = torch.load('../results/CPAmixedTransCompR_results/pretrained_models/encoder_ht29_%s.pt' % i)
    pre_decoder_1 = torch.load('../results/CPAmixedTransCompR_results/pretrained_models/decoder_a375_%s.pt' % i)
    pre_decoder_2 = torch.load('../results/CPAmixedTransCompR_results/pretrained_models/decoder_ht29_%s.pt' % i)

    decoder_1 = Decoder(model_params['latent_dim'], model_params['decoder_1_hiddens'], gene_size,
                        dropRate=model_params['dropout_decoder'],
                        activation=model_params['decoder_activation']).to(device)
    decoder_1.load_state_dict(pre_decoder_1.state_dict())
    decoder_2 = Decoder(model_params['latent_dim'], model_params['decoder_2_hiddens'], gene_size,
                        dropRate=model_params['dropout_decoder'],
                        activation=model_params['decoder_activation']).to(device)
    decoder_2.load_state_dict(pre_decoder_2.state_dict())
    encoder_1 = SimpleEncoder(gene_size, model_params['encoder_1_hiddens'], model_params['latent_dim'],
                              dropRate=model_params['dropout_encoder'],
                              activation=model_params['encoder_activation']).to(device)
    encoder_1.load_state_dict(pre_encoder_1.state_dict())
    encoder_2 = SimpleEncoder(gene_size, model_params['encoder_2_hiddens'], model_params['latent_dim'],
                              dropRate=model_params['dropout_encoder'],
                              activation=model_params['encoder_activation']).to(device)
    encoder_2.load_state_dict(pre_encoder_2.state_dict())
    prior_d = PriorDiscriminator(model_params['latent_dim']).to(device)
    local_d = LocalDiscriminator(model_params['latent_dim'], model_params['latent_dim']).to(device)

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
    adverse_classifier.load_state_dict(pretrained_adv_class.state_dict())

    Vsp = SpeciesCovariate(2, model_params['latent_dim'], dropRate=model_params['V_dropout']).to(device)
    pretrained_Vsp = torch.load('../results/CPAmixedTransCompR_results/pretrained_models/pre_trained_Vsp_%s.pt'%i)
    Vsp.load_state_dict(pretrained_Vsp.state_dict())

    trainInfo_paired = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/train_paired_%s.csv' % i, index_col=0)
    trainInfo_1 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/train_a375_%s.csv' % i, index_col=0)
    trainInfo_2 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/train_ht29_%s.csv' % i, index_col=0)

    valInfo_paired = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/val_paired_%s.csv' % i, index_col=0)
    valInfo_1 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/val_a375_%s.csv' % i, index_col=0)
    valInfo_2 = pd.read_csv('../preprocessing/preprocessed_data/10fold_validation_spit/val_ht29_%s.csv' % i, index_col=0)

    N_paired = len(trainInfo_paired)
    N_1 = len(trainInfo_1)
    N_2 = len(trainInfo_2)
    N = N_1
    if N_2 > N:
        N = N_2

    allParams = list(encoder_1.parameters()) + list(encoder_2.parameters())
    allParams = allParams + list(decoder_1.parameters()) + list(decoder_2.parameters())
    allParams = allParams + list(prior_d.parameters()) + list(local_d.parameters())
    allParams = allParams + list(classifier.parameters())
    allParams = allParams + list(Vsp.parameters())
    optimizer = torch.optim.Adam(allParams, lr=model_params['encoding_lr'], weight_decay=0)
    optimizer_adv = torch.optim.Adam(adverse_classifier.parameters(), lr=model_params['adv_lr'], weight_decay=0)
    if model_params['schedule_step_adv'] is not None:
        scheduler_adv = torch.optim.lr_scheduler.StepLR(optimizer_adv,
                                                        step_size=model_params['schedule_step_adv'],
                                                        gamma=model_params['gamma_adv'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=model_params['schedule_step_enc'],
                                                gamma=model_params['gamma_enc'])
    for e in range(0, NUM_EPOCHS):
        # decoder_1.train()
        # decoder_2.train()
        decoder_1.train()
        decoder_2.train()
        encoder_1.train()
        encoder_2.train()
        prior_d.train()
        local_d.train()
        classifier.train()
        adverse_classifier.train()
        Vsp.train()

        trainloader_1 = getSamples(N_1, bs_1)
        len_1 = len(trainloader_1)
        trainloader_2 = getSamples(N_2, bs_2)
        len_2 = len(trainloader_2)
        trainloader_paired = getSamples(N_paired, bs_paired)
        len_paired = len(trainloader_paired)

        lens = [len_1, len_2, len_paired]
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

        if maxLen > lens[2]:
            trainloader_suppl = getSamples(N_paired, bs_paired)
            for jj in range(maxLen - lens[2]):
                trainloader_paired.insert(jj, trainloader_suppl[jj])

        for j in range(maxLen):
            dataIndex_1 = trainloader_1[j]
            dataIndex_2 = trainloader_2[j]
            dataIndex_paired = trainloader_paired[j]

            df_pairs = trainInfo_paired.iloc[dataIndex_paired, :]
            df_1 = trainInfo_1.iloc[dataIndex_1, :]
            df_2 = trainInfo_2.iloc[dataIndex_2, :]
            paired_inds = len(df_pairs)

            X_1 = torch.tensor(np.concatenate((cmap.loc[df_pairs['sig_id.x']].values,
                                               cmap.loc[df_1.sig_id].values))).float().to(device)
            X_2 = torch.tensor(np.concatenate((cmap.loc[df_pairs['sig_id.y']].values,
                                               cmap.loc[df_2.sig_id].values))).float().to(device)

            z_species_1 = torch.cat((torch.ones(X_1.shape[0], 1),
                                     torch.zeros(X_1.shape[0], 1)), 1).to(device)
            z_species_2 = torch.cat((torch.zeros(X_2.shape[0], 1),
                                     torch.ones(X_2.shape[0], 1)), 1).to(device)

            conditions = np.concatenate((df_pairs.conditionId.values,
                                         df_1.conditionId.values,
                                         df_pairs.conditionId.values,
                                         df_2.conditionId.values))
            size = conditions.size
            conditions = conditions.reshape(size, 1)
            conditions = conditions == conditions.transpose()
            conditions = conditions * 1
            mask = torch.tensor(conditions).to(device).detach()
            pos_mask = mask
            neg_mask = 1 - mask
            log_2 = math.log(2.)
            optimizer.zero_grad()
            optimizer_adv.zero_grad()

            # if e % model_params['adversary_steps']==0:
            z_base_1 = encoder_1(X_1)
            z_base_2 = encoder_2(X_2)
            latent_base_vectors = torch.cat((z_base_1, z_base_2), 0)
            labels_adv = adverse_classifier(latent_base_vectors)
            true_labels = torch.cat((torch.ones(z_base_1.shape[0]),
                                     torch.zeros(z_base_2.shape[0])), 0).long().to(device)
            _, predicted = torch.max(labels_adv, 1)
            predicted = predicted.cpu().numpy()
            cf_matrix = confusion_matrix(true_labels.cpu().numpy(), predicted)
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

            z_1 = Vsp(z_base_1, z_species_1)
            z_2 = Vsp(z_base_2, z_species_2)
            latent_vectors = torch.cat((z_1, z_2), 0)

            # y_pred_1 = decoder_1(z_1)
            y_pred_1 = decoder_1(z_1)
            fitLoss_1 = torch.mean(torch.sum((y_pred_1 - X_1) ** 2, dim=1))
            L2Loss_1 = encoder_1.L2Regularization(model_params['enc_l2_reg']) # + decoder_1.L2Regularization(model_params['dec_l2_reg'])
            loss_1 = fitLoss_1 + L2Loss_1

            # y_pred_2 = decoder_2(z_2)
            y_pred_2 = decoder_2(z_2)
            fitLoss_2 = torch.mean(torch.sum((y_pred_2 - X_2) ** 2, dim=1))
            L2Loss_2 = encoder_2.L2Regularization(model_params['enc_l2_reg']) # + decoder_2.L2Regularization(model_params['dec_l2_reg'])
            loss_2 = fitLoss_2 + L2Loss_2

            silimalityLoss = torch.mean(
                torch.sum((z_base_1[0:paired_inds, :] - z_base_2[0:paired_inds, :]) ** 2, dim=-1))
            cosineLoss = torch.nn.functional.cosine_similarity(z_base_1[0:paired_inds, :], z_base_2[0:paired_inds, :],
                                                               dim=-1).mean()
            # silimalityLoss = torch.mean(
            #     torch.sum((z_1[0:paired_inds, :] - z_2[0:paired_inds, :]) ** 2, dim=-1))
            # cosineLoss = torch.nn.functional.cosine_similarity(z_1[0:paired_inds, :], z_2[0:paired_inds, :],
            #                                                    dim=-1).mean()

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
            true_labels = torch.cat((torch.ones(z_1.shape[0]),
                                     torch.zeros(z_2.shape[0])), 0).long().to(device)
            entropy = class_criterion(labels, true_labels)
            _, predicted = torch.max(labels, 1)
            predicted = predicted.cpu().numpy()
            cf_matrix = confusion_matrix(true_labels.cpu().numpy(), predicted)
            tn, fp, fn, tp = cf_matrix.ravel()
            f1_latent = 2 * tp / (2 * tp + fp + fn)

            # Remove signal from z_basal
            labels_adv = adverse_classifier(latent_base_vectors)
            true_labels = torch.cat((torch.ones(z_base_1.shape[0]),
                                     torch.zeros(z_base_2.shape[0])), 0).long().to(device)
            adv_entropy = class_criterion(labels_adv, true_labels)
            _, predicted = torch.max(labels_adv, 1)
            predicted = predicted.cpu().numpy()
            cf_matrix = confusion_matrix(true_labels.cpu().numpy(), predicted)
            tn, fp, fn, tp = cf_matrix.ravel()
            f1_basal = 2 * tp / (2 * tp + fp + fn)

            # loss = loss_1 + loss_2 + model_params['similarity_reg'] * silimalityLoss + model_params['lambda_mi_loss'] * mi_loss + prior_loss - model_params['cosine_loss'] * cosineLoss
            loss = loss_1 + loss_2 + model_params['similarity_reg'] * silimalityLoss + model_params[
                'lambda_mi_loss'] * mi_loss + prior_loss + model_params['reg_classifier'] * entropy - model_params[
                       'reg_adv'] * adv_entropy + classifier.L2Regularization(
                model_params['state_class_reg']) + Vsp.Regularization(model_params['v_reg']) - model_params[
                       'cosine_loss'] * cosineLoss

            loss.backward()
            optimizer.step()

            pearson_1 = pearson_r(y_pred_1.detach().flatten(), X_1.detach().flatten())
            r2_1 = r_square(y_pred_1.detach().flatten(), X_1.detach().flatten())
            mse_1 = torch.mean(torch.mean((y_pred_1.detach() - X_1.detach()) ** 2, dim=1))

            pearson_2 = pearson_r(y_pred_2.detach().flatten(), X_2.detach().flatten())
            r2_2 = r_square(y_pred_2.detach().flatten(), X_2.detach().flatten())
            mse_2 = torch.mean(torch.mean((y_pred_2.detach() - X_2.detach()) ** 2, dim=1))

            # iteration += iteration

        if model_params['schedule_step_adv'] is not None:
            scheduler_adv.step()
        if (e >= 0):
            scheduler.step()
            outString = 'Split {:.0f}: Epoch={:.0f}/{:.0f}'.format(i + 1, e + 1, NUM_EPOCHS)
            outString += ', r2_1={:.4f}'.format(r2_1.item())
            outString += ', pearson_1={:.4f}'.format(pearson_1.item())
            outString += ', MSE_1={:.4f}'.format(mse_1.item())
            outString += ', r2_2={:.4f}'.format(r2_2.item())
            outString += ', pearson_2={:.4f}'.format(pearson_2.item())
            outString += ', MSE_2={:.4f}'.format(mse_2.item())
            outString += ', MI Loss={:.4f}'.format(mi_loss.item())
            outString += ', Prior Loss={:.4f}'.format(prior_loss.item())
            outString += ', Entropy Loss={:.4f}'.format(entropy.item())
            outString += ', Adverse Entropy={:.4f}'.format(adv_entropy.item())
            outString += ', Cosine Loss={:.4f}'.format(cosineLoss.item())
            outString += ', loss={:.4f}'.format(loss.item())
            outString += ', F1 latent={:.4f}'.format(f1_latent)
            outString += ', F1 basal={:.4f}'.format(f1_basal)
            # if e % model_params["adversary_steps"] == 0 and e>0:
            outString += ', F1 basal trained={:.4f}'.format(f1_basal_trained)
            # else:
            #    outString += ', F1 basal trained= %s'%f1_basal_trained
        if (e == 0 or (e % 50 == 0 and e > 0)):
            print2log(outString)
    print2log(outString)
    decoder_1.eval()
    decoder_2.eval()
    # pre_decoder_1.eval()
    # pre_decoder_2.eval()
    encoder_1.eval()
    encoder_2.eval()
    prior_d.eval()
    local_d.eval()
    classifier.eval()
    adverse_classifier.eval()
    Vsp.eval()

    paired_val_inds = len(valInfo_paired)
    x_1 = torch.tensor(np.concatenate((cmap.loc[valInfo_paired['sig_id.x']].values,
                                       cmap.loc[valInfo_1.sig_id].values))).float().to(device)
    x_2 = torch.tensor(np.concatenate((cmap.loc[valInfo_paired['sig_id.y']].values,
                                       cmap.loc[valInfo_2.sig_id].values))).float().to(device)

    z_species_1 = torch.cat((torch.ones(x_1.shape[0], 1),
                             torch.zeros(x_1.shape[0], 1)), 1).to(device)
    z_species_2 = torch.cat((torch.zeros(x_2.shape[0], 1),
                             torch.ones(x_2.shape[0], 1)), 1).to(device)

    # z_latent_1 = encoder_1(x_1)
    # z_latent_2 = encoder_2(x_2)

    z_latent_base_1 = encoder_1(x_1)
    z_latent_base_2 = encoder_2(x_2)

    z_latent_1 = Vsp(z_latent_base_1, z_species_1)
    z_latent_2 = Vsp(z_latent_base_2, z_species_2)

    labels = classifier(torch.cat((z_latent_1, z_latent_2), 0))
    true_labels = torch.cat((torch.ones(z_latent_1.shape[0]).view(z_latent_1.shape[0], 1),
                             torch.zeros(z_latent_2.shape[0]).view(z_latent_2.shape[0], 1)), 0).long()
    _, predicted = torch.max(labels, 1)
    predicted = predicted.cpu().numpy()
    cf_matrix = confusion_matrix(true_labels.numpy(), predicted)
    tn, fp, fn, tp = cf_matrix.ravel()
    class_acc = (tp + tn) / predicted.size
    f1 = 2 * tp / (2 * tp + fp + fn)

    valF1.append(f1)
    valClassAcc.append(class_acc)

    print2log('Classification accuracy: %s' % class_acc)
    print2log('Classification F1 score: %s' % f1)

    xhat_1 = decoder_1(z_latent_1)
    xhat_2 = decoder_2(z_latent_2)
    # xhat_1 = pre_decoder_1(z_latent_1)
    # xhat_2 = pre_decoder_2(z_latent_2)

    r2_1 = r_square(xhat_1.detach().flatten(), x_1.detach().flatten())
    pearson_1 = pearson_r(xhat_1.detach().flatten(), x_1.detach().flatten())
    mse_1 = torch.mean(torch.mean((xhat_1 - x_1) ** 2, dim=1))
    r2_2 = r_square(xhat_2.detach().flatten(), x_2.detach().flatten())
    pearson_2 = pearson_r(xhat_2.detach().flatten(), x_2.detach().flatten())
    mse_2 = torch.mean(torch.mean((xhat_2 - x_2) ** 2, dim=1))
    rhos = []
    for jj in range(xhat_1.shape[0]):
        rho, p = spearmanr(x_1[jj, :].detach().cpu().numpy(), xhat_1[jj, :].detach().cpu().numpy())
        rhos.append(rho)
    valSpear_1.append(np.mean(rhos))
    acc = pseudoAccuracy(x_1.detach().cpu(), xhat_1.detach().cpu(), eps=1e-6)
    valAccuracy_1.append(np.mean(acc))
    rhos = []
    for jj in range(xhat_2.shape[0]):
        rho, p = spearmanr(x_2[jj, :].detach().cpu().numpy(), xhat_2[jj, :].detach().cpu().numpy())
        rhos.append(rho)
    valSpear_2.append(np.mean(rhos))
    acc = pseudoAccuracy(x_2.detach().cpu(), xhat_2.detach().cpu(), eps=1e-6)
    valAccuracy_2.append(np.mean(acc))

    valPear_1.append(pearson_1.item())
    valPear_2.append(pearson_2.item())
    # print('R^2 1: %s'%r2_1.item())
    print2log('Pearson correlation 1: %s' % pearson_1.item())
    print2log('Spearman correlation 1: %s' % valSpear_1[i])
    print2log('Pseudo-Accuracy 1: %s' % valAccuracy_1[i])
    # print('R^2 2: %s'%r2_2.item())
    print2log('Pearson correlation 2: %s' % pearson_2.item())
    print2log('Spearman correlation 2: %s' % valSpear_2[i])
    print2log('Pseudo-Accuracy 2: %s' % valAccuracy_2[i])

    # x_1_equivalent = torch.tensor(cmap_val.loc[mask.index[np.where(mask>0)[0]],:].values).float().to(device)
    # x_2_equivalent = torch.tensor(cmap_val.loc[mask.columns[np.where(mask>0)[1]],:].values).float().to(device)
    x_1_equivalent = x_1[0:paired_val_inds, :]
    x_2_equivalent = x_2[0:paired_val_inds, :]

    z_species_1_equivalent = z_species_1[0:paired_val_inds, :]
    z_species_2_equivalent = z_species_2[0:paired_val_inds, :]

    pearDirect = pearson_r(x_1_equivalent.detach().flatten(), x_2_equivalent.detach().flatten())
    rhos = []
    for jj in range(x_1_equivalent.shape[0]):
        rho, p = spearmanr(x_1_equivalent[jj, :].detach().cpu().numpy(), x_2_equivalent[jj, :].detach().cpu().numpy())
        rhos.append(rho)
    spearDirect = np.mean(rhos)
    accDirect_2 = np.mean(pseudoAccuracy(x_2_equivalent.detach().cpu(), x_1_equivalent.detach().cpu(), eps=1e-6))
    accDirect_1 = np.mean(pseudoAccuracy(x_1_equivalent.detach().cpu(), x_2_equivalent.detach().cpu(), eps=1e-6))

    z_latent_base_1_equivalent = encoder_1(x_1_equivalent)
    # z_latent_1_equivalent = encoder_1(x_1_equivalent)
    z_latent_1_equivalent = Vsp(z_latent_base_1_equivalent, 1. - z_species_1_equivalent)
    x_hat_2_equivalent = decoder_2(z_latent_1_equivalent).detach()
    # x_hat_2_equivalent = pre_decoder_2(z_latent_1_equivalent).detach()
    r2_2 = r_square(x_hat_2_equivalent.detach().flatten(), x_2_equivalent.detach().flatten())
    pearson_2 = pearson_r(x_hat_2_equivalent.detach().flatten(), x_2_equivalent.detach().flatten())
    mse_2 = torch.mean(torch.mean((x_hat_2_equivalent - x_2_equivalent) ** 2, dim=1))
    rhos = []
    for jj in range(x_hat_2_equivalent.shape[0]):
        rho, p = spearmanr(x_2_equivalent[jj, :].detach().cpu().numpy(),
                           x_hat_2_equivalent[jj, :].detach().cpu().numpy())
        rhos.append(rho)
    rho_2 = np.mean(rhos)
    acc_2 = np.mean(pseudoAccuracy(x_2_equivalent.detach().cpu(), x_hat_2_equivalent.detach().cpu(), eps=1e-6))
    print2log('Pearson of direct translation: %s' % pearDirect.item())
    print2log('Pearson correlation 1 to 2: %s' % pearson_2.item())
    print2log('Pseudo accuracy 1 to 2: %s' % acc_2)

    z_latent_base_2_equivalent = encoder_2(x_2_equivalent)
    # z_latent_2_equivalent = encoder_2(x_2_equivalent)
    z_latent_2_equivalent = Vsp(z_latent_base_2_equivalent, 1. - z_species_2_equivalent)
    x_hat_1_equivalent = decoder_1(z_latent_2_equivalent).detach()
    # x_hat_1_equivalent = pre_decoder_1(z_latent_2_equivalent).detach()
    r2_1 = r_square(x_hat_1_equivalent.detach().flatten(), x_1_equivalent.detach().flatten())
    pearson_1 = pearson_r(x_hat_1_equivalent.detach().flatten(), x_1_equivalent.detach().flatten())
    mse_1 = torch.mean(torch.mean((x_hat_1_equivalent - x_1_equivalent) ** 2, dim=1))
    rhos = []
    for jj in range(x_hat_1_equivalent.shape[0]):
        rho, p = spearmanr(x_1_equivalent[jj, :].detach().cpu().numpy(),
                           x_hat_1_equivalent[jj, :].detach().cpu().numpy())
        rhos.append(rho)
    rho_1 = np.mean(rhos)
    acc_1 = np.mean(pseudoAccuracy(x_1_equivalent.detach().cpu(), x_hat_1_equivalent.detach().cpu(), eps=1e-6))
    print2log('Pearson correlation 2 to 1: %s' % pearson_1.item())
    print2log('Pseudo accuracy 2 to 1: %s' % acc_1)

    valPear.append([pearson_2.item(), pearson_1.item()])
    valSpear.append([rho_2, rho_1])
    valAccuracy.append([acc_2, acc_1])

    valPearDirect.append(pearDirect.item())
    valSpearDirect.append(spearDirect)
    valAccDirect.append([accDirect_2, accDirect_1])

    torch.save(decoder_1, '../results/CPAmixedTransCompR_results/models/decoder_a375_%s.pt' % i)
    torch.save(decoder_2, '../results/CPAmixedTransCompR_results/models/decoder_ht29_%s.pt' % i)
    torch.save(prior_d, '../results/CPAmixedTransCompR_results/models/priorDiscr_%s.pt' % i)
    torch.save(local_d, '../results/CPAmixedTransCompR_results/models/localDiscr_%s.pt' % i)
    torch.save(encoder_1, '../results/CPAmixedTransCompR_results/models/encoder_a375_%s.pt' % i)
    torch.save(encoder_2, '../results/CPAmixedTransCompR_results/models/encoder_ht29_%s.pt' % i)
    torch.save(classifier, '../results/CPAmixedTransCompR_results/models/classifier_%s.pt' % i)
    torch.save(Vsp, '../results/CPAmixedTransCompR_results/models/Vspecies_%s.pt' % i)
    torch.save(adverse_classifier, '../results/CPAmixedTransCompR_results/models/classifier_adverse_%s.pt' % i)


valPear = np.array(valPear)
valPearDirect = np.array(valPearDirect)
valSpear = np.array(valSpear)
valAccuracy= np.array(valAccuracy)
valSpearDirect= np.array(valSpearDirect)
valAccDirect= np.array(valAccDirect)


# In[18]:


print2log(np.mean(valPear,axis=0))
print2log(np.mean(valPearDirect))



df_result = pd.DataFrame({'model_pearsonHT29':valPear[:,0],'model_pearsonA375':valPear[:,1],
                          'model_spearHT29':valSpear[:,0],'model_spearA375':valSpear[:,1],
                          'model_accHT29':valAccuracy[:,0],'model_accA375':valAccuracy[:,1],
                          'recon_pear_ht29':valPear_2 ,'recon_pear_a375':valPear_1,
                          'recon_spear_ht29':valSpear_2 ,'recon_spear_a375':valSpear_1,
                          'recon_acc_ht29':valAccuracy_2 ,'recon_acc_a375':valAccuracy_1,
                          'Direct_pearson':valPearDirect,'Direct_spearman':valSpearDirect,
                          'DirectAcc_ht29':valAccDirect[:,0],'DirectAcc_a375':valAccDirect[:,1]})

df_result.to_csv('../results/CPAmixedTransCompR_results/'+gene_space+'_10foldvalidation_CPAmixedTransCompR_500ep512bs_a375_ht29.csv')