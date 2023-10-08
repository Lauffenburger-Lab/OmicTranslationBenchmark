import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score,confusion_matrix
import torch
import torch.nn.functional as F
import math
from models import SimpleEncoder,Decoder,PriorDiscriminator,LocalDiscriminator,Classifier
from evaluationUtils import r_square,get_cindex,pearson_r,pseudoAccuracy
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
print2log = logger.info


# In[3]:


device = torch.device('cuda')


# In[4]:


# Create a train generators
def getSamples(N, batchSize):
    order = np.random.permutation(N)
    outList = []
    while len(order)>0:
        outList.append(order[0:batchSize])
        order = order[batchSize:]
    return outList


# # Load Data

# In[5]:


# Gex data 
cmap = pd.read_csv('cmap_all_genes_q1_tas03.csv',index_col=0)
#lands = pd.read_csv('cmap_landmarks_HT29_A375.csv',index_col=0)
#lands = lands.columns.values
#cmap = cmap.loc[:,lands]
gene_size = len(cmap.columns)
X = cmap.values


# # Train one trasnlation model

# In[6]:


# model_params = {'encoder_1_hiddens':[640,384],
#                 'encoder_2_hiddens':[640,384],
#                 'latent_dim': 292,
#                 'decoder_1_hiddens':[384,640],
#                 'decoder_2_hiddens':[384,640],
#                 'dropout_decoder':0.2,
#                 'dropout_encoder':0.1,
#                 'encoder_activation':torch.nn.ELU(),
#                 'decoder_activation':torch.nn.ELU(),
#                 'V_dropout':0.25,
#                 'state_class_hidden':[256,128,64],
#                 'state_class_drop_in':0.5,
#                 'state_class_drop':0.25,
#                 'no_states':2,
#                 'adv_class_hidden':[256,128,64],
#                 'adv_class_drop_in':0.3,
#                 'adv_class_drop':0.1,
#                 'no_adv_class':2,
#                 'encoding_lr':0.001,
#                 'adv_lr':0.001,
#                 'schedule_step_adv':200,
#                 'gamma_adv':0.5,
#                 'schedule_step_enc':200,
#                 'gamma_enc':0.8,
#                 'batch_size_1':178,
#                 'batch_size_2':154,
#                 'batch_size_paired':90,
#                 'epochs':1000,
#                 'prior_beta':1.0,
#                 'no_folds':10,
#                 'v_reg':1e-04,
#                 'state_class_reg':1e-02,
#                 'enc_l2_reg':0.01,
#                 'dec_l2_reg':0.01,
#                 'lambda_mi_loss':100,
#                 'effsize_reg': 100,
#                 'cosine_loss': 10,
#                 'adv_penalnty':100,
#                 'reg_adv':1000,
#                 'reg_classifier': 1000,
#                 'similarity_reg' : 10,
#                 'adversary_steps':4,
#                 'autoencoder_wd': 0.,
#                 'adversary_wd': 0.}
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


# In[7]:


latent_dimensions = [512,256,128,64,32,16,8]


# In[8]:


class_criterion = torch.nn.CrossEntropyLoss()
bs_1 = model_params['batch_size_1']
bs_2 = model_params['batch_size_2']
bs_paired = model_params['batch_size_paired']
NUM_EPOCHS=model_params['epochs']
num_genes = cmap.shape[1]


# In[9]:


df_result_all = pd.DataFrame({})
for latent_dim in latent_dimensions:
    print2log('Start model with latent dimension = %s'%latent_dim)
    latent_dim = int(latent_dim)
    Path('LatentDimAnalysis/'+str(latent_dim)+'/models').mkdir(parents=True, exist_ok=True)
    valPear = []
    valPear_1 = []
    valPear_2 = []
    valF1 = []
    valClassAcc =[]
    for i in range(model_params['no_folds']):
        trainInfo_paired = pd.read_csv('10fold_validation_spit/train_paired_pc3_ha1e_%s.csv'%i,index_col=0)
        trainInfo_1 = pd.read_csv('10fold_validation_spit/train_pc3_%s.csv'%i,index_col=0)
        trainInfo_2 = pd.read_csv('10fold_validation_spit/train_ha1e_%s.csv'%i,index_col=0)
        valInfo_paired = pd.read_csv('10fold_validation_spit/val_paired_pc3_ha1e_%s.csv'%i,index_col=0)
        valInfo_1 = pd.read_csv('10fold_validation_spit/val_pc3_%s.csv'%i,index_col=0)
        valInfo_2 = pd.read_csv('10fold_validation_spit/val_ha1e_%s.csv'%i,index_col=0)
        
        N_paired = len(trainInfo_paired)
        N_1 = len(trainInfo_1)
        N_2 = len(trainInfo_2)
        N = N_1
        if N_2>N:
            N=N_2
        # Network
        decoder_1 = Decoder(latent_dim,model_params['decoder_1_hiddens'],num_genes,
                                dropRate=model_params['dropout_decoder'], 
                                activation=model_params['decoder_activation']).to(device)
        decoder_2 = Decoder(latent_dim,model_params['decoder_2_hiddens'],num_genes,
                                dropRate=model_params['dropout_decoder'], 
                                activation=model_params['decoder_activation']).to(device)
        encoder_1 = SimpleEncoder(num_genes,model_params['encoder_1_hiddens'],latent_dim,
                                      dropRate=model_params['dropout_encoder'], 
                                      activation=model_params['encoder_activation']).to(device)
        encoder_2 = SimpleEncoder(num_genes,model_params['encoder_2_hiddens'],latent_dim,
                                          dropRate=model_params['dropout_encoder'], 
                                          activation=model_params['encoder_activation']).to(device)
        classifier = Classifier(in_channel=latent_dim,
                            hidden_layers=model_params['state_class_hidden'],
                            num_classes=model_params['no_states'],
                            drop_in=model_params['state_class_drop_in'],
                            drop=model_params['state_class_drop']).to(device)
        prior_d = PriorDiscriminator(latent_dim).to(device)
        local_d = LocalDiscriminator(latent_dim,latent_dim).to(device)

        allParams = list(decoder_1.parameters()) + list(encoder_1.parameters())
        allParams = allParams + list(decoder_2.parameters()) + list(encoder_2.parameters())
        allParams = allParams  + list(local_d.parameters())
        allParams = allParams + list(prior_d.parameters())
        optimizer = torch.optim.Adam(allParams, lr=model_params['encoding_lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                        step_size=model_params['schedule_step_enc'],
                                                        gamma=model_params['gamma_enc'])
        
        trainLoss = []
        trainLossSTD = []
        for e in range(NUM_EPOCHS):
            trainloader_1 = getSamples(N_1, bs_1)
            len_1 = len(trainloader_1)
            trainloader_2 = getSamples(N_2, bs_2)
            len_2 = len(trainloader_2)
            trainloader_paired = getSamples(N_paired, bs_paired)
            len_paired = len(trainloader_paired)

            lens = [len_1,len_2,len_paired]
            maxLen = np.max(lens)

            if maxLen>lens[0]:
                trainloader_suppl = getSamples(N_1, bs_1)
                for jj in range(maxLen-lens[0]):
                    trainloader_1.insert(jj,trainloader_suppl[jj])

            if maxLen>lens[1]:
                trainloader_suppl = getSamples(N_2, bs_2)
                for jj in range(maxLen-lens[1]):
                    trainloader_2.insert(jj,trainloader_suppl[jj])

            if maxLen>lens[2]:
                trainloader_suppl = getSamples(N_paired, bs_paired)
                for jj in range(maxLen-lens[2]):
                    trainloader_paired.insert(jj,trainloader_suppl[jj])
            encoder_1.train()
            decoder_1.train()
            encoder_2.train()
            decoder_2.train()
            prior_d.train()
            local_d.train()
            classifier.train()

            trainLoss_ALL = []
            
            for j in range(maxLen):
                dataIndex_1 = trainloader_1[j]
                dataIndex_2 = trainloader_2[j]
                dataIndex_paired = trainloader_paired[j]

                df_pairs = trainInfo_paired.iloc[dataIndex_paired,:]
                df_1 = trainInfo_1.iloc[dataIndex_1,:]
                df_2 = trainInfo_2.iloc[dataIndex_2,:]
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
                conditions = conditions.reshape(size,1)
                conditions = conditions == conditions.transpose()
                conditions = conditions*1
                mask = torch.tensor(conditions).to(device).detach()
                pos_mask = mask
                neg_mask = 1 - mask
                log_2 = math.log(2.)
                optimizer.zero_grad()

                z_1 = encoder_1(X_1)
                z_2 = encoder_2(X_2)

                z_un = local_d(torch.cat((z_1, z_2), 0))
                res_un = torch.matmul(z_un, z_un.t())

                y_pred_1 = decoder_1(z_1)
                fitLoss_1 = torch.mean(torch.sum((y_pred_1 - X_1)**2,dim=1))
                L2Loss_1 = decoder_1.L2Regularization(0.01) + encoder_1.L2Regularization(0.01)
                loss_1 = fitLoss_1 + L2Loss_1

                y_pred_2 = decoder_2(z_2)
                fitLoss_2 = torch.mean(torch.sum((y_pred_2 - X_2)**2,dim=1))
                L2Loss_2 = decoder_2.L2Regularization(0.01) + encoder_2.L2Regularization(0.01)
                loss_2 = fitLoss_2 + L2Loss_2

                silimalityLoss = torch.mean(torch.sum((z_1[0:paired_inds,:] - z_2[0:paired_inds,:])**2,dim=-1))

                p_samples = res_un * pos_mask.float()
                q_samples = res_un * neg_mask.float()

                Ep = log_2 - F.softplus(- p_samples)
                Eq = F.softplus(-q_samples) + q_samples - log_2

                Ep = (Ep * pos_mask.float()).sum() / pos_mask.float().sum()
                Eq = (Eq * neg_mask.float()).sum() / neg_mask.float().sum()
                mi_loss = Eq - Ep

                prior = torch.rand_like(torch.cat((z_1, z_2), 0))

                term_a = torch.log(prior_d(prior)).mean()
                term_b = torch.log(1.0 - prior_d(torch.cat((z_1, z_2), 0))).mean()
                prior_loss = -(term_a + term_b) * model_params['prior_beta']

                # Classification loss
                labels = classifier(torch.cat((z_1, z_2), 0))
                true_labels = torch.cat((torch.ones(z_1.shape[0]),
                                         torch.zeros(z_2.shape[0])),0).long().to(device)
                entropy = class_criterion(labels,true_labels)
                _, predicted = torch.max(labels, 1)
                predicted = predicted.cpu().numpy()
                cf_matrix = confusion_matrix(true_labels.cpu().numpy(),predicted)
                tn, fp, fn, tp = cf_matrix.ravel()
                class_acc = (tp+tn)/predicted.size
                f1 = 2*tp/(2*tp+fp+fn)

                loss = loss_1 + loss_2 + mi_loss + prior_loss + silimalityLoss + 100*entropy +classifier.L2Regularization(1e-2)
                loss.backward()
                optimizer.step()

                pearson_1 = pearson_r(y_pred_1.detach().flatten(), X_1.detach().flatten())
                r2_1 = r_square(y_pred_1.detach().flatten(), X_1.detach().flatten())
                mse_1 = torch.mean(torch.mean((y_pred_1.detach() - X_1.detach())**2,dim=1))

                pearson_2 = pearson_r(y_pred_2.detach().flatten(), X_2.detach().flatten())
                r2_2 = r_square(y_pred_2.detach().flatten(), X_2.detach().flatten())
                mse_2 = torch.mean(torch.mean((y_pred_2.detach() - X_2.detach())**2,dim=1))


            scheduler.step()
            outString = 'Split {:.0f}: Epoch={:.0f}/{:.0f}'.format(i+1,e+1,NUM_EPOCHS)
            outString += ', r2_1={:.4f}'.format(r2_1.item())
            outString += ', pearson_1={:.4f}'.format(pearson_1.item())
            outString += ', MSE_1={:.4f}'.format(mse_1.item())
            outString += ', r2_2={:.4f}'.format(r2_2.item())
            outString += ', pearson_2={:.4f}'.format(pearson_2.item())
            outString += ', MSE_2={:.4f}'.format(mse_2.item())
            outString += ', MI Loss={:.4f}'.format(mi_loss.item())
            outString += ', Prior Loss={:.4f}'.format(prior_loss.item())
            outString += ', Entropy Loss={:.4f}'.format(entropy.item())
            outString += ', F1={:.4f}'.format(f1)
            outString += ', Accuracy={:.4f}'.format(class_acc)
            outString += ', loss={:.4f}'.format(loss.item())
            if (e%250==0):
                print2log(outString)
        print2log(outString)
        #trainLoss.append(splitLoss)
        decoder_1.eval()
        decoder_2.eval()
        encoder_1.eval()
        encoder_2.eval()
        prior_d.eval()
        local_d.eval()
        classifier.eval()
        #model.eval()
        #master_encoder.eval()

        paired_val_inds = len(valInfo_paired)
        x_1 = torch.tensor(np.concatenate((cmap.loc[valInfo_paired['sig_id.x']].values,
                                              cmap.loc[valInfo_1.sig_id].values))).float().to(device)
        x_2 = torch.tensor(np.concatenate((cmap.loc[valInfo_paired['sig_id.y']].values,
                                              cmap.loc[valInfo_2.sig_id].values))).float().to(device)

        z_latent_1 = encoder_1(x_1)
        z_latent_2 = encoder_2(x_2)

        labels = classifier(torch.cat((z_latent_1, z_latent_2), 0))
        true_labels = torch.cat((torch.ones(z_latent_1.shape[0]).view(z_latent_1.shape[0],1),
                                 torch.zeros(z_latent_2.shape[0]).view(z_latent_2.shape[0],1)),0).long()
        _, predicted = torch.max(labels, 1)
        predicted = predicted.cpu().numpy()
        cf_matrix = confusion_matrix(true_labels.numpy(),predicted)
        tn, fp, fn, tp = cf_matrix.ravel()
        class_acc = (tp+tn)/predicted.size
        f1 = 2*tp/(2*tp+fp+fn)

        valF1.append(f1)
        valClassAcc.append(class_acc)

        print2log('Classification accuracy: %s'%class_acc)
        print2log('Classification F1 score: %s'%f1)

        xhat_1 = decoder_1(z_latent_1)
        xhat_2 = decoder_2(z_latent_2)


        pearson_1 = pearson_r(xhat_1.detach().flatten(), x_1.detach().flatten())
        pearson_2 = pearson_r(xhat_2.detach().flatten(), x_2.detach().flatten())
        valPear_1.append(pearson_1.item())
        valPear_2.append(pearson_2.item())
        print2log('Pearson correlation 1: %s'%pearson_1.item())
        print2log('Pearson correlation 2: %s'%pearson_2.item())

        x_1_equivalent = x_1[0:paired_val_inds,:]
        x_2_equivalent = x_2[0:paired_val_inds,:]

        z_latent_1_equivalent  = encoder_1(x_1_equivalent)
        x_hat_2_equivalent = decoder_2(z_latent_1_equivalent).detach()
        print2log('Pearson correlation 1 to 2: %s'%pearson_2.item())
        
        z_latent_2_equivalent  = encoder_2(x_2_equivalent)
        x_hat_1_equivalent = decoder_1(z_latent_2_equivalent).detach()
        pearson_1 = pearson_r(x_hat_1_equivalent.detach().flatten(), x_1_equivalent.detach().flatten())
        print2log('Pearson correlation 2 to 1: %s'%pearson_1.item())
        valPear.append([pearson_2.item(),pearson_1.item()])
            
        torch.save(decoder_1.state_dict(),'LatentDimAnalysis/'+str(latent_dim)+'/models/decoder_1_%s.pth'%i)
        torch.save(decoder_2.state_dict(),'LatentDimAnalysis/'+str(latent_dim)+'/models/decoder_2_%s.pth'%i)
        torch.save(prior_d.state_dict(),'LatentDimAnalysis/'+str(latent_dim)+'/models/priorDiscr_%s.pth'%i)
        torch.save(local_d.state_dict(),'LatentDimAnalysis/'+str(latent_dim)+'/models/localDiscr_%s.pth'%i)
        torch.save(encoder_1.state_dict(),'LatentDimAnalysis/'+str(latent_dim)+'/models/encoder_1_%s.pth'%i)
        torch.save(encoder_2.state_dict(),'LatentDimAnalysis/'+str(latent_dim)+'/models/encoder_2_%s.pth'%i)
        torch.save(classifier.state_dict(),'LatentDimAnalysis/'+str(latent_dim)+'/models/classifier_%s.pth'%i)
    
    valPear = np.array(valPear)
    df_result = pd.DataFrame({'model_pearson1to2':valPear[:,0],'model_pearson2to1':valPear[:,1],
                              'recon_pear_2':valPear_2 ,'recon_pear_1':valPear_1,
                              'ClassF1':valF1,'ClassAcc':valClassAcc})
    df_result['latent_dim'] = latent_dim
    df_result_all = df_result_all.append(df_result)
    df_result_all.to_csv('LatentDimAnalysis/validation_results.csv')
    print2log('Finished model with latent dimension = %s'%latent_dim)


# In[ ]:




