from __future__ import absolute_import, division
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from trainingUtils import MultipleOptimizer, MultipleScheduler, compute_kernel, compute_mmd
from models import Decoder,Encoder, SimpleEncoder,LocalDiscriminator,PriorDiscriminator,Classifier,SpeciesCovariate
import math
import numpy as np
import pandas as pd
import sys
import random
import os
from sklearn.metrics import silhouette_score,confusion_matrix
from scipy.stats import spearmanr
from evaluationUtils import r_square,get_cindex,pearson_r,pseudoAccuracy


class GlobalAEFramework(torch.nn.Module):
    def __init__(self,num_of_modalities,paired_modalities,
                 in_channels, enc_hidden_layers, latent_dim, dec_hidden_layers ,
                 dropRateEnc=0.1, activationEnc=None, biasEnc=True,normalizeLatent=False,
                 dropInDec=0,dropRateDec=0.1, activationDec=None, biasDec=True,
                 covariateDrop = 0.1,
                 variational=False):

        super(GlobalAEFramework, self).__init__()

        self.num_of_modalities = num_of_modalities
        self.paired_modalities = paired_modalities
        self.encoders = torch.nn.ModuleList()
        self.decoders = torch.nn.ModuleList()
        self.Vmodalities = SpeciesCovariate(self.num_of_modalities,latent_dim,covariateDrop)

        for i in range(self.num_of_modalities):
            self.decoders.append(Decoder(latent_dim, dec_hidden_layers[i], in_channels[i], dropRateDec, dropInDec, activationDec, biasDec))
            if variational == True:
                self.encoders.append(Encoder(in_channels[i], enc_hidden_layers[i], latent_dim,dropRateEnc, activationEnc, biasEnc))
            else:
                self.encoders.append(SimpleEncoder(in_channels[i], enc_hidden_layers[i], latent_dim, dropRateEnc, activationEnc,normalizeLatent, biasEnc))

    def forward(self, x , x_modality):

        outs = []
        z_latents_base = []
        z_latents = []
        for i in range(self.num_of_modalities):
            zbase = self.encoders[i](x[i])
            z = self.Vmodalities(zbase,x_modality[i])
            xhat = self.decoders[i](z)
            outs.append(xhat)
            z_latents_base.append(zbase)
            z_latents.append(z)
        return torch.cat(z_latents_base,0),torch.cat(z_latents,0),torch.cat(outs,0)

    def translate(self,modality1,modality2,x1,x_modality2):
        # Translate from 1 ---> 2. modality1 and 2 are the indexes
        zbase1 = self.encoders[modality1](x1)
        z = self.Vmodalities(zbase1,x_modality2)
        xtranslated = self.decoders[modality2](z)
        return xtranslated

    def single_encode(self,modality,x,x_modality):
        zbase = self.encoders[modality](x)
        z = self.Vmodalities(zbase, x_modality)
        return zbase,z

    def single_decode(self,modality,z):
        xhat = self.decoders[modality](z)
        return xhat

class GlobalTrainFramework():

    def __init__(self,Framework,adverseClassifier,classifier, localDiscriminator,priorDiscriminator,
                 model_params,samplesModalities):

        super(GlobalTrainFramework, self).__init__()

        self.num_of_modalities = Framework.num_of_modalities
        self.paired_modalities = Framework.paired_modalities
        self.globalFrame = Framework
        self.adverseClassifier = adverseClassifier
        self.classifier = classifier
        self.localDiscriminator = localDiscriminator
        self.priorDiscriminator = priorDiscriminator
        self.samplesModalities = samplesModalities

        self.model_params = model_params
        allParams = list(self.globalFrame.parameters())
        allParams = allParams + list(self.priorDiscriminator.parameters()) + list(self.localDiscriminator.parameters())
        allParams = allParams + list(self.classifier.parameters())
        self.optimizer = torch.optim.Adam(allParams, lr=model_params['encoding_lr'], weight_decay=0)
        self.optimizer_adv = torch.optim.Adam(self.adverseClassifier.parameters(), lr=model_params['adv_lr'], weight_decay=0)
        if model_params['schedule_step_adv'] is not None:
            self.scheduler_adv = torch.optim.lr_scheduler.StepLR(self.optimizer_adv,
                                                            step_size=model_params['schedule_step_adv'],
                                                            gamma=model_params['gamma_adv'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                    step_size=model_params['schedule_step_enc'],
                                                    gamma=model_params['gamma_enc'])

        #Create paired data
        self.trainInfo_paired_list = []
        for i in range(len(self.paired_modalities)):
            sampleInfo1 = self.samplesModalities[self.paired_modalities[i][0]]
            sampleInfo2 = self.samplesModalities[self.paired_modalities[i][1]]
            # CODE TO CREATE PAIRED CONDITIONS. BASICALLY USE SOMETHING LIKE R
            self.trainInfo_paired_list.append()

    def train_framework():
        N_paired = len(self.trainInfo_paired)
        sample_sizes = [len(data) for data in self.samplesModalities]



        


