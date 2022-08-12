from __future__ import absolute_import, division
from tqdm.auto import tqdm
import torch
from torch.autograd import Variable, grad
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np
import pandas as pd
import sys
import random
import os
import math

def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
             _reset(nn)

class Encoder(torch.nn.Module):
    def __init__(self, in_channel, hidden_layers, latent_dim,dropRate=0.1, activation=None, bias=True):

        super(Encoder, self).__init__()

        self.bias = bias
        self.num_hidden_layers = len(hidden_layers)
        self.bn = torch.nn.ModuleList()
        self.linear_layers = torch.nn.ModuleList()
        self.linear_layers.append(torch.nn.Linear(in_channel, hidden_layers[0], bias=bias))
        self.bn.append(torch.nn.BatchNorm1d(num_features=hidden_layers[0], momentum=0.6))
        for i in range(1, len(hidden_layers)):
            self.linear_layers.append(torch.nn.Linear(hidden_layers[i - 1], hidden_layers[i], bias=bias))
            self.bn.append(torch.nn.BatchNorm1d(num_features=hidden_layers[i], momentum=0.6))
        # for i in range(1, num_hidden_layers + 1):
        #     self.linear_layers.append(torch.nn.Linear(in_channel // (2 ** (i - 1)), in_channel // (2 ** i),
        #                                                           bias=bias))
        #     self.bn.append(torch.nn.BatchNorm1d(num_features=in_channel // (2 ** i), momentum=0.6))

        self.linear_latent_mu = torch.nn.Linear(hidden_layers[-1],
                                                latent_dim,
                                                bias=False)
        # self.linear_latent_var = torch.nn.Linear(hidden_layers[-1],
        #                                        latent_dim,
        #                                         bias=False)
        if activation is not None:
            self.activation = activation
        # self.bn = nn.BatchNorm1d(num_features=latent_dim, momentum=0.6, dtype=torch.double)
        self.dropout = torch.nn.Dropout(dropRate)
        self.drop_in = torch.nn.Dropout(0.5)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0.

        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)


    # def reparameterize(self, mu, log_var):
    #
    #     if self.training:
    #         std = torch.exp(0.5 * log_var)
    #         eps = self.N.sample(std.shape)
    #         # eps = Variable(std.normal_())
    #         # z = eps.mul(std).add_(mu)
    #         z = self.N.sample(mu.shape).mul(std).add_(mu)
    #         return z
    #     else:
    #         return mu

    def forward(self, x):
        x = self.drop_in(x)
        for i in range(self.num_hidden_layers):
            x = self.linear_layers[i](x)
            x = self.bn[i](x)
            x = self.activation(x)
            x = self.dropout(x)
        z_latent = self.linear_latent_mu(x)
        # z_mu = self.linear_latent_mu(x)
        # z_log_var = self.linear_latent_var(x)
        # z_latent = self.reparameterize(z_mu, z_log_var)

        #self.kl = -0.5 * (1 - torch.exp(z_log_var) - z_mu ** 2 + z_log_var).sum()

        return z_latent

    def L2Regularization(self, L2):

        weightLoss = 0.
        biasLoss = 0.
        for i in range(self.num_hidden_layers):
            weightLoss = weightLoss + L2 * torch.sum((self.linear_layers[i].weight)**2)
            if self.bias==True:
                biasLoss = biasLoss + L2 * torch.sum((self.linear_layers[i].bias)**2)
        L2Loss = biasLoss + weightLoss
        return(L2Loss)



class Decoder(torch.nn.Module):
    def __init__(self, latent_dim, hidden_layers, out_dim,dropRate=0.1,dropIn=0, activation=None, bias=True):

        super(Decoder, self).__init__()

        self.bias = bias
        self.num_hidden_layers = len(hidden_layers)
        self.bn = torch.nn.ModuleList()
        self.linear_layers = torch.nn.ModuleList()
        self.linear_layers.append(torch.nn.Linear(latent_dim, hidden_layers[0], bias=bias))
        self.bn.append(torch.nn.BatchNorm1d(num_features=hidden_layers[0], momentum=0.6))
        for i in range(1, len(hidden_layers)):
            self.linear_layers.append(torch.nn.Linear(hidden_layers[i - 1], hidden_layers[i], bias=bias))
            self.bn.append(torch.nn.BatchNorm1d(num_features=hidden_layers[i], momentum=0.6))
        # for i in range(1,num_hidden_layers + 1):
        #     self.linear_layers.append(torch.nn.Linear(latent_dim * (2 ** (i - 1)), latent_dim * (2 ** i),
        #                                                           bias=bias))
        #     self.bn.append(torch.nn.BatchNorm1d(num_features=latent_dim * (2 ** i), momentum=0.6))

        self.output_linear = torch.nn.Linear(hidden_layers[-1],
                                             out_dim,
                                             bias=False)

        if activation is not None:
            self.activation = activation
        # self.bn = nn.BatchNorm1d(num_features=latent_dim, momentum=0.6, dtype=torch.double)
        self.dropout = torch.nn.Dropout(dropRate)
        self.dropIn = dropIn
        if dropIn>0:
            self.drop_input = torch.nn.Dropout(dropIn)
        #self.activation_output = torch.nn.Sigmoid()

        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x):
        if self.dropIn>0:
            x = self.drop_input(x)
        for i in range(self.num_hidden_layers):
            x = self.linear_layers[i](x)
            x = self.bn[i](x)
            x = self.activation(x)
            x = self.dropout(x)

        output = self.output_linear(x)
        #output = self.activation_output(output)

        return output

    def L2Regularization(self, L2):

        weightLoss = 0.
        biasLoss = 0.
        for i in range(self.num_hidden_layers):
            weightLoss = weightLoss + L2 * torch.sum((self.linear_layers[i].weight)**2)
            if self.bias==True:
                biasLoss = biasLoss + L2 * torch.sum((self.linear_layers[i].bias)**2)
        L2Loss = biasLoss + weightLoss
        return(L2Loss)

class VAE(torch.nn.Module):

    def __init__(self, enc, dec, device):

        super(VAE, self).__init__()

        self.encoder = enc
        self.decoder = dec
        self.device = device
        # self.log_scale = torch.nn.Parameter(torch.Tensor([0.0]))

    def forward(self, x):
        z_latent = self.encoder(x)
        predicted = self.decoder(z_latent)

        return z_latent, predicted

    # def kl_divergence(self, z, mu, std):
    #     # Initialize the distributions
    #     p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
    #     q = torch.distributions.Normal(mu, std)
    #
    #     # get the probabilities
    #     log_qzx = q.log_prob(z)
    #     log_pz = p.log_prob(z)
    #
    #     # kl
    #     kl = (log_qzx - log_pz)
    #
    #     # sum over last dim to go from single dim distribution to multi-dim
    #     kl = kl.sum(-1)
    #     return kl
    #
    # def reconstruction_likelihood_loss(self, x_hat, logscale, x):
    #
    #     scale = torch.exp(logscale)
    #     mean = x_hat
    #     dist = torch.distributions.Normal(mean, scale)
    #
    #     # measure prob of seeing image under p(x|z)
    #     log_pxz = dist.log_prob(x)
    #     return log_pxz.sum()

    def encode(self, x):
        z_latent = self.encoder(x)
        return z_latent

    def decode(self, x):
        decoded_output = self.decoder(x)
        return decoded_output

    def L2Regularization(self, L2):

        encoderL2 = self.encoder.L2Regularization(L2)
        decoderL2 = self.decoder.L2Regularization(L2)

        L2Loss = encoderL2 + decoderL2
        return(L2Loss)

class SimpleEncoder(torch.nn.Module):
    def __init__(self, in_channel, hidden_layers, latent_dim,dropRate=0.1, activation=None,normalizeOutput=False, bias=True):

        super(SimpleEncoder, self).__init__()

        self.bias = bias
        self.normalizeOutput = normalizeOutput
        self.num_hidden_layers = len(hidden_layers)
        self.bn = torch.nn.ModuleList()
        self.linear_layers = torch.nn.ModuleList()
        self.linear_layers.append(torch.nn.Linear(in_channel, hidden_layers[0], bias=bias))
        self.bn.append(torch.nn.BatchNorm1d(num_features=hidden_layers[0], momentum=0.6))
        for i in range(1, len(hidden_layers)):
            self.linear_layers.append(torch.nn.Linear(hidden_layers[i - 1], hidden_layers[i], bias=bias))
            self.bn.append(torch.nn.BatchNorm1d(num_features=hidden_layers[i], momentum=0.6))

        self.linear_latent = torch.nn.Linear(hidden_layers[-1],
                                             latent_dim,
                                             bias=False)
        if activation is not None:
            self.activation = activation
        # self.bn = nn.BatchNorm1d(num_features=latent_dim, momentum=0.6, dtype=torch.double)
        self.dropout = torch.nn.Dropout(dropRate)
        self.drop_in = torch.nn.Dropout(0.5)

        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x):
        x = self.drop_in(x)
        for i in range(self.num_hidden_layers):
            x = self.linear_layers[i](x)
            x = self.bn[i](x)
            x = self.activation(x)
            x = self.dropout(x)
        z_latent = self.linear_latent(x)
        if self.normalizeOutput == True:
            z_latent = torch.nn.functional.normalize(z_latent)
        return z_latent

    def L2Regularization(self, L2):

        weightLoss = 0.
        biasLoss = 0.
        for i in range(self.num_hidden_layers):
            weightLoss = weightLoss + L2 * torch.sum((self.linear_layers[i].weight)**2)
            if self.bias==True:
                biasLoss = biasLoss + L2 * torch.sum((self.linear_layers[i].bias)**2)
        weightLoss = weightLoss + L2 * torch.sum((self.linear_latent.weight)**2)
        L2Loss = biasLoss + weightLoss
        return(L2Loss)

class LocalDiscriminator(torch.nn.Module):
    r"""Implemented from https://github.com/BioSysLab/deepSNEM"""
    def __init__(self, input_dim, dim):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Linear(input_dim, input_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(input_dim, input_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(input_dim, dim),
            torch.nn.ReLU()
        )
        self.linear_shortcut = torch.nn.Linear(input_dim, dim)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)


class PriorDiscriminator(torch.nn.Module):
    r"""Implemented from https://github.com/BioSysLab/deepSNEM"""
    def __init__(self, input_dim):
        super().__init__()
        self.l0 = torch.nn.Linear(input_dim, input_dim)
        self.l1 = torch.nn.Linear(input_dim, input_dim)
        self.l2 = torch.nn.Linear(input_dim, 1)

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))

class EmbInfomax(torch.nn.Module):
    r"""Implemented from https://github.com/BioSysLab/deepSNEM"""
    def __init__(self, latent_dim, encoder, beta=1.0):
        super(EmbInfomax, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = encoder

        self.local_d = LocalDiscriminator(latent_dim, latent_dim)

        self.prior_d = PriorDiscriminator(latent_dim)
        self.beta = beta

        self.reset_parameters()
        self.init_emb()

    def reset_parameters(self):
        reset(self.encoder)

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x):
        #neg_z = self.encoder.corrupt_forward(data)
        z = self.encoder(x)
        #self.pos_mask = mask
        #self.neg_mask = 1 - mask

        z_un = self.local_d(z)
        res_un = torch.matmul(z_un, z_un.t())

        return res_un, z

    def loss(self, res_un,z,mask):
        r"""Computes the mutual information maximization objective.
        Implemented from https://github.com/BioSysLab/deepSNEM"""
        pos_mask = mask
        neg_mask = 1 - mask
        log_2 = math.log(2.)
        p_samples = res_un * pos_mask
        q_samples = res_un * neg_mask

        Ep = log_2 - F.softplus(- p_samples)
        Eq = F.softplus(-q_samples) + q_samples - log_2

        Ep = (Ep * pos_mask).sum() / pos_mask.sum()
        Eq = (Eq * neg_mask).sum() / neg_mask.sum()
        LOCAL = Eq - Ep

        prior = torch.rand_like(z)

        term_a = torch.log(self.prior_d(prior)).mean()
        term_b = torch.log(1.0 - self.prior_d(z)).mean()
        PRIOR = -(term_a + term_b) * self.beta

        return LOCAL, PRIOR

    def encode(self, x):
        z_latent = self.encoder(x)
        return z_latent

    def L2Regularization(self, L2):

        L2Loss = self.encoder.L2Regularization(L2)

        #L2Loss = encoderL2
        return(L2Loss)
    
    
class CellStateEncoder(torch.nn.Module):
    def __init__(self, latent_dim , in_dim,hidden_layers = [512,256], dropRate=0.5,activation=torch.nn.LeakyReLU(0.2), bias=True):

        super(CellStateEncoder, self).__init__()

        self.bias = bias
        self.linear_layers = torch.nn.ModuleList()
        self.linear_layers.append(torch.nn.Linear(in_dim, hidden_layers[0], bias=bias))
        for i in range(1, len(hidden_layers)):
            self.linear_layers.append(torch.nn.Linear(hidden_layers[i - 1], hidden_layers[i], bias=bias))
        self.linear_latent = torch.nn.Linear(hidden_layers[- 1], latent_dim, bias=bias)
        # self.linear_latent_mu = torch.nn.Linear(hidden_layers[-1],
        #                                         latent_dim,
        #                                         bias=False)
        # self.linear_latent_var = torch.nn.Linear(hidden_layers[-1],
        #                                          latent_dim,
        #                                          bias=False)
        #
        # self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.activation = torch.nn.LeakyReLU(0.2)
        self.dropout = torch.nn.Dropout(dropRate)

        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    # def reparameterize(self, mu, log_var):
    #
    #     if self.training:
    #         std = torch.exp(0.5 * log_var)
    #         eps = self.N.sample(std.shape)
    #         # eps = Variable(std.normal_())
    #         # z = eps.mul(std).add_(mu)
    #         z = self.N.sample(mu.shape).mul(std).add_(mu)
    #         return z
    #     else:
    #         return mu

    def forward(self, x):
        xd = self.dropout(x)
        x2 = self.linear_layers[0](xd)
        x2 = self.activation(x2)
        for i in range(1,len(self.linear_layers)):
            x2 = self.linear_layers[i](x2)
            x2 = self.activation(x2)
        #z_mu = self.linear_latent_mu(x2)
        #z_log_var = self.linear_latent_var(x2)
        #z_latent = self.reparameterize(z_mu, z_log_var)
        z_latent = self.linear_latent(x2)
        return z_latent



class CellStateDecoder(torch.nn.Module):
    def __init__(self, latent_dim , out_dim, hidden_layers = [256,512], dropRate=0.8,activation=torch.nn.LeakyReLU(0.2), bias=True):

        super(CellStateDecoder, self).__init__()

        self.bias = bias
        self.linear_layers = torch.nn.ModuleList()
        self.linear_layers.append(torch.nn.Linear(latent_dim, hidden_layers[0],bias=bias))
        for i in range(1,len(hidden_layers)):
            self.linear_layers.append(torch.nn.Linear(hidden_layers[i-1], hidden_layers[i], bias=bias))

        self.output_linear = torch.nn.Linear(hidden_layers[-1], out_dim,bias=bias)
        # self.final_dense = torch.nn.Linear(2, 1,bias=bias)
        self.activation = activation
        self.dropout = torch.nn.Dropout(dropRate)
        #self.activation_output = torch.nn.Sigmoid()

        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, z):
        for i in range(len(self.linear_layers)):
            z = self.linear_layers[i](z)
            z = self.activation(z)
        z = self.dropout(self.output_linear(z))
        # z2 = torch.cat([z.unsqueeze(-1),x_noisy.unsqueeze(-1)],axis=-1)
        # output = self.final_dense(z2).squeeze()
        return z.squeeze()

class CellStateVAE(torch.nn.Module):
    def __init__(self, encoder, decoder):

        super(CellStateVAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z_latent = self.encoder(x)
        predicted = self.decoder(z_latent)

        return z_latent, predicted

class MultiEncInfomax(torch.nn.Module):
    r"""Implemented from https://github.com/BioSysLab/deepSNEM"""
    def __init__(self, latent_dim, encoders, beta=1.0):
        super(MultiEncInfomax, self).__init__()
        self.latent_dim = latent_dim
        self.encoders = torch.nn.ModuleList(encoders)

        self.local_d = LocalDiscriminator(latent_dim, latent_dim)

        self.prior_d = PriorDiscriminator(latent_dim)
        self.beta = beta

        self.reset_parameters()
        self.init_emb()

    def reset_parameters(self):
        reset(self.encoders)

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, xs):
        #neg_z = self.encoder.corrupt_forward(data)
        z = []
        for i in range(len(xs)):
            z.append(self.encoders[i](xs[i]))
        z = torch.cat(z,0)
        #self.pos_mask = mask
        #self.neg_mask = 1 - mask

        z_un = self.local_d(z)
        res_un = torch.matmul(z_un, z_un.t())

        return res_un, z

    def loss(self, res_un,z,mask):
        r"""Computes the mutual information maximization objective.
        Implemented from https://github.com/BioSysLab/deepSNEM"""
        pos_mask = mask
        neg_mask = 1 - mask
        log_2 = math.log(2.)
        p_samples = res_un * pos_mask
        q_samples = res_un * neg_mask

        Ep = log_2 - F.softplus(- p_samples)
        Eq = F.softplus(-q_samples) + q_samples - log_2

        Ep = (Ep * pos_mask).sum() / pos_mask.sum()
        Eq = (Eq * neg_mask).sum() / neg_mask.sum()
        LOCAL = Eq - Ep

        prior = torch.rand_like(z)

        term_a = torch.log(self.prior_d(prior)).mean()
        term_b = torch.log(1.0 - self.prior_d(z)).mean()
        PRIOR = -(term_a + term_b) * self.beta

        return LOCAL, PRIOR

    def encode(self, x,enc_id):
        z_latent = self.encoders[enc_id](x)
        return z_latent

    def L2Regularization(self, L2):

        L2Loss = 0.
        for encoder in self.encoders:
            L2Loss = L2Loss + encoder.L2Regularization(L2)

        #L2Loss = encoderL2
        return(L2Loss)


class Classifier(torch.nn.Module):
    def __init__(self, in_channel, hidden_layers, num_classes,drop_in=0.5, drop=0.2, bn=0.6, bias=True):
        super(Classifier, self).__init__()
        self.drop_in = drop_in
        self.num_hidden_layers = len(hidden_layers)
        self.bias = bias
        self.num_classes = num_classes
        self.bn = torch.nn.ModuleList()
        self.linear_layers = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()
        self.linear_layers.append(torch.nn.Linear(in_channel, hidden_layers[0], bias=bias))
        self.bn.append(torch.nn.BatchNorm1d(num_features=hidden_layers[0], momentum=bn))
        self.dropouts.append(torch.nn.Dropout(drop))
        self.activations.append(torch.nn.ReLU())
        for i in range(1, len(hidden_layers)):
            self.linear_layers.append(torch.nn.Linear(hidden_layers[i - 1], hidden_layers[i],
                                                      bias=bias))
            self.bn.append(torch.nn.BatchNorm1d(num_features=hidden_layers[i], momentum=bn))
            self.dropouts.append(torch.nn.Dropout(drop))
            self.activations.append(torch.nn.ReLU())
        self.out_linear = torch.nn.Linear(hidden_layers[i], num_classes, bias=bias)
        self.softmax = torch.nn.Softmax(dim=1)
        if self.drop_in > 0:
            self.InputDrop = torch.nn.Dropout(self.drop_in)

        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x):
        if self.drop_in > 0:
            x = self.InputDrop(x)
        for i in range(self.num_hidden_layers):
            x = self.linear_layers[i](x)
            x = self.bn[i](x)
            x = self.activations[i](x)
            x = self.dropouts[i](x)

        return self.softmax(self.out_linear(x))

    def L2Regularization(self, L2):

        weightLoss = 0.
        biasLoss = 0.
        for i in range(self.num_hidden_layers):
            weightLoss = weightLoss + L2 * torch.sum((self.linear_layers[i].weight) ** 2)
            if self.bias == True:
                biasLoss = biasLoss + L2 * torch.sum((self.linear_layers[i].bias) ** 2)
        L2Loss = biasLoss + weightLoss
        return (L2Loss)

class GaussianDecoder(torch.nn.Module):
    def __init__(self, latent_dim, hidden_layers, out_dim,dropRate=0.1, activation=None, bias=True):

        super(GaussianDecoder, self).__init__()

        self.bias = bias
        self.num_hidden_layers = len(hidden_layers)
        self.bn = torch.nn.ModuleList()
        self.linear_layers = torch.nn.ModuleList()
        self.linear_layers.append(torch.nn.Linear(latent_dim, hidden_layers[0], bias=bias))
        self.bn.append(torch.nn.BatchNorm1d(num_features=hidden_layers[0], momentum=0.6))
        for i in range(1, len(hidden_layers)):
            self.linear_layers.append(torch.nn.Linear(hidden_layers[i - 1], hidden_layers[i], bias=bias))
            self.bn.append(torch.nn.BatchNorm1d(num_features=hidden_layers[i], momentum=0.6))

        self.output_linear_mu = torch.nn.Linear(hidden_layers[-1],
                                                out_dim,
                                                bias=False)
        self.output_linear_var = torch.nn.Linear(hidden_layers[-1],
                                                 out_dim,
                                                 bias=False)

        if activation is not None:
            self.activation = activation
        self.dropout = torch.nn.Dropout(dropRate)

        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x):
        for i in range(self.num_hidden_layers):
            x = self.linear_layers[i](x)
            x = self.bn[i](x)
            x = self.activation(x)
            x = self.dropout(x)

        output_mu = self.output_linear_mu(x)
        output_var = self.output_linear_var(x)

        return output_mu,output_var

    def L2Regularization(self, L2):

        weightLoss = 0.
        biasLoss = 0.
        for i in range(self.num_hidden_layers):
            weightLoss = weightLoss + L2 * torch.sum((self.linear_layers[i].weight)**2)
            if self.bias==True:
                biasLoss = biasLoss + L2 * torch.sum((self.linear_layers[i].bias)**2)
        L2Loss = biasLoss + weightLoss
        return(L2Loss)

class SpeciesCovariate(torch.nn.Module):
    def __init__(self,latent_dim1, latent_dim2,dropRate=0.1):

        super(SpeciesCovariate, self).__init__()

        # self.MPL = torch.nn.Sequential(torch.nn.Linear(in_channel,latent_dim1//8,bias=True),
        #                                torch.nn.BatchNorm1d(num_features=latent_dim1//8, momentum=0.6),
        #                                torch.nn.LeakyReLU(),
        #                                torch.nn.Dropout(0.25),
        #                                torch.nn.Linear(latent_dim1 // 8, latent_dim1 // 4, bias=True),
        #                                torch.nn.BatchNorm1d(num_features=latent_dim1 // 4, momentum=0.6),
        #                                torch.nn.LeakyReLU(),
        #                                torch.nn.Dropout(0.25),
        #                                torch.nn.Linear(latent_dim1 // 4, latent_dim1 // 2, bias=True),
        #                                torch.nn.BatchNorm1d(num_features=latent_dim1 // 2, momentum=0.6),
        #                                torch.nn.LeakyReLU(),
        #                                torch.nn.Dropout(0.25),
        #                                torch.nn.Linear(latent_dim1 // 2, latent_dim1 , bias=True),
        #                                torch.nn.BatchNorm1d(num_features=latent_dim1 , momentum=0.6),
        #                                torch.nn.LeakyReLU())
        self.Vspecies = torch.nn.Linear(latent_dim1, latent_dim2, bias=False)
        self.dropOut = torch.nn.Dropout(dropRate)

        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, zbasal, zspecies):

        # zspecies = self.MPL(zspecies)
        z_cov = zbasal + self.dropOut(self.Vspecies(zspecies))

        return z_cov

    def Regularization(self, L = 1e-4):

        # Regularize L2 and also regularize not to be zero
        weightLoss = L * torch.sum((torch.square(self.Vspecies.weight)))

        return(weightLoss)

class GammaDecoder(torch.nn.Module):
    def __init__(self, latent_dim, hidden_layers, out_dim,dropRate=0.1, activation=None, bias=True):

        super(GammaDecoder, self).__init__()

        self.bias = bias
        self.num_hidden_layers = len(hidden_layers)
        self.bn = torch.nn.ModuleList()
        self.linear_layers = torch.nn.ModuleList()
        self.linear_layers.append(torch.nn.Linear(latent_dim, hidden_layers[0], bias=bias))
        self.bn.append(torch.nn.BatchNorm1d(num_features=hidden_layers[0], momentum=0.6))
        for i in range(1, len(hidden_layers)):
            self.linear_layers.append(torch.nn.Linear(hidden_layers[i - 1], hidden_layers[i], bias=bias))
            self.bn.append(torch.nn.BatchNorm1d(num_features=hidden_layers[i], momentum=0.6))

        self.output_linear_kappa = torch.nn.Linear(hidden_layers[-1],
                                                out_dim,
                                                bias=False)
        self.output_linear_theta = torch.nn.Linear(hidden_layers[-1],
                                                 out_dim,
                                                 bias=False)

        if activation is not None:
            self.activation = activation
        self.dropout = torch.nn.Dropout(dropRate)
        self.out_activation = torch.nn.ReLU()

        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x):
        for i in range(self.num_hidden_layers):
            x = self.linear_layers[i](x)
            x = self.bn[i](x)
            x = self.activation(x)
            x = self.dropout(x)

        output_kappa = self.out_activation(self.output_linear_kappa(x)) + 1e-4
        output_theta = self.out_activation(self.output_linear_theta(x)) + 1e-4

        return output_kappa,output_theta

    def L2Regularization(self, L2):

        weightLoss = 0.
        biasLoss = 0.
        for i in range(self.num_hidden_layers):
            weightLoss = weightLoss + L2 * torch.sum((self.linear_layers[i].weight)**2)
            if self.bias==True:
                biasLoss = biasLoss + L2 * torch.sum((self.linear_layers[i].bias)**2)
        L2Loss = biasLoss + weightLoss
        return(L2Loss)