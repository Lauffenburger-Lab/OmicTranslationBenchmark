import torch
import torch.nn.functional as F
from torch.distributions import NegativeBinomial
from torch.distributions.gamma import Gamma
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,cross_validate,KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score,confusion_matrix,r2_score
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
from trainingUtils import MultipleOptimizer, MultipleScheduler, compute_kernel, compute_mmd,NBLoss,_convert_mean_disp_to_counts_logits, GammaLoss#,GaussLoss
from models import Decoder, SimpleEncoder,LocalDiscriminator,PriorDiscriminator,Classifier,SpeciesCovariate, VarDecoder
from evaluationUtils import r_square,get_cindex,pearson_r,pseudoAccuracy
import argparse
import logging
from captum.attr import IntegratedGradients
from captum.attr import LayerConductance
from captum.attr import NeuronConductance


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
print2log = logger.info

parser = argparse.ArgumentParser(prog='TransCompR mixed with ANNs approaches')
parser.add_argument('--filter_pcs', action='store', default=False)
parser.add_argument('--latent_dim', action='store', default=32)
# parser.add_argument('--outPattern', action='store')
args = parser.parse_args()
filter_pcs = args.filter_pcs
latent_dim = args.latent_dim
# outPattern = args.outPattern
if latent_dim is None:
    print2log('No latent dimension was defined.Filtered PCs will be used.')
    filter_pcs = True
else:
    latent_dim = int(latent_dim)

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
    
### Load data
print2log('Loading data...')
human_exprs = pd.read_csv('../data/human_exprs.csv',index_col=0)
human_metadata = pd.read_csv('../data/human_metadata.csv',index_col=0)
primates_exprs = pd.read_csv('../data/primates_exprs.csv',index_col=0)
primates_metadata = pd.read_csv('../data/primates_metadata.csv',index_col=0)
Xh = torch.tensor(human_exprs.values).double()
Xm = torch.tensor(primates_exprs.values).double()
Yh = torch.tensor(human_metadata.loc[:,['trt','protected']].values).long()
Ym = torch.tensor(primates_metadata.loc[:,['Vaccine','ProtectBinary']].values).long()

gene_size_human = len(human_exprs.columns)
gene_size_primates = len(primates_exprs.columns)


# ## Split in 10fold validation
k_folds=10
print2log('Begin TransCompR modeling...')
### Train TransCompR model with Decoder architecture
### Perform PCA for each each cell-line
print2log('Building PCA space for each species...')
pca1 = PCA()
pca2 = PCA()
pca_space_1 = pca1.fit(human_exprs.values)
pca_space_2 = pca2.fit(primates_exprs.values)
exp_var_pca1 = pca_space_1.explained_variance_ratio_
exp_var_pca2 = pca_space_2.explained_variance_ratio_
cum_sum_eigenvalues_1 = np.cumsum(exp_var_pca1)
cum_sum_eigenvalues_2 = np.cumsum(exp_var_pca2)
n1 = np.min(np.where(cum_sum_eigenvalues_1>=0.9)[0])
n2 = np.min(np.where(cum_sum_eigenvalues_2 >= 0.9)[0])
if filter_pcs==True:
    exp_var_pca1 = pca_space_1.explained_variance_ratio_
    cum_sum_eigenvalues_1 = np.cumsum(exp_var_pca1)
    nComps1 = np.min(np.where(cum_sum_eigenvalues_1>=0.99)[0])
    pca1 = PCA(n_components=nComps1)
    pca_space_1 = pca1.fit(human_exprs.values)
    exp_var_pca2 = pca_space_2.explained_variance_ratio_
    cum_sum_eigenvalues_2 = np.cumsum(exp_var_pca2)
    nComps2 = np.min(np.where(cum_sum_eigenvalues_2 >= 0.99)[0])
    pca2 = PCA(n_components=nComps2)
    pca_space_2 = pca2.fit(primates_exprs.values)
    pca_transformed_1 = pca_space_1.transform(human_exprs.values)
    pca_transformed_2 = pca_space_2.transform(primates_exprs.values)
else:
    nComps1 = latent_dim
    nComps2 = latent_dim
    pca1 = PCA(n_components=nComps1)
    pca2 = PCA(n_components=nComps2)
    pca_space_1 = pca1.fit(human_exprs.values)
    pca_space_2 = pca2.fit(primates_exprs.values)
    pca_transformed_1 = pca_space_1.transform(human_exprs.values)
    pca_transformed_2 = pca_space_2.transform(primates_exprs.values)
    exp_var_pca1 = pca_space_1.explained_variance_ratio_
    exp_var_pca2 = pca_space_2.explained_variance_ratio_


model_params = {'encoder_1_hiddens':[64],
                'encoder_2_hiddens':[64],
                'decoder_1_hiddens': [64],
                'decoder_2_hiddens': [128],
                'latent_dim1': nComps1,
                'latent_dim2': nComps2,
                'dropout_decoder': 0.2,
                'decoder_dropIn':0.0,
                'dropout_encoder':0.1,
                'encoder_activation':torch.nn.ELU(),#torch.nn.LeakyReLU(negative_slope=0.9),
                'decoder_activation':torch.nn.ELU(),
                'V_dropout':0.5,
                'intermediateEncoder1':[nComps1],
                'intermediateEncoder2':[nComps2],
                'intermediate_latent1':nComps1,
                'intermediate_latent2':nComps2,
                'intermediate_enc_l2_reg':1e-05,
                'intermediate_dropout':0.1,
                'inter_dropIn': 0.1,
                'state_class_hidden':[32,16,8],#[32,16,8,4],
                'state_class_drop_in':0.25,
                'state_class_drop':0.1,
                'no_states':2,
                'protect_class_hidden':[32,16,8],#[32,16,8,4,2],
                'protect_class_drop_in':0.25,
                'protect_class_drop':0.1,
                'protect_states':2,
                'species_class_hidden':[32,16,8],
                'species_class_drop_in':0.1,
                'species_class_drop':0.1,
                'no_species':2,
                'adv_class_hidden':[32,16,8],
                'adv_class_drop_in':0.2,
                'adv_class_drop':0.1,
                'no_adv_class':2,
                'encoding_lr':0.001,
                'adv_lr':0.001,
                'schedule_step_adv':300,
                'gamma_adv':0.5,
                'schedule_step_enc':300,
                'gamma_enc':0.8,
                'batch_size_1':35,
                'batch_size_2':15,
                'epochs':2000, # it was 1000 for only Vsp
                'prior_beta':1.0,
                'no_folds':k_folds,
                'v_reg':1e-04,
                'state_class_reg':1e-05,
                'species_class_reg':1e-04,
                'enc_l2_reg':1e-06,
                'dec_l2_reg':1e-06,
                'lambda_mi_loss':100.,
                'effsize_reg': 1.,
                'cosine_loss': 70,
                'adv_penalnty':1.,
                'reg_adv':100,
                'reg_classifier': 100.,
                'similarity_reg' : 40.,
                'adversary_steps':4,
                'autoencoder_wd': 0.,
                'adversary_wd': 0.}


for i in range(model_params["no_folds"]):
    xtrain_primates = torch.load('../data/10fold_cross_validation/train/xtrain_primates_%s.pt' % i).float().to(device)
    ytrain_primates = torch.load('../data/10fold_cross_validation/train/ytrain_primates_%s.pt' % i)
    # xtest_primates = torch.load('../data/10fold_cross_validation/train/xtest_primates_%s.pt' % i).float().to(device)
    # ytest_primates = torch.load('../data/10fold_cross_validation/train/ytest_primates_%s.pt' % i)
    xtrain_human = torch.load('../data/10fold_cross_validation/train/xtrain_human_%s.pt' % i).float().to(device)
    ytrain_human = torch.load('../data/10fold_cross_validation/train/ytrain_human_%s.pt' % i)
    # xtest_human = torch.load('../data/10fold_cross_validation/train/xtest_human_%s.pt' % i).float().to(device)
    # ytest_human = torch.load('../data/10fold_cross_validation/train/ytest_human_%s.pt' % i)

    gene_size_primates = xtrain_primates.shape[1]
    gene_size_human = xtrain_human.shape[1]

    # decoder_1 = torch.load( '../results/models/10fold/decoder_human_%s.pt' % i)
    # decoder_2 = torch.load( '../results/models/10fold/decoder_primates_%s.pt' % i)
    encoder_1 = torch.load('../results/models/10fold/encoder_human_%s.pt' % i)
    encoder_2 = torch.load('../results/models/10fold/encoder_primates_%s.pt' % i)
    # classifier = torch.load('../results/models/10fold/classifier_%s.pt' % i)
    # species_classifier = torch.load('../results/models/10fold/species_classifier_%s.pt' % i)
    protection_classifier = torch.load('../results/models/10fold/protection_classifier_%s.pt' % i)

    encoder_1.eval()
    encoder_2.eval()
    protection_classifier.eval()
    # classifier.eval()
    # species_classifier.eval()
    # decoder_1.eval()
    # decoder_2.eval()

    z_base_1 = encoder_1(xtrain_human).detach()
    z_base_2 = encoder_2(xtrain_primates).detach()
    latent_base_vectors = torch.cat((z_base_1, z_base_2), 0)

    print2log('Start classification importance for model %s'%i)
    # Classifier importance
    ig = IntegratedGradients(protection_classifier)
    # z_base_1.requires_grad_()
    attr1 = ig.attribute(z_base_1,target=1,n_steps=1000, return_convergence_delta=False)
    attr1 = attr1.detach().cpu().numpy()
    attr2 = ig.attribute(z_base_1,target=0,n_steps=1000, return_convergence_delta=False)
    attr2 = attr2.detach().cpu().numpy()
    df1 = pd.DataFrame(attr1)
    df1.columns = ['z'+str(i) for i in range(model_params['latent_dim1'])]
    df1.to_csv('../importance_results_cpa/important_scores_to_classify_human_protection_%s.csv'%i)
    df2 = pd.DataFrame(attr2)
    df2.columns = ['z'+str(i) for i in range(model_params['latent_dim1'])]
    df2.to_csv('../importance_results_cpa/important_scores_to_classify_human_nonprotection_%s.csv'%i)

    print2log('Start human feature importance for model %s'%i)
    # Per output latent variable input importance translation captum
    # 1st dimesion input
    # 2nd dimesion output
    ig = IntegratedGradients(encoder_1)
    hid_dim = model_params['latent_dim1']
    scores_human = torch.zeros((gene_size_human, hid_dim)).to(device)
    for z in range(hid_dim):
        # encoder_1.zero_grad()
        attr = ig.attribute(xtrain_human, target=z, n_steps=1000, return_convergence_delta=False)
        scores_human[:, z] = torch.mean(attr, 0)
        # if z % 2 == 0 :
        #     print2log(z)
    df_human = pd.DataFrame(scores_human.cpu().numpy())
    df_human.columns = ['z' + str(i) for i in range(model_params['latent_dim1'])]
    df_human.index = human_exprs.columns
    df_human.to_csv('../importance_results_cpa/important_scores_human_features_%s.csv'%i)

    print2log('Start primates feature importance for model %s'%i)
    # Per output latent variable input importance translation captum
    # 1st dimesion input
    # 2nd dimesion output
    ig = IntegratedGradients(encoder_2)
    scores_primates = torch.zeros((gene_size_primates, hid_dim)).to(device)
    for z in range(hid_dim):
        # encoder_1.zero_grad()
        attr = ig.attribute(xtrain_primates, target=z, n_steps=1000, return_convergence_delta=False)
        scores_primates[:, z] = torch.mean(attr, 0)
        # if z % 2 == 0 :
        #     print2log(z)
    df_primates = pd.DataFrame(scores_primates.cpu().numpy())
    df_primates.columns = ['z' + str(i) for i in range(model_params['latent_dim2'])]
    df_primates.index = primates_exprs.columns
    df_primates.to_csv('../importance_results_cpa/important_scores_primates_features_%s.csv'%i)

    print2log('Finished model %s'%i)

### Perform importance calculation for translation
### Load features of interest
interesting_feats = pd.read_csv('../importance_results_cpa/interesting_features.csv',index_col=0)
### Load data and trim
print2log('Translation feature importance')
print2log('Loading data...')
human_exprs = pd.read_csv('../data/human_exprs.csv',index_col=0)
interesting_human_feats_inds = [human_exprs.columns.get_loc(feat) for feat in interesting_feats[interesting_feats['species']=='human'].feature]
human_metadata = pd.read_csv('../data/human_metadata.csv',index_col=0)
primates_exprs = pd.read_csv('../data/primates_exprs.csv',index_col=0)
interesting_primates_feats_inds = [primates_exprs.columns.get_loc(feat) for feat in interesting_feats[interesting_feats['species']=='primates'].feature]
primates_metadata = pd.read_csv('../data/primates_metadata.csv',index_col=0)
Xh = torch.tensor(human_exprs.values).double()
Xm = torch.tensor(primates_exprs.values).double()
Yh = torch.tensor(human_metadata.loc[:,['trt','protected']].values).long()
Ym = torch.tensor(primates_metadata.loc[:,['Vaccine','ProtectBinary']].values).long()

# Define translator model
class TranslatorModel(torch.nn.Module):
    def __init__(self, encoder,Vcov,decoder):
        super(TranslatorModel, self).__init__()

        self.encoder = encoder
        self.Vcov = Vcov
        self.decoder = decoder

    def forward(self, x,z_species):
        z = self.encoder(x)
        z = self.Vcov(z,z_species)
        y = self.decoder(z)
        return y

print2log('Begin finding features to translate from primates to human')
for i in range(model_params["no_folds"]):
    xtrain_primates = torch.load('../data/10fold_cross_validation/train/xtrain_primates_%s.pt' % i).float().to(device)
    ytrain_primates = torch.load('../data/10fold_cross_validation/train/ytrain_primates_%s.pt' % i)
    # xtest_primates = torch.load('../data/10fold_cross_validation/train/xtest_primates_%s.pt' % i).float().to(device)
    # ytest_primates = torch.load('../data/10fold_cross_validation/train/ytest_primates_%s.pt' % i)
    xtrain_human = torch.load('../data/10fold_cross_validation/train/xtrain_human_%s.pt' % i).float().to(device)
    ytrain_human = torch.load('../data/10fold_cross_validation/train/ytrain_human_%s.pt' % i)
    # xtest_human = torch.load('../data/10fold_cross_validation/train/xtest_human_%s.pt' % i).float().to(device)
    # ytest_human = torch.load('../data/10fold_cross_validation/train/ytest_human_%s.pt' % i)
    z_species_1 = torch.cat((torch.ones(xtrain_primates.shape[0], 1),
                             torch.zeros(xtrain_primates.shape[0], 1)), 1).to(device)

    gene_size_primates = xtrain_primates.shape[1]
    gene_size_human = xtrain_human.shape[1]

    decoder_1 = torch.load( '../results/models/10fold/decoder_human_%s.pt' % i)
    # decoder_2 = torch.load( '../results/models/10fold/decoder_primates_%s.pt' % i)
    # encoder_1 = torch.load('../results/models/10fold/encoder_human_%s.pt' % i)
    encoder_2 = torch.load('../results/models/10fold/encoder_primates_%s.pt' % i)
    Vsp = torch.load('../results/models/10fold/Vspecies_%s.pt' % i).to(device)

    translator =TranslatorModel(encoder_2,Vsp,decoder_1).to(device)

    # encoder_1.eval()
    encoder_2.eval()
    decoder_1.eval()
    Vsp.eval()
    # decoder_2.eval()
    translator.eval()

    # Per output latent variable input importance translation captum
    # 1st dimesion input
    # 2nd dimesion output
    ig = IntegratedGradients(translator)
    scores_primates = torch.zeros((gene_size_primates, len(interesting_human_feats_inds))).to(device)
    ii = 0
    for feat in interesting_human_feats_inds:
        # encoder_1.zero_grad()
        attr, _ = ig.attribute((xtrain_primates,z_species_1), target=feat, n_steps=2000, return_convergence_delta=False)
        scores_primates[:, ii] = torch.mean(attr, 0)
        ii = ii + 1
        # if z % 2 == 0 :
        #     print2log(z)
    df_primates = pd.DataFrame(scores_primates.cpu().numpy())
    df_primates.columns = interesting_feats[interesting_feats['species']=='human'].feature
    df_primates.index = primates_exprs.columns
    df_primates.to_csv('../importance_results_cpa/translation/important_scores_primates_to_human_%s.csv' % i)
    print2log('Finished model %s' % i)