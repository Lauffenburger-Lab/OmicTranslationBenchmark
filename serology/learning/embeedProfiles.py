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
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

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
# Gradient penalnty function
def compute_gradients(output, input):
    grads = torch.autograd.grad(output, input, create_graph=True)
    grads = grads[0].pow(2).mean()
    return grads
### Load cell-line data
print2log('Loading data...')
human_exprs = pd.read_csv('../data/human_exprs.csv',index_col=0)
human_metadata = pd.read_csv('../data/human_metadata.csv',index_col=0)
human_metadata["infect"] = human_metadata["infect"] - int(1)
primates_exprs = pd.read_csv('../data/primates_exprs.csv',index_col=0)
primates_metadata = pd.read_csv('../data/primates_metadata.csv',index_col=0)
Xh = torch.tensor(human_exprs.values).float().to(device)
Xp = torch.tensor(primates_exprs.values).float().to(device)
Yh = torch.tensor(human_metadata.loc[:,['trt','infect']].values).long()
Yp = torch.tensor(primates_metadata.loc[:,['Vaccine','ProtectBinary']].values).long()

gene_size_human = len(human_exprs.columns)
gene_size_primates = len(primates_exprs.columns)

z_human_base_all = torch.zeros((10,Xh.shape[0],latent_dim)).to(device)
z_human_all = torch.zeros((10,Xh.shape[0],latent_dim)).to(device)
z_primates_base_all = torch.zeros((10,Xp.shape[0],latent_dim)).to(device)
z_primates_all = torch.zeros((10,Xp.shape[0],latent_dim)).to(device)
for i in range(10):

    # Network
    encoder_1 = torch.load('../results_intermediate_encoders/models/10fold/encoder_human_%s.pt' % i).to(device)
    encoder_2 = torch.load('../results_intermediate_encoders/models/10fold/encoder_primates_%s.pt' % i).to(device)
    encoder_interm_1 = torch.load('../results_intermediate_encoders/models/10fold/encoder_intermediate_human_%s.pt' % i).to(device)
    encoder_interm_2 = torch.load('../results_intermediate_encoders/models/10fold/encoder_intermediate_primates_%s.pt' % i).to(device)
    
    encoder_1.eval()
    encoder_2.eval()
    encoder_interm_1.eval()
    encoder_interm_2.eval()

    z_latent_base_1 = encoder_1(Xh)
    z_latent_base_2 = encoder_2(Xp)

    z_latent_1 = encoder_interm_1(z_latent_base_1)
    z_latent_2 = encoder_interm_2(z_latent_base_2)

    z_human_base_all[i,:,:] = z_latent_base_1.detach()
    z_primates_base_all[i, :, :] = z_latent_base_2.detach()
    z_human_all[i, :, :] = z_latent_1.detach()
    z_primates_all[i, :, :] = z_latent_2.detach()
    print2log('Finished using model %s'%i)

### Covert to dataframe and save
z_human_base = torch.mean(z_human_base_all,0)
z_primates_base = torch.mean(z_primates_base_all,0)
z_human= torch.mean(z_human_all,0)
z_primates = torch.mean(z_primates_all,0)

z_human_base = pd.DataFrame(z_human_base.detach().cpu().numpy())
z_human_base.index = human_exprs.index
z_human_base.columns = ['z'+str(i) for i in range(latent_dim)]
z_human_base.to_csv('../results_intermediate_encoders/embs/z_human_base.csv')
z_primates_base = pd.DataFrame(z_primates_base.detach().cpu().numpy())
z_primates_base.index = primates_exprs.index
z_primates_base.columns = ['z'+str(i) for i in range(latent_dim)]
z_primates_base.to_csv('../results_intermediate_encoders/embs/z_primates_base.csv')
z_human= pd.DataFrame(z_human.detach().cpu().numpy())
z_human.index = human_exprs.index
z_human.columns = ['z'+str(i) for i in range(latent_dim)]
z_human.to_csv('../results_intermediate_encoders/embs/z_human.csv')
z_primates = pd.DataFrame(z_primates.detach().cpu().numpy())
z_primates.index = primates_exprs.index
z_primates.columns = ['z'+str(i) for i in range(latent_dim)]
z_primates.to_csv('../results_intermediate_encoders/embs/z_primates.csv')

### Find also std
z_human_base = torch.std(z_human_base_all,0)
z_primates_base = torch.std(z_primates_base_all,0)
z_human= torch.std(z_human_all,0)
z_primates = torch.std(z_primates_all,0)

z_human_base = pd.DataFrame(z_human_base.detach().cpu().numpy())
z_human_base.index = human_exprs.index
z_human_base.columns = ['z'+str(i) for i in range(latent_dim)]
z_human_base.to_csv('../results_intermediate_encoders/embs/z_human_base_std.csv')
z_primates_base = pd.DataFrame(z_primates_base.detach().cpu().numpy())
z_primates_base.index = primates_exprs.index
z_primates_base.columns = ['z'+str(i) for i in range(latent_dim)]
z_primates_base.to_csv('../results_intermediate_encoders/embs/z_primates_base_std.csv')
z_human= pd.DataFrame(z_human.detach().cpu().numpy())
z_human.index = human_exprs.index
z_human.columns = ['z'+str(i) for i in range(latent_dim)]
z_human.to_csv('../results_intermediate_encoders/embs/z_human_std.csv')
z_primates = pd.DataFrame(z_primates.detach().cpu().numpy())
z_primates.index = primates_exprs.index
z_primates.columns = ['z'+str(i) for i in range(latent_dim)]
z_primates.to_csv('../results_intermediate_encoders/embs/z_primates_std.csv')