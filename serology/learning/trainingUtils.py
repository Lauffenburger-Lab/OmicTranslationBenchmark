from typing import Tuple
import numpy as np
import torch
import torch.nn as nn

dev = torch.device('cuda:0')

class MultipleOptimizer:
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()


class MultipleScheduler:
    def __init__(self, *op):
        self.optimizers = op

    def step(self):
        for op in self.optimizers:
            op.step()


def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)

    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / float(dim)
    return torch.exp(-kernel_input)


def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()

    return mmd  #

# Taken and implenented from https://github.com/facebookresearch/CPA
class NBLoss(torch.nn.Module):
    def __init__(self):
        super(NBLoss, self).__init__()

    def forward(self, mu, y, theta, eps=1e-8):
        """Negative binomial negative log-likelihood. It assumes targets `y` with n
        rows and d columns, but estimates `yhat` with n rows and 2d columns.
        The columns 0:d of `yhat` contain estimated means, the columns d:2*d of
        `yhat` contain estimated variances. This module assumes that the
        estimated mean and inverse dispersion are positive---for numerical
        stability, it is recommended that the minimum estimated variance is
        greater than a small number (1e-3).
        Parameters
        ----------
        yhat: Tensor
                Torch Tensor of reeconstructed data.
        y: Tensor
                Torch Tensor of ground truth data.
        eps: Float
                numerical stability constant.
        """
        if theta.ndimension() == 1:
            # In this case, we reshape theta for broadcasting
            theta = theta.view(1, theta.size(0))
        log_theta_mu_eps = torch.log(theta + mu + eps)
        res = (
                theta * (torch.log(theta + eps) - log_theta_mu_eps)
                + y * (torch.log(mu + eps) - log_theta_mu_eps)
                + torch.lgamma(y + theta)
                - torch.lgamma(theta)
                - torch.lgamma(y + 1)
        )
        res = _nan2inf(res)
        return -torch.mean(res)

# Taken and modified NBLoss from https://github.com/facebookresearch/CPA
class GammaLoss(torch.nn.Module):
    def __init__(self):
        super(GammaLoss, self).__init__()

    def forward(self, mu, y, theta, eps=1e-8):
        """Gamma log-likelihood. It assumes targets `y` with n
        rows and d columns, but estimates `yhat` with n rows and 2d columns.
        The columns 0:d of `yhat` contain estimated means, the columns d:2*d of
        `yhat` contain estimated variances. This module assumes that the
        estimated mean and inverse dispersion are positive---for numerical
        stability, it is recommended that the minimum estimated variance is
        greater than a small number (1e-3).
        Parameters
        ----------
        yhat: Tensor
                Torch Tensor of reeconstructed data.
        y: Tensor
                Torch Tensor of ground truth data.
        eps: Float
                numerical stability constant.
        """
        if theta.ndimension() == 1:
            # In this case, we reshape theta for broadcasting
            theta = theta.view(1, theta.size(0))
        mu_div_theta_eps = mu/(theta+eps)
        res = torch.log(theta+eps)* mu_div_theta_eps - (mu_div_theta_eps -1) * torch.log(y+eps) + y/(theta+eps) + torch.lgamma(mu_div_theta_eps + eps)
        res = _nan2inf(res)
        return -torch.mean(res)


def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)

def _convert_mean_disp_to_counts_logits(mu, theta, eps=1e-6):
    r"""NB parameterizations conversion
    Parameters
    ----------
    mu :
        mean of the NB distribution.
    theta :
        inverse overdispersion.
    eps :
        constant used for numerical log stability. (Default value = 1e-6)
    Returns
    -------
    type
        the number of failures until the experiment is stopped
        and the success probability.
    """
    assert (mu is None) == (
        theta is None
    ), "If using the mu/theta NB parameterization, both parameters must be specified"
    logits = (mu + eps).log() - (theta + eps).log()
    total_count = theta
    return total_count, logits

# class GaussLoss(torch.nn.Module):
#     def __init__(self):
#         super(GaussLoss, self).__init__()
#
#     def forward(self, mu, y, var, eps=1e-3):
#         if var.ndimension() == 1:
#             # In this case, we reshape theta for broadcasting
#             var = var.view(1, var.size(0))
#         res = 0.5*torch.log(torch.exp(var+eps)+1) + 0.5*(mu - y)**2/torch.exp(var+eps) + 1e-6
#         res = _nan2inf(res)
#         return -torch.mean(res)