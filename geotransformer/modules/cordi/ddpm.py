import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter, ModuleList, Linear
import torch.nn as nn
import numpy as np
import math
#import wandb

#from utils.common import *



class VarianceSchedule(Module):

    def __init__(self, num_steps, beta_1, beta_T, mode='linear'):
        super().__init__()
        assert mode in ('linear', )
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas


class SinusoidalPositionEmbeddings(Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings


class DiffusionPoint(Module):

    def __init__(self, net, var_sched:VarianceSchedule, time_emb, num_steps):
        super().__init__()
        self.net = net
        self.var_sched = var_sched
        self.time_emb = time_emb
        self.num_steps = num_steps

    def get_loss(self, x_0, feat0, feat1, t=None):
        """
        Args:
            x_0:  Input point cloud, (B, N, d).
            context:  Shape latent, (B, F).
        """
        batch_size, _, point_dim = x_0.size()
        if t == None:
            #t = self.var_sched.uniform_sample_t(batch_size)
            t = torch.randint(0, self.num_steps, (batch_size,), device='cuda').long()
        
        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t]

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)       # (B, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)   # (B, 1, 1)
        #torch.manual_seed(1001)
        #e_rand = torch.randn_like(x_0, memory_format=torch.contiguous_format)  # (B, N, d)
        e_rand = torch.randn_like(x_0)
        #e_theta = self.net(c0 * x_0 + c1 * e_rand, beta=beta, context_a=context_a, context_b=context_b)
        t = self.time_emb(t)
        e_theta = self.net(c0 * x_0 + c1 * e_rand, t, feat0, feat1)
        # collect all elements in matrix e_theta and show as a histogram in wandb
        #wandb.log({"e_theta": wandb.Histogram(e_theta.detach().cpu().numpy())})
        # calculate the mean and variance of e_theta and show in wandb
        mean = torch.mean(e_theta)
        var = torch.var(e_theta)
        #wandb.log({"mean": mean.detach().cpu().numpy(), "var": var.detach().cpu().numpy()})
        
        loss = F.mse_loss(e_theta.reshape(-1), e_rand.reshape(-1), reduction='mean')
        return loss
    
    def get_x_t(self, x_0, t):
        batch_size, _, point_dim = x_0.size()
        if t == None:
            t = self.var_sched.uniform_sample_t(batch_size)
        alpha_bar = self.var_sched.alpha_bars[t]

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)       # (B, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)   # (B, 1, 1)
        #torch.manual_seed(1001)
        #e_rand = torch.randn_like(x_0, memory_format=torch.contiguous_format)  # (B, N, d)
        e_rand = torch.randn_like(x_0)
        x_t = c0 * x_0 + c1 * e_rand
        
        return x_t

    def sample(self, x_T, feat0, feat1, flexibility=0.0, ret_traj=False):
        batch_size = feat0.size(0)
        traj = {self.var_sched.num_steps: x_T}
        for t in range(self.var_sched.num_steps, 0, -1):
            z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            sigma = self.var_sched.get_sigmas(t, flexibility)

            c0 = 1.0 / torch.sqrt(alpha)
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)
            if t == 1:
                pass
            x_t = traj[t]
            beta = self.var_sched.betas[[t]*batch_size]
            t_ = torch.full((1,), t, device='cuda', dtype=torch.long)
            t_ = self.time_emb(t_)
            e_theta = self.net(x_t, t_, feat0, feat1)
            x_next = c0 * (x_t - c1 * e_theta) + sigma * z
            traj[t-1] = x_next.detach()     # Stop gradient and save trajectory.
            traj[t] = traj[t].cpu()         # Move previous output to CPU memory.
            if not ret_traj:
                del traj[t]
        
        if ret_traj:
            return traj
        else:
            return traj[0]

