import torch

from torch.nn import Module
from geotransformer.modules.cordi.ddpm import *
from geotransformer.modules.cordi.transformer import *
from geotransformer.datasets.registration.linemod.bop_utils import *

class Cordi(Module):

    def __init__(self, cfg):
        super(Cordi, self).__init__()
        self.cfg = cfg
        self.ref_sample_num = cfg.ddpm.ref_sample_num
        self.src_sample_num = cfg.ddpm.src_sample_num
        self.diffusion = DiffusionPoint(
            net = transformer(
                n_layers=cfg.ddpm_transformer.n_layers,
                n_heads=cfg.ddpm_transformer.n_heads,
                query_dimensions=cfg.ddpm_transformer.query_dimensions,
                feed_forward_dimensions=cfg.ddpm_transformer.feed_forward_dimensions,
                activation=cfg.ddpm_transformer.activation
            ),
            var_sched = VarianceSchedule(
                num_steps=cfg.ddpm.num_steps,
                beta_1=cfg.ddpm.beta_1,
                beta_T=cfg.ddpm.beta_T,
                mode=cfg.ddpm.sched_mode
            )
        )
    
    def get_loss(self, latent_dict):

        ref_points = latent_dict['ref_points_c']
        src_points = latent_dict['src_points_c']
        ref_feats = latent_dict['ref_feats_c']
        src_feats = latent_dict['src_feats_c']
        gt_corr = latent_dict['gt_node_corr_indices']
        # randomly sample points from ref and src with length of ref_sample_num and src_sample_num
        ref_sample_indices = torch.randint(0, ref_points.shape[0], self.ref_sample_num)
        src_sample_indices = torch.randint(0, src_points.shape[0], self.src_sample_num)
        # get gt_corr for sampled points
        gt_corr_sample = gt_corr[ref_sample_indices, src_sample_indices]
        
        ref_sample_points = ref_points[ref_sample_indices]
        src_sample_points = src_points[src_sample_indices]
        ref_sample_feats = ref_feats[ref_sample_indices]
        src_sample_feats = src_feats[src_sample_indices]
        



        #return self.diffusion.get_loss(mat, feats)
    
    def sample(self, mat_T, feats, flexibility):
        return self.diffusion.sample(mat_T, feats, flexibility)
        

def create_cordi(cfg):
    model = Cordi(cfg)
    return model
