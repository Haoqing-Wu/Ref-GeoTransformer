import torch
from torch.nn import Module
from geotransformer.modules.cordi.ddpm import *

class Cordi(Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.ref_sample_num = cfg.ddpm.ref_sample_num
        self.src_sample_num = cfg.ddpm.src_sample_num
        self.diffusion = DiffusionPoint(
            # backbone
            
            var_sched = VarianceSchedule(
                num_steps=cfg.ddpm.num_steps,
                beta_1=cfg.ddpm.beta_1,
                beta_T=cfg.ddpm.beta_T,
                mode=cfg.ddpm.sched_mode
                )
        )
    


    def get_loss(self, latent_dict):

        ref_feats = latent_dict['ref_feats_c']
        src_feats = latent_dict['src_feats_c']
        gt_corr = latent_dict['gt_node_corr_indices']
        # resize the feats and points

        return self.diffusion.get_loss(mat, feats)
    
    def sample(self, mat_T, feats, flexibility):
        return self.diffusion.sample(mat_T, feats, flexibility)
    
    

def create_cordi(cfg):
    return Cordi(cfg)