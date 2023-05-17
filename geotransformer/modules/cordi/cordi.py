import torch
from torch.nn import Module
from geotransformer.modules.cordi.ddpm import *

class Cordi(Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.diffusion = DiffusionPoint(
            # backbone
            
            var_sched = VarianceSchedule(
                num_steps=args.num_steps,
                beta_1=args.beta_1,
                beta_T=args.beta_T,
                mode=args.sched_mode
                )
        )

    def get_loss(self, mat, feats):
        return self.diffusion.get_loss(mat, feats)
    
    def sample(self, mat_T, feats, flexibility):
        return self.diffusion.sample(mat_T, feats, flexibility)
    

def create_cordi(args):
    return Cordi(args)