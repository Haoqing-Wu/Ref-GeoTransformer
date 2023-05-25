import torch

from torch.nn import Module, Linear, ReLU
from geotransformer.modules.cordi.ddpm import *
from geotransformer.modules.cordi.transformer import *
from geotransformer.datasets.registration.linemod.bop_utils import *

class Cordi(Module):

    def __init__(self, cfg):
        super(Cordi, self).__init__()
        self.cfg = cfg
        self.ref_sample_num = cfg.ddpm.ref_sample_num
        self.src_sample_num = cfg.ddpm.src_sample_num
        self.sample_topk = cfg.ddpm.sample_topk
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
            ),
            time_emb = nn.Sequential(
                SinusoidalPositionEmbeddings(cfg.ddpm.time_emb_dim),
                nn.Linear(cfg.ddpm.time_emb_dim, 
                          cfg.ddpm_transformer.n_heads*cfg.ddpm_transformer.query_dimensions),
                nn.ReLU()
            )
        )

    def downsample(self, latent_dict):
        # Get the required data from the latent dictionary
        ref_points = latent_dict.get('ref_points_c')
        src_points = latent_dict.get('src_points_c')
        ref_feats = latent_dict.get('ref_feats_c')
        src_feats = latent_dict.get('src_feats_c')
        gt_corr = latent_dict.get('gt_node_corr_indices')

        # Randomly sample points from ref and src with length of ref_sample_num and src_sample_num
        ref_sample_indices = np.random.choice(ref_points.shape[0], self.ref_sample_num, replace=False)
        src_sample_indices = np.random.choice(src_points.shape[0], self.src_sample_num, replace=False)

        # Get gt_corr for sampled points
        gt_corr_sampled = []
        gt_corr_set = set(map(tuple, gt_corr.tolist()))

        for i, ref_index in enumerate(ref_sample_indices):
            for j, src_index in enumerate(src_sample_indices):
                if (ref_index, src_index) in gt_corr_set:
                    gt_corr_sampled.append([i, j])
        
        # Get the sampled points and features
        ref_points_sampled = ref_points[ref_sample_indices]
        src_points_sampled = src_points[src_sample_indices]
        ref_feats_sampled = ref_feats[ref_sample_indices]
        src_feats_sampled = src_feats[src_sample_indices]

        corr_matrix = torch.full((ref_points_sampled.shape[0], src_points_sampled.shape[0]),-1.0)
        for pair in gt_corr_sampled:
            corr_matrix[pair[0], pair[1]] = 1.0
        feat_matrix = torch.zeros((ref_points_sampled.shape[0], src_points_sampled.shape[0], 
                                ref_feats_sampled.shape[1]))
        # add the features of ref_feats_sampled and src_feats_sampled
        for i in range(ref_points_sampled.shape[0]):
            for j in range(src_points_sampled.shape[0]):
                feat_matrix[i, j] = ref_feats_sampled[i] + src_feats_sampled[j]
        


        return {
            'ref_points': ref_points_sampled,
            'src_points': src_points_sampled,
            'ref_feats': ref_feats_sampled,
            'src_feats': src_feats_sampled,
            'gt_corr': torch.tensor(gt_corr_sampled),
            'gt_corr_matrix': corr_matrix,
            'feat_matrix': feat_matrix
        }
    
    def get_loss(self, latent_dict):

        d_dict = self.downsample(latent_dict)
        mat = d_dict.get('gt_corr_matrix')
        feats = d_dict.get('feat_matrix')
        mat = torch.unsqueeze(mat, dim=0).cuda()
        feats = torch.unsqueeze(feats, dim=0).cuda()
        loss = self.diffusion.get_loss(mat, feats)
        return {'loss': loss}
    
    def sample(self, latent_dict):
        d_dict = self.downsample(latent_dict)
        mat_T = torch.randn((1, self.ref_sample_num, self.src_sample_num)).cuda()
        feats = d_dict.get('feat_matrix')
        feats = torch.unsqueeze(feats, dim=0).cuda()
        pred_corr_mat = self.diffusion.sample(mat_T, feats).cpu()
        pred_corr = get_corr_from_matrix_topk(pred_corr_mat, self.sample_topk)
        return {
            'pred_corr_mat': pred_corr_mat,
            'pred_corr': pred_corr,
            'gt_corr_matrix': d_dict.get('gt_corr_matrix'),
            'gt_corr': d_dict.get('gt_corr'),
            'ref_points': d_dict.get('ref_points'),
            'src_points': d_dict.get('src_points')
            }
        

def create_cordi(cfg):
    model = Cordi(cfg)
    return model
