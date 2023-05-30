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
        self.sample_topk_1_2 = cfg.ddpm.sample_topk_1_2
        self.sample_topk_1_4 = cfg.ddpm.sample_topk_1_4
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

    def downsample(self, batch_latent_data, slim=False):
        b_ref_points_sampled = []
        b_src_points_sampled = []
        b_ref_feats_sampled = []
        b_src_feats_sampled = []
        b_gt_corr_sampled = []
        b_corr_matrix = []
        b_feat_matrix = []
        b_init_corr_sampled = []
        b_init_corr_matrix = []
        for latent_dict in batch_latent_data:
            # Get the required data from the latent dictionary
            ref_points = latent_dict.get('ref_points_c')
            src_points = latent_dict.get('src_points_c')
            ref_feats = latent_dict.get('ref_feats_c')
            src_feats = latent_dict.get('src_feats_c')
            gt_corr = latent_dict.get('gt_node_corr_indices')
            init_ref_corr_indices = latent_dict.get('ref_node_corr_indices').unsqueeze(-1)
            init_src_corr_indices = latent_dict.get('src_node_corr_indices').unsqueeze(-1)
            # Make ref and src indices pairs
            init_corr_indices = torch.cat([init_ref_corr_indices, init_src_corr_indices], dim=1)

            
            # Randomly sample points from ref and src with length of ref_sample_num and src_sample_num
            ref_sample_indices = np.random.choice(ref_points.shape[0], self.ref_sample_num, replace=False)
            src_sample_indices = np.random.choice(src_points.shape[0], self.src_sample_num, replace=False)

            # Get gt_corr for sampled points
            gt_corr_sampled = []
            gt_corr_set = set(map(tuple, gt_corr.tolist()))
            init_corr_sampled = []
            init_corr_set = set(map(tuple, init_corr_indices.tolist()))

            for i, ref_index in enumerate(ref_sample_indices):
                for j, src_index in enumerate(src_sample_indices):
                    if (ref_index, src_index) in gt_corr_set:
                        gt_corr_sampled.append([i, j])
                    if (ref_index, src_index) in init_corr_set:
                        init_corr_sampled.append([i, j])
            
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
            
            init_corr_matrix = torch.full((ref_points_sampled.shape[0], src_points_sampled.shape[0]),-1.0)
            for pair in init_corr_sampled:
                init_corr_matrix[pair[0], pair[1]] = 1.0
            
            b_ref_points_sampled.append(ref_points_sampled.unsqueeze(0))
            b_src_points_sampled.append(src_points_sampled.unsqueeze(0))
            b_ref_feats_sampled.append(ref_feats_sampled.unsqueeze(0))
            b_src_feats_sampled.append(src_feats_sampled.unsqueeze(0))
            #b_gt_corr_sampled.append(torch.tensor(gt_corr_sampled))
            b_corr_matrix.append(corr_matrix.unsqueeze(0))
            b_feat_matrix.append(feat_matrix.unsqueeze(0))
            #b_init_corr_sampled.append(torch.tensor(init_corr_sampled))
            b_init_corr_matrix.append(init_corr_matrix.unsqueeze(0))


        d_dict = {
            'ref_points': torch.cat(b_ref_points_sampled, dim=0),
            'src_points': torch.cat(b_src_points_sampled, dim= 0),
            'ref_feats': torch.cat(b_ref_feats_sampled, dim=0),
            'src_feats': torch.cat(b_src_feats_sampled, dim=0),
            #'gt_corr': torch.cat(b_gt_corr_sampled, dim=0),
            'gt_corr_matrix': torch.cat(b_corr_matrix, dim=0),
            'feat_matrix': torch.cat(b_feat_matrix, dim=0),
            #'init_corr': torch.cat(b_init_corr_sampled, dim=0),
            'init_corr_matrix': torch.cat(b_init_corr_matrix , dim=0)
        }
        return d_dict
    
    
    def get_loss(self, batch_latent_data):

        d_dict = self.downsample(batch_latent_data)
        mat = d_dict.get('gt_corr_matrix').cuda()
        feats = d_dict.get('feat_matrix').cuda()
        loss = self.diffusion.get_loss(mat, feats)
        return {'loss': loss}
    
    def sample(self, latent_dict):
        latent_dict = [latent_dict]
        d_dict = self.downsample(latent_dict)
        #mat_T = torch.randn((1, self.ref_sample_num, self.src_sample_num)).cuda()
        mat_T = d_dict.get('init_corr_matrix').cuda()
        feats = d_dict.get('feat_matrix').cuda()
        pred_corr_mat = self.diffusion.sample(mat_T, feats).cpu()
        pred_corr = get_corr_from_matrix_topk(pred_corr_mat, self.sample_topk)
        pred_corr_1_2 = get_corr_from_matrix_topk(pred_corr_mat, self.sample_topk_1_2)
        pred_corr_1_4 = get_corr_from_matrix_topk(pred_corr_mat, self.sample_topk_1_4)
        return {
            'pred_corr_mat': pred_corr_mat,
            'pred_corr': pred_corr,
            'pred_corr_1_2': pred_corr_1_2,
            'pred_corr_1_4': pred_corr_1_4,
            'gt_corr_matrix': d_dict.get('gt_corr_matrix').squeeze(0),
            #'gt_corr': d_dict.get('gt_corr'),
            'init_corr_matrix': d_dict.get('init_corr_matrix').squeeze(0),
            #'init_corr': d_dict.get('init_corr'),
            'ref_points': d_dict.get('ref_points').squeeze(0),
            'src_points': d_dict.get('src_points').squeeze(0),
            }
        

def create_cordi(cfg):
    model = Cordi(cfg)
    return model
