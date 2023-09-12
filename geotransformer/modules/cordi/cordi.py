import torch
import open3d as o3d

from torch.nn import Module, Linear, ReLU
from geotransformer.modules.cordi.ddpm import *
from geotransformer.modules.cordi.transformer import *
from geotransformer.datasets.registration.linemod.bop_utils import *
from geotransformer.modules.diffusion import create_diffusion
from geotransformer.modules.geotransformer.geotransformer import GeometricStructureEmbedding
from positional_encodings.torch_encodings import PositionalEncoding1D

class Cordi(Module):

    def __init__(self, cfg):
        super(Cordi, self).__init__()
        self.cfg = cfg
        self.ref_sample_num = cfg.ddpm.ref_sample_num
        self.src_sample_num = cfg.ddpm.src_sample_num
        self.geo_embedding_dim = cfg.ddpm.geo_embedding_dim
        self.adaptive_size = cfg.ddpm.adaptive_size
        self.size_factor = cfg.ddpm.size_factor
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
            ),
            num_steps=cfg.ddpm.num_steps
        )
        self.diffusion_new = create_diffusion(timestep_respacing="ddim50")
        self.net = transformer(
                n_layers=cfg.ddpm_transformer.n_layers,
                n_heads=cfg.ddpm_transformer.n_heads,
                query_dimensions=cfg.ddpm_transformer.query_dimensions,
                feed_forward_dimensions=cfg.ddpm_transformer.feed_forward_dimensions,
                activation=cfg.ddpm_transformer.activation,
                time_emb_dim=cfg.ddpm.time_emb_dim
            )
        self.geo_embedding = GeometricStructureEmbedding(
            hidden_dim=cfg.ddpm.geo_embedding_dim,
            sigma_d=cfg.geotransformer.sigma_d,
            sigma_a=cfg.geotransformer.sigma_a,
            angle_k=cfg.geotransformer.angle_k,
            reduction_a=cfg.geotransformer.reduction_a
        )
        self.geo_proj_ref = nn.Sequential(
            nn.Linear(self.ref_sample_num*cfg.ddpm.geo_embedding_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 256)
        )
        self.geo_proj_src = nn.Sequential(
            nn.Linear(self.src_sample_num*cfg.ddpm.geo_embedding_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 256)
        )
        self.voxel_emb = SinusoidalPositionEmbeddings3D(128)

    def downsample(self, batch_latent_data, slim=False):
        b_ref_points_sampled = []
        b_src_points_sampled = []
        b_ref_feats_sampled = []
        b_src_feats_sampled = []
        b_ref_mid_feats_sampled = []
        b_src_mid_feats_sampled = []
        b_feat_matrix = []
        b_init_corr_matrix = []
        b_init_corr_score_matrix = []
        b_gt_corr_score_matrix_sampled = []
        b_gt_corr_matrix_sampled =[]
        b_feat_2d = []
        for latent_dict in batch_latent_data:
            # Get the required data from the latent dictionary
            ref_points = latent_dict.get('ref_points_sel_c')
            src_points = latent_dict.get('src_points_sel_c')
            ref_feats = latent_dict.get('ref_feats_c')
            src_feats = latent_dict.get('src_feats_c')
            ref_mid_feats = latent_dict.get('ref_mid_feats_sel')
            src_mid_feats = latent_dict.get('src_mid_feats_sel')
            gt_corr = latent_dict.get('gt_node_corr_indices')
            init_ref_corr_indices = latent_dict.get('ref_node_corr_indices').unsqueeze(-1)
            init_src_corr_indices = latent_dict.get('src_node_corr_indices').unsqueeze(-1)
            ref_no_match_indices = latent_dict.get('ref_no_match_indices').unsqueeze(-1)
            src_no_match_indices = latent_dict.get('src_no_match_indices').unsqueeze(-1)
            gt_corr_score_matrix = latent_dict.get('gt_node_corr_score')
            init_corr_score_matrix = latent_dict.get('node_corr_score')
            
            init_corr_matrix = torch.zeros((ref_points.shape[0], src_points.shape[0])).cuda()
            init_corr_matrix[init_ref_corr_indices, init_src_corr_indices] = 1
            init_corr_matrix[ref_no_match_indices, src_no_match_indices] = 1

            # Get gt corr matrix from gt corr pairs
            gt_corr_matrix = torch.zeros((ref_points.shape[0], src_points.shape[0])).cuda()
            gt_corr_matrix[gt_corr[:, 0], gt_corr[:, 1]] = 1

            # Get the sampled points and features
            ref_points_sampled = ref_points
            src_points_sampled = src_points
            ref_feats_sampled = ref_feats
            src_feats_sampled = src_feats
            ref_mid_feats_sampled = ref_mid_feats
            src_mid_feats_sampled = src_mid_feats


            gt_corr_score_matrix_sampled = gt_corr_score_matrix

            # concatinate the features of ref_feats_sampled and src_feats_sampled
            feat_matrix = torch.cat([ref_feats_sampled.unsqueeze(1).repeat(1, src_feats_sampled.shape[0], 1),
                                    src_feats_sampled.unsqueeze(0).repeat(ref_feats_sampled.shape[0], 1, 1)], dim=-1)
            feat_matrix = feat_matrix.view(ref_points_sampled.shape[0], src_points_sampled.shape[0], -1)
            
            
            b_ref_points_sampled.append(ref_points_sampled.unsqueeze(0))
            b_src_points_sampled.append(src_points_sampled.unsqueeze(0))
            b_ref_feats_sampled.append(ref_feats_sampled.unsqueeze(0))
            b_src_feats_sampled.append(src_feats_sampled.unsqueeze(0))
            b_ref_mid_feats_sampled.append(ref_mid_feats_sampled.unsqueeze(0))
            b_src_mid_feats_sampled.append(src_mid_feats_sampled.unsqueeze(0))
            b_feat_matrix.append(feat_matrix.unsqueeze(0))
            b_init_corr_matrix.append(init_corr_matrix.unsqueeze(0))
            b_init_corr_score_matrix.append(init_corr_score_matrix.unsqueeze(0))
            b_gt_corr_score_matrix_sampled.append(gt_corr_score_matrix_sampled.unsqueeze(0))
            b_gt_corr_matrix_sampled.append(gt_corr_matrix.unsqueeze(0))
            b_feat_2d.append(latent_dict.get('feat_2d').unsqueeze(0))

        d_dict = {
            'ref_points': torch.cat(b_ref_points_sampled, dim=0),
            'src_points': torch.cat(b_src_points_sampled, dim=0),
            'ref_feats': torch.cat(b_ref_feats_sampled, dim=0),
            'src_feats': torch.cat(b_src_feats_sampled, dim=0),
            'ref_mid_feats': torch.cat(b_ref_mid_feats_sampled, dim=0),
            'src_mid_feats': torch.cat(b_src_mid_feats_sampled, dim=0),
            'init_corr_matrix': torch.cat(b_init_corr_matrix, dim=0),   
            'init_corr_score_matrix': torch.cat(b_init_corr_score_matrix, dim=0),
            'gt_corr_score_matrix': torch.cat(b_gt_corr_score_matrix_sampled, dim=0),
            'gt_corr_matrix': torch.cat(b_gt_corr_matrix_sampled, dim=0),
            'feat_2d': torch.cat(b_feat_2d, dim=0)
        }
        return d_dict
    

    def geometric_embedding(self, ref_pcd, src_pcd):
        ref_geo_emb = self.geo_embedding(ref_pcd)
        src_geo_emb = self.geo_embedding(src_pcd)
        ref_geo_emb = self.geo_proj_ref(ref_geo_emb.reshape(ref_geo_emb.shape[0], ref_geo_emb.shape[1], -1))
        src_geo_emb = self.geo_proj_src(src_geo_emb.reshape(src_geo_emb.shape[0], src_geo_emb.shape[1], -1))

        return ref_geo_emb, src_geo_emb
    
    def voxel_embedding(self, pcd_in, voxel_size=0.1):
        for batch in range(pcd_in.shape[0]):
            pcd = pcd_in[batch]
            pcd = pcd.cpu().numpy()
            pcd_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_o3d, voxel_size)
            voxels = np.ndarray((pcd.shape[0], 3))
            for i in range(pcd.shape[0]):
                voxel_index = voxel_grid.get_voxel(pcd[i])
                voxels[i] = voxel_index
            voxel_grid = torch.from_numpy(voxels).cuda()
            if batch == 0:
                voxel_grids = voxel_grid.unsqueeze(0)
            else:
                voxel_grids = torch.cat([voxel_grids, voxel_grid.unsqueeze(0)], dim=0)

        # positional encoding
        emb = self.voxel_emb(voxel_grids)

        return emb

    def knn_indices(self, pcd, k):
        dists = torch.cdist(pcd, pcd)
        knn_indices = torch.topk(dists, k=k, dim=2, largest=False).indices
        return knn_indices
    
    def add_feature_from_indices(self, feats, indices):
        # feats: (B, N, C)
        # indices: (B, N, K)
        # return: (B, N, C)
        B, N, K = indices.shape
        C = feats.shape[2]
        indices = indices.reshape(B, N*K)
        feats = feats.reshape(B*N, C)
        knn_feats = feats[indices, :]
        knn_feats = knn_feats.reshape(B, N, K, C)
        knn_feats = torch.mean(knn_feats, dim=2)
        return knn_feats
    
    def create_weighted_mask(self, mask, w1, w2):
        # replace 1 with w1, 0 with w2
        mask = mask * w1 + (1 - mask) * w2
        return mask
        

    
    def get_loss(self, batch_latent_data):

        d_dict = self.downsample(batch_latent_data)
    
        mat = d_dict.get('gt_corr_score_matrix').cuda().unsqueeze(1)
        mask = d_dict.get('init_corr_matrix').cuda()
        weighted_mask = self.create_weighted_mask(mask, 1.0, 0.1)
        ref_feats = d_dict.get('ref_feats').cuda()
        src_feats = d_dict.get('src_feats').cuda()
        ref_mid_feats = d_dict.get('ref_mid_feats').cuda()
        src_mid_feats = d_dict.get('src_mid_feats').cuda()
        feat_2d = d_dict.get('feat_2d').cuda()
        ref_points = d_dict.get('ref_points')
        src_points = d_dict.get('src_points')
        ref_dist_emb, src_dist_emb = self.geometric_embedding(ref_points, src_points)
        ref_voxel_emb = self.voxel_embedding(ref_points)
        src_voxel_emb = self.voxel_embedding(src_points)
        ref_knn_indices = self.knn_indices(ref_points, 3)
        src_knn_indices = self.knn_indices(src_points, 3)
        ref_knn_emb = self.voxel_emb(ref_knn_indices)
        src_knn_emb = self.voxel_emb(src_knn_indices)
        ref_knn_feats = self.add_feature_from_indices(ref_feats, ref_knn_indices)
        src_knn_feats = self.add_feature_from_indices(src_feats, src_knn_indices)

        feats = {}
        feats['ref_feats'] = ref_feats
        feats['src_feats'] = src_feats
        feats['ref_mid_feats'] = ref_mid_feats
        feats['src_mid_feats'] = src_mid_feats
        feats['feat_2d'] = feat_2d
        feats['ref_dist_emb'] = ref_dist_emb
        feats['src_dist_emb'] = src_dist_emb
        feats['ref_voxel_emb'] = ref_voxel_emb
        feats['src_voxel_emb'] = src_voxel_emb
        feats['ref_knn_emb'] = ref_knn_emb
        feats['src_knn_emb'] = src_knn_emb
        feats['ref_knn_feats'] = ref_knn_feats
        feats['src_knn_feats'] = src_knn_feats
        feats['mask'] = mask
        feats['weighted_mask'] = weighted_mask

        t = torch.randint(0, self.diffusion_new.num_timesteps, (mat.shape[0],), device='cuda')
        loss_dict = self.diffusion_new.training_losses(self.net, mat, t, feats)
        loss = loss_dict["loss"].mean()
        return {'loss': loss}
    
    def sample(self, latent_dict):
        latent_dict = [latent_dict]
        d_dict = self.downsample(latent_dict)
        #mat_T = torch.randn((1, self.ref_sample_num, self.src_sample_num)).cuda()
        mat_T = torch.randn_like(d_dict.get('gt_corr_score_matrix')).cuda().unsqueeze(1)
        mask = d_dict.get('init_corr_matrix').cuda()
        weighted_mask = self.create_weighted_mask(mask, 1.0, 0.1)
        ref_feats = d_dict.get('ref_feats').cuda()
        src_feats = d_dict.get('src_feats').cuda()
        ref_mid_feats = d_dict.get('ref_mid_feats').cuda()
        src_mid_feats = d_dict.get('src_mid_feats').cuda()
        ref_points = d_dict.get('ref_points')
        src_points = d_dict.get('src_points')
        ref_dist_emb, src_dist_emb = self.geometric_embedding(ref_points, src_points)
        ref_voxel_emb = self.voxel_embedding(ref_points)
        src_voxel_emb = self.voxel_embedding(src_points)
        ref_knn_indices = self.knn_indices(ref_points, 3)
        src_knn_indices = self.knn_indices(src_points, 3)
        ref_knn_emb = self.voxel_emb(ref_knn_indices)
        src_knn_emb = self.voxel_emb(src_knn_indices)
        ref_knn_feats = self.add_feature_from_indices(ref_feats, ref_knn_indices)
        src_knn_feats = self.add_feature_from_indices(src_feats, src_knn_indices)

        feat_2d = d_dict.get('feat_2d').cuda()

        feats = {}
        feats['ref_feats'] = ref_feats
        feats['src_feats'] = src_feats
        feats['ref_mid_feats'] = ref_mid_feats
        feats['src_mid_feats'] = src_mid_feats
        feats['feat_2d'] = feat_2d
        feats['ref_dist_emb'] = ref_dist_emb
        feats['src_dist_emb'] = src_dist_emb
        feats['ref_voxel_emb'] = ref_voxel_emb
        feats['src_voxel_emb'] = src_voxel_emb
        feats['ref_knn_emb'] = ref_knn_emb
        feats['src_knn_emb'] = src_knn_emb
        feats['ref_knn_feats'] = ref_knn_feats
        feats['src_knn_feats'] = src_knn_feats
        feats['mask'] = mask
        feats['weighted_mask'] = weighted_mask

        pred_corr_mat = self.diffusion_new.p_sample_loop(
            self.net, mat_T.shape, mat_T, clip_denoised=False, model_kwargs=feats, progress=True, device='cuda'
        ).cpu()
        pred_corr_mat = pred_corr_mat.squeeze(1)

        pred_corr = get_corr_from_matrix_topk(pred_corr_mat, 40)
        pred_corr_1_2 = get_corr_from_matrix_topk(pred_corr_mat, 32)
        pred_corr_1_4 = get_corr_from_matrix_topk(pred_corr_mat, 16)
        pred_corr_0_9, num_corr_0_9 = get_corr_from_matrix_gt(pred_corr_mat, 0.9, 1.1)
        pred_corr_0_95, num_corr_0_95 = get_corr_from_matrix_gt(pred_corr_mat, 0.95, 1.05)
        pred_corr_1, num_corr_1 = get_corr_from_matrix_gt(pred_corr_mat, 0.98, 1.02)
        return {
            'pred_corr_mat': pred_corr_mat,
            'pred_corr': pred_corr,
            'pred_corr_1_2': pred_corr_1_2,
            'pred_corr_1_4': pred_corr_1_4,
            'pred_corr_0_9': pred_corr_0_9,
            'pred_corr_0_95': pred_corr_0_95,
            'pred_corr_1': pred_corr_1,
            'num_corr_0_9': num_corr_0_9,
            'num_corr_0_95': num_corr_0_95,
            'num_corr_1': num_corr_1,
            'gt_corr_matrix': d_dict.get('gt_corr_matrix').squeeze(0),
            'gt_corr_score_matrix': d_dict.get('gt_corr_score_matrix').squeeze(0),
            'init_corr_matrix': d_dict.get('init_corr_matrix').squeeze(0),
            'init_corr_score_matrix': d_dict.get('init_corr_score_matrix').squeeze(0),
            'ref_points': d_dict.get('ref_points').squeeze(0),
            'src_points': d_dict.get('src_points').squeeze(0),
            }
        
class SinusoidalPositionEmbeddings3D(Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        

    def forward(self, voxels):
        # voxels: (N, 3)
        
        device = voxels.device
        part_dim = self.dim // 6
        embeddings = math.log(10000) / (part_dim - 1)
        embeddings = torch.exp(torch.arange(part_dim, device=device) * -embeddings)

        
        x = voxels[:, :, 0]
        y = voxels[:, :, 1]
        z = voxels[:, :, 2]
        x_emb = x.unsqueeze(-1) * embeddings.unsqueeze(0)
        y_emb = y.unsqueeze(-1) * embeddings.unsqueeze(0)
        z_emb = z.unsqueeze(-1) * embeddings.unsqueeze(0)

        emb = torch.cat([x_emb.sin(), x_emb.cos(), y_emb.sin(), y_emb.cos(), z_emb.sin(), z_emb.cos()], dim=-1)
        # pad 0s to the end make it to desired dim
        pad = torch.zeros((emb.shape[0], emb.shape[1], self.dim - emb.shape[-1])).cuda()
        emb = torch.cat([emb, pad], dim=-1)
           
        return emb.to(torch.float32)

def create_cordi(cfg):
    model = Cordi(cfg)
    return model
