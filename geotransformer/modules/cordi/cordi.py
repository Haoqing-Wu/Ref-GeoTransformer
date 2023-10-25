import torch
import open3d as o3d

from torch.nn import Module, Linear, ReLU
from geotransformer.modules.cordi.ddpm import *
from geotransformer.modules.cordi.transformer import *
from geotransformer.datasets.registration.linemod.bop_utils import *
from geotransformer.modules.diffusion import create_diffusion
from geotransformer.modules.cordi.rotation_tools import compute_rotation_matrix_from_ortho6d

class Cordi(Module):

    def __init__(self, cfg):
        super(Cordi, self).__init__()
        self.cfg = cfg
        self.multi_hypothesis = cfg.ddpm.multi_hypothesis
        self.rotation_type = cfg.ddpm.rotation_type
        self.norm_factor = cfg.data.norm_factor

        self.diffusion_new = create_diffusion(timestep_respacing="ddim100")
        self.net = transformer(
                n_layers=cfg.ddpm_transformer.n_layers,
                n_heads=cfg.ddpm_transformer.n_heads,
                query_dimensions=cfg.ddpm_transformer.query_dimensions,
                time_emb_dim=cfg.ddpm.time_emb_dim,
                dino_emb_dim=cfg.dino.output_dim,
                recon_emb_dim=cfg.recon.feat_dims,
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
        b_init_corr_num = []
        b_gt_corr_score_matrix_sampled = []
        b_feat_2d = []
        b_rt = []
        for latent_dict in batch_latent_data:
            # Get the required data from the latent dictionary
            ref_points = latent_dict.get('ref_points_c')
            src_points = latent_dict.get('src_points_c')
            ref_feats = latent_dict.get('ref_feats_c')
            src_feats = latent_dict.get('src_feats_c')
            gt_corr = latent_dict.get('gt_node_corr_indices')
            init_ref_corr_indices = latent_dict.get('ref_node_corr_indices').unsqueeze(-1)
            init_src_corr_indices = latent_dict.get('src_node_corr_indices').unsqueeze(-1)
            gt_corr_score_matrix = latent_dict.get('gt_node_corr_score')
            # Make ref and src indices pairs
            init_corr_indices = torch.cat([init_ref_corr_indices, init_src_corr_indices], dim=1)

            
            # Randomly sample points from ref and src with length of ref_sample_num and src_sample_num
            if self.adaptive_size:
                ref_sample_indices = np.random.choice(ref_points.shape[0], int(ref_points.shape[0]*self.size_factor), replace=False)
                src_sample_indices = np.random.choice(src_points.shape[0], int(src_points.shape[0]*self.size_factor), replace=False)
            else:
                if self.ref_sample_num >= ref_points.shape[0]:
                    self.ref_sample_num = ref_points.shape[0]
                if self.src_sample_num >= src_points.shape[0]:
                    self.src_sample_num = src_points.shape[0]
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

            gt_corr_score_matrix_sampled = torch.full((ref_points_sampled.shape[0], src_points_sampled.shape[0]),0.0)
            for i in range(gt_corr_score_matrix_sampled.shape[0]):
                for j in range(gt_corr_score_matrix_sampled.shape[1]):
                    gt_corr_score_matrix_sampled[i, j] = gt_corr_score_matrix[ref_sample_indices[i], src_sample_indices[j]]
            '''
            feat_matrix = torch.zeros((ref_points_sampled.shape[0], src_points_sampled.shape[0], 
                                    ref_feats_sampled.shape[1]))
            # add the features of ref_feats_sampled and src_feats_sampled
            for i in range(ref_points_sampled.shape[0]):
                for j in range(src_points_sampled.shape[0]):
                    feat_matrix[i, j] = ref_feats_sampled[i] + src_feats_sampled[j]
            '''
            # concatinate the features of ref_feats_sampled and src_feats_sampled
            feat_matrix = torch.cat([ref_feats_sampled.unsqueeze(1).repeat(1, src_feats_sampled.shape[0], 1),
                                    src_feats_sampled.unsqueeze(0).repeat(ref_feats_sampled.shape[0], 1, 1)], dim=-1)
            feat_matrix = feat_matrix.view(ref_points_sampled.shape[0], src_points_sampled.shape[0], -1)
            
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
            b_init_corr_num.append(len(init_corr_sampled))
            b_gt_corr_score_matrix_sampled.append(gt_corr_score_matrix_sampled.unsqueeze(0))
            b_feat_2d.append(latent_dict.get('feat_2d').unsqueeze(0))
            b_rt.append(latent_dict.get('rt').unsqueeze(0))

        d_dict = {
            'ref_points': torch.cat(b_ref_points_sampled, dim=0),
            'src_points': torch.cat(b_src_points_sampled, dim=0),
            'ref_feats': torch.cat(b_ref_feats_sampled, dim=0),
            'src_feats': torch.cat(b_src_feats_sampled, dim=0),
            #'gt_corr': torch.cat(b_gt_corr_sampled, dim=0),
            'gt_corr_matrix': torch.cat(b_corr_matrix, dim=0),
            #'feat_matrix': torch.cat(b_feat_matrix, dim=0),
            #'init_corr': torch.cat(b_init_corr_sampled, dim=0),
            'init_corr_matrix': torch.cat(b_init_corr_matrix , dim=0),
            'init_corr_num': b_init_corr_num,
            'gt_corr_score_matrix': torch.cat(b_gt_corr_score_matrix_sampled, dim=0),
            'feat_2d': torch.cat(b_feat_2d, dim=0),
            'rt': torch.cat(b_rt, dim=0)
        }
        return d_dict
    
    
    def get_loss(self, d_dict):

        feat_2d = d_dict.get('feat_2d')
        feat_3d = d_dict.get('feat_3d')
        rt = d_dict.get('rt').unsqueeze(1)
        feats = {}
        feats['feat_2d'] = feat_2d
        feats['feat_3d'] = feat_3d
        t = torch.randint(0, self.diffusion_new.num_timesteps, (rt.shape[0],), device='cuda')
        loss_dict = self.diffusion_new.training_losses(self.net, rt, t, feats)
        loss = loss_dict["loss"].mean()
        return {'loss': loss}
    
    def sample(self, d_dict):

        feat_2d = d_dict.get('feat_2d')
        feat_3d = d_dict.get('feat_3d')
        feats = {}
        feats['feat_2d'] = feat_2d.repeat(self.multi_hypothesis, 1)
        feats['feat_3d'] = feat_3d.repeat(self.multi_hypothesis, 1)

        rt_T = torch.randn_like(d_dict.get('rt').repeat(self.multi_hypothesis, 1)).cuda().unsqueeze(1)

        traj = self.diffusion_new.p_sample_loop(
            self.net, rt_T.shape, rt_T, clip_denoised=False, model_kwargs=feats, progress=True, device='cuda'
        )
        pred_rt = traj[-1].cpu()
        pred_rt = pred_rt.squeeze(1)
        mean_rt = pred_rt.mean(dim=0)
        var_rt = pred_rt.var(dim=0)
        mean_var_rt = var_rt.mean(dim=0)

        return {
            'ref_points': d_dict.get('ref_points').squeeze(0),
            'src_points': d_dict.get('src_points').squeeze(0),
            'center_ref': d_dict.get('center_ref').squeeze(0),
            'pred_rt': mean_rt,
            'var_rt': mean_var_rt,
            'traj': traj,
            }

    def refine(self, output_dict):

        ref_points = output_dict.get('ref_points')
        src_points = output_dict.get('src_points')
        rt_init = output_dict.get('pred_rt')
      
        if self.rotation_type == 'quat':
            quat = rt_init[:4]
            trans = rt_init[4:]
            r = Rotation.from_quat(quat)
            rot = rot.as_matrix()
        elif self.rotation_type == 'mrp':
            mrp = rt_init[:3]
            trans = rt_init[3:]
            r = Rotation.from_mrp(mrp)
            rot = r.as_matrix()
        elif self.rotation_type == 'ortho6d':
            ortho6d = rt_init[:6]
            trans = rt_init[6:]
            # TODO: rewrite this part
            rot = compute_rotation_matrix_from_ortho6d(ortho6d.unsqueeze(0).cuda()).squeeze(0).cpu()
        
        init_trans = torch.from_numpy(get_transform_from_rotation_translation(rot, trans).astype(np.float32)).cuda()
        init_trans_icp = torch.from_numpy(get_transform_from_rotation_translation(rot, trans/self.norm_factor).astype(np.float32)).cuda()
        ref_points = ref_points.cpu().numpy()

        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(src_points.cpu().numpy())
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(ref_points)
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, 0.005, init_trans_icp.cpu().numpy(),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 20000))
        #output_dict['refined_trans'] = torch.from_numpy(reg_p2p.transformation.astype(np.float32)).cuda()
        refined_trans = torch.from_numpy(reg_p2p.transformation.astype(np.float32)).cuda()
        refined_trans[:3, 3] = refined_trans[:3, 3] * self.norm_factor
        output_dict['refined_trans'] = refined_trans
        output_dict['coarse_trans'] = init_trans   
        return output_dict

def create_cordi(cfg):
    model = Cordi(cfg)
    return model
