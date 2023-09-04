import torch

from torch.nn import Module, Linear, ReLU
from geotransformer.modules.cordi.ddpm import *
from geotransformer.modules.cordi.transformer import *
from geotransformer.datasets.registration.linemod.bop_utils import *
from geotransformer.modules.diffusion import create_diffusion
from fastnode2vec import Node2Vec, Graph

class Cordi(Module):

    def __init__(self, cfg):
        super(Cordi, self).__init__()
        self.cfg = cfg
        self.ref_sample_num = cfg.ddpm.ref_sample_num
        self.src_sample_num = cfg.ddpm.src_sample_num
        self.dist_embedding_dim = cfg.ddpm.dist_embedding_dim
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
        self.dist_emb_ref = nn.Sequential(
            nn.Linear(self.ref_sample_num, self.dist_embedding_dim),
            nn.ReLU(),
            nn.Linear(self.dist_embedding_dim, self.dist_embedding_dim)
        )
        self.dist_emb_src = nn.Sequential(
            nn.Linear(self.src_sample_num, self.dist_embedding_dim),
            nn.ReLU(),
            nn.Linear(self.dist_embedding_dim, self.dist_embedding_dim)
        )

    def downsample(self, batch_latent_data, slim=False):
        b_ref_points_sampled = []
        b_src_points_sampled = []
        b_ref_feats_sampled = []
        b_src_feats_sampled = []
        b_ref_mid_feats_sampled = []
        b_src_mid_feats_sampled = []
        b_feat_matrix = []
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
            gt_corr_score_matrix = latent_dict.get('gt_node_corr_score')
                    
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
            'gt_corr_score_matrix': torch.cat(b_gt_corr_score_matrix_sampled, dim=0),
            'gt_corr_matrix': torch.cat(b_gt_corr_matrix_sampled, dim=0),
            'feat_2d': torch.cat(b_feat_2d, dim=0)
        }
        return d_dict
    
    def node2vector(self, pcd, weighted=True, radius=0.1):
        pcd = pcd.cpu().numpy()
        
        weighted_node_list = []
        for i in range(pcd.shape[0]):
            for j in range(pcd.shape[0]):
                dist = np.linalg.norm(pcd[i] - pcd[j])
                # append tuple of nodes and weight in list
                weighted_node_list.append((str(i), str(j), dist))
        G = Graph(weighted_node_list, directed=False, weighted=True)
        node2vec = Node2Vec(G, dim=64, walk_length=20, window=10, workers=1)
        node2vec.train(epochs=10, verbose=False)
        vectors = []
        for i in range(pcd.shape[0]):
            vectors.append(node2vec.wv[str(i)])
        vectors = torch.tensor(vectors)
        return vectors
    
    def dist_embedding(self, ref_pcd, src_pcd):
        ref_dist_matrix = torch.zeros((ref_pcd.shape[0], ref_pcd.shape[0])).cuda()
        for i in range(ref_pcd.shape[0]):
            for j in range(ref_pcd.shape[0]):
                ref_dist_matrix[i, j] = torch.norm(ref_pcd[i] - ref_pcd[j])
        # normalize
        ref_dist_matrix = ref_dist_matrix / torch.max(ref_dist_matrix)
        # embedding n x n -> n x emb_dim
        ref_dist_emb = self.dist_emb_ref(ref_dist_matrix)

        src_dist_matrix = torch.zeros((src_pcd.shape[0], src_pcd.shape[0])).cuda()
        for i in range(src_pcd.shape[0]):
            for j in range(src_pcd.shape[0]):
                src_dist_matrix[i, j] = torch.norm(src_pcd[i] - src_pcd[j])
        # normalize
        src_dist_matrix = src_dist_matrix / torch.max(src_dist_matrix)
        # embedding n x n -> n x emb_dim
        src_dist_emb = self.dist_emb_src(src_dist_matrix)

        return ref_dist_emb, src_dist_emb     
    
    def get_loss(self, batch_latent_data):

        d_dict = self.downsample(batch_latent_data)
        #mat = d_dict.get('gt_corr_matrix').cuda()
        mat = d_dict.get('gt_corr_score_matrix').cuda().unsqueeze(1)
        ref_feats = d_dict.get('ref_feats').cuda()
        src_feats = d_dict.get('src_feats').cuda()
        ref_mid_feats = d_dict.get('ref_mid_feats').cuda()
        src_mid_feats = d_dict.get('src_mid_feats').cuda()
        feat_2d = d_dict.get('feat_2d').cuda()
        ref_points = d_dict.get('ref_points').squeeze(0)
        src_points = d_dict.get('src_points').squeeze(0)
        ref_dist_emb, src_dist_emb = self.dist_embedding(ref_points, src_points)
        #ref_node_vec = self.node2vector(ref_points).cuda()
        #src_node_vec = self.node2vector(src_points).cuda()
        feats = {}
        feats['ref_feats'] = ref_feats
        feats['src_feats'] = src_feats
        feats['ref_mid_feats'] = ref_mid_feats
        feats['src_mid_feats'] = src_mid_feats
        feats['feat_2d'] = feat_2d
        feats['ref_dist_emb'] = ref_dist_emb
        feats['src_dist_emb'] = src_dist_emb
        #feats['ref_node_vec'] = ref_node_vec.unsqueeze(0)
        #feats['src_node_vec'] = src_node_vec.unsqueeze(0)
        #loss = self.diffusion.get_loss(mat, ref_feats, src_feats)
        t = torch.randint(0, self.diffusion_new.num_timesteps, (mat.shape[0],), device='cuda')
        loss_dict = self.diffusion_new.training_losses(self.net, mat, t, feats)
        loss = loss_dict["loss"].mean()
        return {'loss': loss}
    
    def sample(self, latent_dict):
        latent_dict = [latent_dict]
        d_dict = self.downsample(latent_dict)
        #mat_T = torch.randn((1, self.ref_sample_num, self.src_sample_num)).cuda()
        mat_T = torch.randn_like(d_dict.get('gt_corr_score_matrix')).cuda().unsqueeze(1)
        #mat_T = d_dict.get('init_corr_matrix').cuda().unsqueeze(1)
        ref_feats = d_dict.get('ref_feats').cuda()
        src_feats = d_dict.get('src_feats').cuda()
        ref_mid_feats = d_dict.get('ref_mid_feats').cuda()
        src_mid_feats = d_dict.get('src_mid_feats').cuda()
        ref_points = d_dict.get('ref_points').squeeze(0)
        src_points = d_dict.get('src_points').squeeze(0)
        ref_dist_emb, src_dist_emb = self.dist_embedding(ref_points, src_points)
        #ref_node_vec = self.node2vector(ref_points).cuda()
        #src_node_vec = self.node2vector(src_points).cuda()
        feat_2d = d_dict.get('feat_2d').cuda()
        feats = {}
        feats['ref_feats'] = ref_feats
        feats['src_feats'] = src_feats
        feats['ref_mid_feats'] = ref_mid_feats
        feats['src_mid_feats'] = src_mid_feats
        feats['feat_2d'] = feat_2d
        feats['ref_dist_emb'] = ref_dist_emb
        feats['src_dist_emb'] = src_dist_emb
        #feats['ref_node_vec'] = ref_node_vec.unsqueeze(0)
        #feats['src_node_vec'] = src_node_vec.unsqueeze(0)
        #pred_corr_mat = self.diffusion.sample(mat_T, ref_feats, src_feats).cpu()
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
            'ref_points': d_dict.get('ref_points').squeeze(0),
            'src_points': d_dict.get('src_points').squeeze(0),
            }
        

def create_cordi(cfg):
    model = Cordi(cfg)
    return model
