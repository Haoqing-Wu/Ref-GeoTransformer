import os
import os.path as osp
import argparse

from easydict import EasyDict as edict

from geotransformer.utils.common import ensure_dir


_C = edict()

# common
_C.seed = 7351

# dirs
_C.working_dir = osp.dirname(osp.realpath(__file__))
_C.root_dir = osp.dirname(osp.dirname(_C.working_dir))
_C.exp_name = osp.basename(_C.working_dir)
_C.output_dir = osp.join(_C.root_dir, "output", _C.exp_name)
_C.snapshot_encoder_dir = osp.join(_C.output_dir, "snapshots/encoder")
_C.snapshot_ddpm_dir = osp.join(_C.output_dir, "snapshots/ddpm")
_C.snapshot_recon_dir = osp.join(_C.output_dir, "snapshots/recon")
_C.log_dir = osp.join(_C.output_dir, "logs")
_C.event_dir = osp.join(_C.output_dir, "events")
_C.result_pcd_dir = osp.join(_C.output_dir, "result/pcd")
_C.result_csv_dir = osp.join(_C.output_dir, "result/csv")

ensure_dir(_C.output_dir)
ensure_dir(_C.snapshot_encoder_dir)
ensure_dir(_C.snapshot_ddpm_dir)
ensure_dir(_C.snapshot_recon_dir)
ensure_dir(_C.log_dir)
ensure_dir(_C.event_dir)
ensure_dir(_C.result_pcd_dir)
ensure_dir(_C.result_csv_dir)

# wandb ddpm
_C.wandb_ddpm = edict()
_C.wandb_ddpm.enable = False
_C.wandb_ddpm.project = "cordi_pose_base"
_C.wandb_ddpm.name = "lm6_pbr_b16_S_res_t_o6d_mh16_400step_norm_d256_add_dino_foldnet_4l_icp"

# wandb recon
_C.wandb_recon = edict()
_C.wandb_recon.enable = False
_C.wandb_recon.project = "cordi_recon_comp"
_C.wandb_recon.name = "lm_pbr_cls_b32_or100_foldnet_plane_k64_d512"

# data
_C.data = edict()
_C.data.dataset = "linemod"
_C.data.norm_factor = 1.0
_C.data.residual_t = True

# train data
_C.train = edict()
_C.train.batch_size = 16
_C.train.num_workers = 8
_C.train.noise_magnitude = 0.05
_C.train.class_indices = "all"

# test data
_C.test = edict()
_C.test.batch_size = 1
_C.test.num_workers = 8
_C.test.noise_magnitude = 0.05
_C.test.class_indices = "all"

# evaluation
_C.eval = edict()
_C.eval.acceptance_overlap = 0.0
_C.eval.acceptance_radius = 0.01
_C.eval.inlier_ratio_threshold = 0.05
_C.eval.rre_threshold = 1.0
_C.eval.rte_threshold = 0.1

# ransac
_C.ransac = edict()
_C.ransac.distance_threshold = 0.05
_C.ransac.num_points = 3
_C.ransac.num_iterations = 1000

# optim
_C.optim = edict()
_C.optim.lr = 1e-4
_C.optim.weight_decay = 1e-6
_C.optim.warmup_steps = 1000
_C.optim.eta_init = 0.1
_C.optim.eta_min = 0.01
_C.optim.max_iteration = 500000
_C.optim.snapshot_steps = 5000
_C.optim.grad_acc_steps = 1

# model - backbone
_C.backbone = edict()
_C.backbone.num_stages = 3
_C.backbone.init_voxel_size = 0.05
_C.backbone.kernel_size = 15
_C.backbone.base_radius = 3.0
_C.backbone.base_sigma = 2.0
_C.backbone.init_radius = _C.backbone.base_radius * _C.backbone.init_voxel_size
_C.backbone.init_sigma = _C.backbone.base_sigma * _C.backbone.init_voxel_size
_C.backbone.group_norm = 32
_C.backbone.input_dim = 1
_C.backbone.init_dim = 64
_C.backbone.output_dim = 256

# model - Global
_C.model = edict()
_C.model.ground_truth_matching_radius = 0.05
_C.model.num_points_in_patch = 128
_C.model.num_sinkhorn_iterations = 100

# model - Coarse Matching
_C.coarse_matching = edict()
_C.coarse_matching.num_targets = 128
_C.coarse_matching.overlap_threshold = 0.1
_C.coarse_matching.num_correspondences = 128
_C.coarse_matching.num_correspondences_m = 32
_C.coarse_matching.num_correspondences_s = 16
_C.coarse_matching.dual_normalization = True

# model - GeoTransformer
_C.geotransformer = edict()
_C.geotransformer.input_dim = 512
_C.geotransformer.hidden_dim = 256
_C.geotransformer.output_dim = 128
_C.geotransformer.num_heads = 4
_C.geotransformer.blocks = ["self", "cross", "self", "cross", "self", "cross"]
_C.geotransformer.sigma_d = 0.2
_C.geotransformer.sigma_a = 15
_C.geotransformer.angle_k = 3
_C.geotransformer.reduction_a = "max"

# model - Fine Matching
_C.fine_matching = edict()
_C.fine_matching.topk = 3
_C.fine_matching.acceptance_radius = 0.1
_C.fine_matching.mutual = True
_C.fine_matching.confidence_threshold = 0.05
_C.fine_matching.use_dustbin = False
_C.fine_matching.use_global_score = False
_C.fine_matching.correspondence_threshold = 3
_C.fine_matching.correspondence_limit = None
_C.fine_matching.num_refinement_steps = 5

# model - DINO
_C.dino = edict()
_C.dino.arch = 'vit_base'
_C.dino.patch_size = 8
_C.dino.pretrained_weights = ''
_C.dino.checkpoint_key = "teacher"
_C.dino.output_dim = 768
_C.dino.vis = False

# model - Recon
_C.recon = edict()
_C.recon.encoder = 'foldnet'
_C.recon.k = 64
_C.recon.feat_dims = 512
_C.recon.shape = 'plane'
_C.recon.cls_emb = False


# model - DDPM
_C.ddpm = edict()
_C.ddpm.batch_size = 64
_C.ddpm.num_steps = 1000
_C.ddpm.beta_1 = 1e-4
_C.ddpm.beta_T = 0.02
_C.ddpm.sched_mode = 'linear'
_C.ddpm.multi_hypothesis = 16
_C.ddpm.rotation_type = 'ortho6d'
_C.ddpm.time_emb_dim = 256

# model - DDPM - Transformer
_C.ddpm_transformer = edict()
_C.ddpm_transformer.n_layers = 8
_C.ddpm_transformer.n_heads = 4
_C.ddpm_transformer.query_dimensions = 128
_C.ddpm_transformer.value_dimensions = 64
_C.ddpm_transformer.feed_forward_dimensions = 1024
_C.ddpm_transformer.attention_type = "full"
_C.ddpm_transformer.activation = "gelu"

# loss - Coarse level
_C.coarse_loss = edict()
_C.coarse_loss.positive_margin = 0.1
_C.coarse_loss.negative_margin = 1.4
_C.coarse_loss.positive_optimal = 0.1
_C.coarse_loss.negative_optimal = 1.4
_C.coarse_loss.log_scale = 24
_C.coarse_loss.positive_overlap = 0.1

# loss - Fine level
_C.fine_loss = edict()
_C.fine_loss.positive_radius = 0.05

# loss - Overall
_C.loss = edict()
_C.loss.weight_coarse_loss = 1.0
_C.loss.weight_fine_loss = 1.0


def make_cfg():
    return _C


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--link_output", dest="link_output", action="store_true", help="link output dir")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = make_cfg()
    if args.link_output:
        os.symlink(cfg.output_dir, "output")


if __name__ == "__main__":
    main()
