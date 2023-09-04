import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed

from geotransformer.modules.ops import point_to_node_partition, index_select
from geotransformer.modules.registration import get_node_correspondences
from geotransformer.modules.sinkhorn import LearnableLogOptimalTransport
from geotransformer.modules.geotransformer import (
    GeometricTransformer,
    SuperPointMatching,
    SuperPointTargetGenerator,
    LocalGlobalRegistration,
)

from backbone import KPConvFPN
from geotransformer.datasets.registration.linemod.bop_utils import get_corr_score_matrix


class GeoTransformer(nn.Module):
    def __init__(self, cfg, cordi=None):
        super(GeoTransformer, self).__init__()
        self.num_points_in_patch = cfg.model.num_points_in_patch
        self.matching_radius = cfg.model.ground_truth_matching_radius
        self.sel_ref_num = cfg.ddpm.ref_sample_num
        self.sel_src_num = cfg.ddpm.src_sample_num

        self.backbone = KPConvFPN(
            cfg.backbone.input_dim,
            cfg.backbone.output_dim,
            cfg.backbone.init_dim,
            cfg.backbone.kernel_size,
            cfg.backbone.init_radius,
            cfg.backbone.init_sigma,
            cfg.backbone.group_norm,
        )

        self.transformer = GeometricTransformer(
            cfg.geotransformer.input_dim,
            cfg.geotransformer.output_dim,
            cfg.geotransformer.hidden_dim,
            cfg.geotransformer.num_heads,
            cfg.geotransformer.blocks,
            cfg.geotransformer.sigma_d,
            cfg.geotransformer.sigma_a,
            cfg.geotransformer.angle_k,
            reduction_a=cfg.geotransformer.reduction_a,
        )

        self.coarse_target = SuperPointTargetGenerator(
            cfg.coarse_matching.num_targets, cfg.coarse_matching.overlap_threshold
        )

        self.coarse_matching = SuperPointMatching(
            cfg.coarse_matching.num_correspondences, cfg.coarse_matching.dual_normalization
        )
        self.coarse_matching_m = SuperPointMatching(
            cfg.coarse_matching.num_correspondences_m, cfg.coarse_matching.dual_normalization
        )
        self.coarse_matching_s = SuperPointMatching(
            cfg.coarse_matching.num_correspondences_s, cfg.coarse_matching.dual_normalization
        )

        self.fine_matching = LocalGlobalRegistration(
            cfg.fine_matching.topk,
            cfg.fine_matching.acceptance_radius,
            mutual=cfg.fine_matching.mutual,
            confidence_threshold=cfg.fine_matching.confidence_threshold,
            use_dustbin=cfg.fine_matching.use_dustbin,
            use_global_score=cfg.fine_matching.use_global_score,
            correspondence_threshold=cfg.fine_matching.correspondence_threshold,
            correspondence_limit=cfg.fine_matching.correspondence_limit,
            num_refinement_steps=cfg.fine_matching.num_refinement_steps,
        )

        self.optimal_transport = LearnableLogOptimalTransport(cfg.model.num_sinkhorn_iterations)

        self.cordi = cordi
        self.use_cordi = cfg.ddpm.use_ddpm_reference

    def forward(self, data_dict):
        output_dict = {}

        # Downsample point clouds
        feats = data_dict['features'].detach()
        transform = data_dict['transform'].detach()

        ref_length_c = data_dict['lengths'][-1][0].item()
        ref_length_f = data_dict['lengths'][0][0].item()
        ref_length = data_dict['lengths'][0][0].item()
        points_c = data_dict['points'][-1].detach()
        points_f = data_dict['points'][0].detach()
        points = data_dict['points'][0].detach()

        ref_points_c = points_c[:ref_length_c]
        src_points_c = points_c[ref_length_c:]
        ref_points_f = points_f[:ref_length_f]
        src_points_f = points_f[ref_length_f:]
        ref_points = points[:ref_length]
        src_points = points[ref_length:]

        output_dict['ref_points_c'] = ref_points_c
        output_dict['src_points_c'] = src_points_c
        output_dict['ref_points_f'] = ref_points_f
        output_dict['src_points_f'] = src_points_f
        output_dict['ref_points'] = ref_points
        output_dict['src_points'] = src_points

        sel_ref_indices = torch.randperm(ref_points_c.shape[0])[:self.sel_ref_num]
        sel_ref_indices = torch.sort(sel_ref_indices)[0]
        sel_src_indices = torch.randperm(src_points_c.shape[0])[:self.sel_src_num]
        sel_src_indices = torch.sort(sel_src_indices)[0]
        ref_points_sel_c = ref_points_c[sel_ref_indices]
        src_points_sel_c = src_points_c[sel_src_indices]
        # 1. Generate ground truth node correspondences
        _, ref_node_masks, ref_node_knn_indices, ref_node_knn_masks = point_to_node_partition(
            ref_points_f, ref_points_c, self.num_points_in_patch
        )
        _, src_node_masks, src_node_knn_indices, src_node_knn_masks = point_to_node_partition(
            src_points_f, src_points_c, self.num_points_in_patch
        )

        ref_node_masks_sel = ref_node_masks[sel_ref_indices]
        src_node_masks_sel = src_node_masks[sel_src_indices]
        ref_node_knn_indices_sel = ref_node_knn_indices[sel_ref_indices]
        src_node_knn_indices_sel = src_node_knn_indices[sel_src_indices]
        ref_node_knn_masks_sel = ref_node_knn_masks[sel_ref_indices]
        src_node_knn_masks_sel = src_node_knn_masks[sel_src_indices]

        ref_padded_points_f = torch.cat([ref_points_f, torch.zeros_like(ref_points_f[:1])], dim=0)
        src_padded_points_f = torch.cat([src_points_f, torch.zeros_like(src_points_f[:1])], dim=0)
        ref_node_knn_points = index_select(ref_padded_points_f, ref_node_knn_indices_sel, dim=0)
        src_node_knn_points = index_select(src_padded_points_f, src_node_knn_indices_sel, dim=0)

        gt_node_corr_indices, gt_node_corr_overlaps = get_node_correspondences(
            ref_points_sel_c,
            src_points_sel_c,
            ref_node_knn_points,
            src_node_knn_points,
            transform,
            self.matching_radius,
            ref_masks=ref_node_masks_sel,
            src_masks=src_node_masks_sel,
            ref_knn_masks=ref_node_knn_masks_sel,
            src_knn_masks=src_node_knn_masks_sel,
        )
        gt_corr_score_matrix = get_corr_score_matrix(ref_points_sel_c, src_points_sel_c, transform)

        output_dict['gt_node_corr_indices'] = gt_node_corr_indices
        output_dict['gt_node_corr_overlaps'] = gt_node_corr_overlaps
        output_dict['gt_node_corr_score'] = gt_corr_score_matrix

        # 2. KPFCNN Encoder
        feats_list = self.backbone(feats, data_dict)

        feats_c = feats_list[-1]
        feats_f = feats_list[0]

        # 3. Conditional Transformer
        ref_feats_c = feats_c[:ref_length_c]
        src_feats_c = feats_c[ref_length_c:]

        # randomly select N and M points from ref and src points and feats  
        output_dict['ref_points_sel_c'] = ref_points_sel_c
        output_dict['src_points_sel_c'] = src_points_sel_c

        ref_feats_c, src_feats_c, ref_mid_feats, src_mid_feats= self.transformer(
            ref_points_c.unsqueeze(0),
            src_points_c.unsqueeze(0),
            ref_feats_c.unsqueeze(0),
            src_feats_c.unsqueeze(0),
        )
        ref_feats_sel_c = ref_feats_c.squeeze(0)[sel_ref_indices]
        src_feats_sel_c = src_feats_c.squeeze(0)[sel_src_indices]
        ref_feats_sel_c_norm = F.normalize(ref_feats_sel_c, p=2, dim=1)
        src_feats_sel_c_norm = F.normalize(src_feats_sel_c, p=2, dim=1)

        ref_mid_feats_sel = []
        src_mid_feats_sel = []
        for feat in ref_mid_feats:
            feat_sel = feat.squeeze(0)[sel_ref_indices]
            feat_sel_norm = F.normalize(feat_sel, p=2, dim=1).unsqueeze(0)
            ref_mid_feats_sel.append(feat_sel_norm)
        for feat in src_mid_feats:
            feat_sel = feat.squeeze(0)[sel_src_indices]
            feat_sel_norm = F.normalize(feat_sel, p=2, dim=1).unsqueeze(0)
            src_mid_feats_sel.append(feat_sel_norm)
        
        output_dict['ref_mid_feats_sel'] = torch.cat(ref_mid_feats_sel, dim=0)
        output_dict['src_mid_feats_sel'] = torch.cat(src_mid_feats_sel, dim=0)


        output_dict['ref_feats_c'] = ref_feats_sel_c_norm
        output_dict['src_feats_c'] = src_feats_sel_c_norm

        # 5. Head for fine level matching
        ref_feats_f = feats_f[:ref_length_f]
        src_feats_f = feats_f[ref_length_f:]
        output_dict['ref_feats_f'] = ref_feats_f
        output_dict['src_feats_f'] = src_feats_f

        # 6. Select topk nearest node correspondences
        with torch.no_grad():
            ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_matching(
                ref_feats_sel_c_norm, src_feats_sel_c_norm, ref_node_masks_sel, src_node_masks_sel
            )

            output_dict['ref_node_corr_indices'] = ref_node_corr_indices
            output_dict['src_node_corr_indices'] = src_node_corr_indices

            ref_node_corr_indices_m, src_node_corr_indices_m, _ = self.coarse_matching_m(
                ref_feats_sel_c_norm, src_feats_sel_c_norm, ref_node_masks_sel, src_node_masks_sel
            )

            output_dict['ref_node_corr_indices_m'] = ref_node_corr_indices_m
            output_dict['src_node_corr_indices_m'] = src_node_corr_indices_m

            ref_node_corr_indices_s, src_node_corr_indices_s, _ = self.coarse_matching_s(
                ref_feats_sel_c_norm, src_feats_sel_c_norm, ref_node_masks_sel, src_node_masks_sel
            )

            output_dict['ref_node_corr_indices_s'] = ref_node_corr_indices_s
            output_dict['src_node_corr_indices_s'] = src_node_corr_indices_s

            # 7 Random select ground truth node correspondences during training
            if self.training:
                ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_target(
                    gt_node_corr_indices, gt_node_corr_overlaps
                )
        
        with torch.no_grad():
            if self.use_cordi:
                cordi_output_dict = self.cordi.sample(output_dict)
                pred_node_corr_pairs = cordi_output_dict['pred_corr']
                pred_node_corr_scores = cordi_output_dict['pred_corr_mat']
                ref_node_corr_indices = pred_node_corr_pairs[:, 0]
                src_node_corr_indices = pred_node_corr_pairs[:, 1]
                node_corr_scores = pred_node_corr_scores

        # 7.2 Generate batched node points & feats
        ref_node_corr_knn_indices = ref_node_knn_indices_sel[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_indices = src_node_knn_indices_sel[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_masks = ref_node_knn_masks[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_masks = src_node_knn_masks[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_points = ref_node_knn_points[ref_node_corr_indices]  # (P, K, 3)
        src_node_corr_knn_points = src_node_knn_points[src_node_corr_indices]  # (P, K, 3)

        ref_padded_feats_f = torch.cat([ref_feats_f, torch.zeros_like(ref_feats_f[:1])], dim=0)
        src_padded_feats_f = torch.cat([src_feats_f, torch.zeros_like(src_feats_f[:1])], dim=0)
        ref_node_corr_knn_feats = index_select(ref_padded_feats_f, ref_node_corr_knn_indices, dim=0)  # (P, K, C)
        src_node_corr_knn_feats = index_select(src_padded_feats_f, src_node_corr_knn_indices, dim=0)  # (P, K, C)

        output_dict['ref_node_corr_knn_points'] = ref_node_corr_knn_points
        output_dict['src_node_corr_knn_points'] = src_node_corr_knn_points
        output_dict['ref_node_corr_knn_masks'] = ref_node_corr_knn_masks
        output_dict['src_node_corr_knn_masks'] = src_node_corr_knn_masks

        # 8. Optimal transport
        matching_scores = torch.einsum('bnd,bmd->bnm', ref_node_corr_knn_feats, src_node_corr_knn_feats)  # (P, K, K)
        matching_scores = matching_scores / feats_f.shape[1] ** 0.5
        matching_scores = self.optimal_transport(matching_scores, ref_node_corr_knn_masks, src_node_corr_knn_masks)

        output_dict['matching_scores'] = matching_scores

        # 9. Generate final correspondences during testing
        with torch.no_grad():
            if not self.fine_matching.use_dustbin:
                matching_scores = matching_scores[:, :-1, :-1]

            ref_corr_points, src_corr_points, corr_scores, estimated_transform = self.fine_matching(
                ref_node_corr_knn_points,
                src_node_corr_knn_points,
                ref_node_corr_knn_masks,
                src_node_corr_knn_masks,
                matching_scores,
                node_corr_scores,
            )

            output_dict['ref_corr_points'] = ref_corr_points
            output_dict['src_corr_points'] = src_corr_points
            output_dict['corr_scores'] = corr_scores
            output_dict['estimated_transform'] = estimated_transform

        return output_dict


def create_model(cfg, cordi=None):
    model = GeoTransformer(cfg)
    return model


def main():
    from config import make_cfg

    cfg = make_cfg()
    model = create_model(cfg)
    print(model.state_dict().keys())
    print(model)


if __name__ == '__main__':
    main()
