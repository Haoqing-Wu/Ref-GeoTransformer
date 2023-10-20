import torch
import torch.nn as nn

from geotransformer.modules.ops import apply_transform, pairwise_distance
from geotransformer.modules.loss import WeightedCircleLoss
from geotransformer.modules.registration.metrics import isotropic_transform_error
from scipy.spatial.transform import Rotation
from geotransformer.datasets.registration.linemod.bop_utils import *

class CoarseMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super(CoarseMatchingLoss, self).__init__()
        self.weighted_circle_loss = WeightedCircleLoss(
            cfg.coarse_loss.positive_margin,
            cfg.coarse_loss.negative_margin,
            cfg.coarse_loss.positive_optimal,
            cfg.coarse_loss.negative_optimal,
            cfg.coarse_loss.log_scale,
        )
        self.positive_overlap = cfg.coarse_loss.positive_overlap

    def forward(self, output_dict):
        ref_feats = output_dict['ref_feats_c']
        src_feats = output_dict['src_feats_c']
        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]

        feat_dists = torch.sqrt(pairwise_distance(ref_feats, src_feats, normalized=True))

        overlaps = torch.zeros_like(feat_dists)
        overlaps[gt_ref_node_corr_indices, gt_src_node_corr_indices] = gt_node_corr_overlaps
        pos_masks = torch.gt(overlaps, self.positive_overlap)
        neg_masks = torch.eq(overlaps, 0)
        pos_scales = torch.sqrt(overlaps * pos_masks.float())

        loss = self.weighted_circle_loss(pos_masks, neg_masks, feat_dists, pos_scales)

        return loss


class FineMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super(FineMatchingLoss, self).__init__()
        self.positive_radius = cfg.fine_loss.positive_radius

    def forward(self, output_dict, data_dict):
        ref_node_corr_knn_points = output_dict['ref_node_corr_knn_points']
        src_node_corr_knn_points = output_dict['src_node_corr_knn_points']
        ref_node_corr_knn_masks = output_dict['ref_node_corr_knn_masks']
        src_node_corr_knn_masks = output_dict['src_node_corr_knn_masks']
        matching_scores = output_dict['matching_scores']
        transform = data_dict['transform']

        src_node_corr_knn_points = apply_transform(src_node_corr_knn_points, transform)
        dists = pairwise_distance(ref_node_corr_knn_points, src_node_corr_knn_points)  # (B, N, M)
        gt_masks = torch.logical_and(ref_node_corr_knn_masks.unsqueeze(2), src_node_corr_knn_masks.unsqueeze(1))
        gt_corr_map = torch.lt(dists, self.positive_radius ** 2)
        gt_corr_map = torch.logical_and(gt_corr_map, gt_masks)
        slack_row_labels = torch.logical_and(torch.eq(gt_corr_map.sum(2), 0), ref_node_corr_knn_masks)
        slack_col_labels = torch.logical_and(torch.eq(gt_corr_map.sum(1), 0), src_node_corr_knn_masks)

        labels = torch.zeros_like(matching_scores, dtype=torch.bool)
        labels[:, :-1, :-1] = gt_corr_map
        labels[:, :-1, -1] = slack_row_labels
        labels[:, -1, :-1] = slack_col_labels

        loss = -matching_scores[labels].mean()

        return loss


class OverallLoss(nn.Module):
    def __init__(self, cfg):
        super(OverallLoss, self).__init__()
        self.coarse_loss = CoarseMatchingLoss(cfg)
        self.fine_loss = FineMatchingLoss(cfg)
        self.weight_coarse_loss = cfg.loss.weight_coarse_loss
        self.weight_fine_loss = cfg.loss.weight_fine_loss

    def forward(self, output_dict, data_dict):
        coarse_loss = self.coarse_loss(output_dict)
        fine_loss = self.fine_loss(output_dict, data_dict)

        loss = self.weight_coarse_loss * coarse_loss + self.weight_fine_loss * fine_loss

        return {
            'loss': loss,
            'c_loss': coarse_loss,
            'f_loss': fine_loss,
        }


class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = x.pow(2).sum(dim=-1)
        yy = y.pow(2).sum(dim=-1)
        zz = torch.bmm(x, y.transpose(2, 1))
        rx = xx.unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy.unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P

    def loss(self, preds, gts):
        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)
        loss_1 = torch.sum(mins)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.sum(mins)
        return loss_1 + loss_2
    
    def forward(self, output_dict, data_dict):
        recon = output_dict['recon']
        src_points = data_dict['src_points']
        transform = data_dict['transform']
        gt_src_points = apply_transform(src_points, transform)
        loss = self.loss(recon, gt_src_points)

        return {
            'loss': loss
        }


    
class Evaluator(nn.Module):
    def __init__(self, cfg):
        super(Evaluator, self).__init__()
        self.acceptance_overlap = cfg.eval.acceptance_overlap
        self.acceptance_radius = cfg.eval.acceptance_radius
        self.acceptance_rre = cfg.eval.rre_threshold
        self.acceptance_rte = cfg.eval.rte_threshold

    @torch.no_grad()
    def evaluate_coarse(self, output_dict):
        ref_length_c = output_dict['ref_points_c'].shape[0]
        src_length_c = output_dict['src_points_c'].shape[0]
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']
        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        masks = torch.gt(gt_node_corr_overlaps, self.acceptance_overlap)
        gt_node_corr_indices = gt_node_corr_indices[masks]
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]
        gt_node_corr_map = torch.zeros(ref_length_c, src_length_c).cuda()
        gt_node_corr_map[gt_ref_node_corr_indices, gt_src_node_corr_indices] = 1.0

        ref_node_corr_indices = output_dict['ref_node_corr_indices']
        src_node_corr_indices = output_dict['src_node_corr_indices']

        precision = gt_node_corr_map[ref_node_corr_indices, src_node_corr_indices].mean()

        return precision

    @torch.no_grad()
    def evaluate_fine(self, output_dict, data_dict):
        transform = data_dict['transform']
        ref_corr_points = output_dict['ref_corr_points']
        src_corr_points = output_dict['src_corr_points']
        src_corr_points = apply_transform(src_corr_points, transform)
        corr_distances = torch.linalg.norm(ref_corr_points - src_corr_points, dim=1)
        precision = torch.lt(corr_distances, self.acceptance_radius).float().mean()
        return precision

    @torch.no_grad()
    def evaluate_registration(self, output_dict, data_dict):
        transform = data_dict['transform']
        est_transform = output_dict['estimated_transform']
        src_points = output_dict['src_points']

        rre, rte = isotropic_transform_error(transform, est_transform)
        recall = torch.logical_and(torch.lt(rre, self.acceptance_rre), torch.lt(rte, self.acceptance_rte)).float()

        gt_src_points = apply_transform(src_points, transform)
        est_src_points = apply_transform(src_points, est_transform)
        rmse = torch.linalg.norm(est_src_points - gt_src_points, dim=1).mean()

        return rre, rte, rmse, recall

    def forward(self, output_dict, data_dict):
        c_precision = self.evaluate_coarse(output_dict)
        f_precision = self.evaluate_fine(output_dict, data_dict)
        rre, rte, rmse, recall = self.evaluate_registration(output_dict, data_dict)

        return {
            'PIR': c_precision,
            'IR': f_precision,
            'RRE': rre,
            'RTE': rte,
            'RMSE': rmse,
            'RR': recall,
        }

class DDPMEvaluator(nn.Module):
    def __init__(self, cfg):
        super(DDPMEvaluator, self).__init__()
        self.acceptance_overlap = cfg.eval.acceptance_overlap
        self.acceptance_radius = cfg.eval.acceptance_radius
        self.acceptance_rre = cfg.eval.rre_threshold
        self.acceptance_rte = cfg.eval.rte_threshold

    @torch.no_grad()
    def evaluate_coarse_geotransformer(self, output_dict):
        ref_length_c = output_dict['ref_points_c'].shape[0]
        src_length_c = output_dict['src_points_c'].shape[0]
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']
        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        masks = torch.gt(gt_node_corr_overlaps, self.acceptance_overlap)
        gt_node_corr_indices = gt_node_corr_indices[masks]
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]
        gt_node_corr_map = torch.zeros(ref_length_c, src_length_c).cuda()
        gt_node_corr_map[gt_ref_node_corr_indices, gt_src_node_corr_indices] = 1.0

        ref_node_corr_indices = output_dict['ref_node_corr_indices']
        src_node_corr_indices = output_dict['src_node_corr_indices']

        precision = gt_node_corr_map[ref_node_corr_indices, src_node_corr_indices].mean()

        ref_node_corr_indices_m = output_dict['ref_node_corr_indices_m']
        src_node_corr_indices_m = output_dict['src_node_corr_indices_m']

        precision_m = gt_node_corr_map[ref_node_corr_indices_m, src_node_corr_indices_m].mean()

        ref_node_corr_indices_s = output_dict['ref_node_corr_indices_s']
        src_node_corr_indices_s = output_dict['src_node_corr_indices_s']

        precision_s = gt_node_corr_map[ref_node_corr_indices_s, src_node_corr_indices_s].mean()

        return precision, precision_m, precision_s

    @torch.no_grad()
    def evaluate_coarse(self, output_dict):   
        gt_corr_matrix = (output_dict['gt_corr_matrix'] + 1) / 2
        pred_corr = output_dict['pred_corr']
        if len(pred_corr) < 1:
            precision = 0.0
        else:
            pred_ref_corr_indices = pred_corr[:, 0]
            pred_src_corr_indices = pred_corr[:, 1]
            precision = gt_corr_matrix[pred_ref_corr_indices, pred_src_corr_indices].mean()

        pred_corr_1_2 = output_dict['pred_corr_1_2']
        if len(pred_corr_1_2) < 1:
            precision_1_2 = 0.0
        else:
            pred_ref_corr_indices_1_2 = pred_corr_1_2[:, 0]
            pred_src_corr_indices_1_2 = pred_corr_1_2[:, 1]
            precision_1_2 = gt_corr_matrix[pred_ref_corr_indices_1_2, pred_src_corr_indices_1_2].mean()

        pred_corr_1_4 = output_dict['pred_corr_1_4']
        if len(pred_corr_1_4) < 1:
            precision_1_4 = 0.0
        else:
            pred_ref_corr_indices_1_4 = pred_corr_1_4[:, 0]
            pred_src_corr_indices_1_4 = pred_corr_1_4[:, 1]
            precision_1_4 = gt_corr_matrix[pred_ref_corr_indices_1_4, pred_src_corr_indices_1_4].mean()
        
        pred_corr_0_9 = output_dict['pred_corr_0_9']
        if len(pred_corr_0_9) < 1:
            precision_0_9 = 0.0
        else:
            pred_ref_corr_indices_0_9 = pred_corr_0_9[:, 0]
            pred_src_corr_indices_0_9 = pred_corr_0_9[:, 1]
            precision_0_9 = gt_corr_matrix[pred_ref_corr_indices_0_9, pred_src_corr_indices_0_9].mean()
        
        pred_corr_0_95 = output_dict['pred_corr_0_95']
        if len(pred_corr_0_95) < 1:
            precision_0_95 = 0.0
        else:
            pred_ref_corr_indices_0_95 = pred_corr_0_95[:, 0]
            pred_src_corr_indices_0_95 = pred_corr_0_95[:, 1]
            precision_0_95 = gt_corr_matrix[pred_ref_corr_indices_0_95, pred_src_corr_indices_0_95].mean()

        pred_corr_1 = output_dict['pred_corr_1']
        if len(pred_corr_1) < 1:
            precision_1 = 0.0
        else:
            pred_ref_corr_indices_1 = pred_corr_1[:, 0]
            pred_src_corr_indices_1 = pred_corr_1[:, 1]
            precision_1 = gt_corr_matrix[pred_ref_corr_indices_1, pred_src_corr_indices_1].mean()

        num_corr_0_9 = output_dict['num_corr_0_9']
        num_corr_0_95 = output_dict['num_corr_0_95']
        num_corr_1 = output_dict['num_corr_1']

        init_corr_matrix = (output_dict['init_corr_matrix'] + 1) / 2
        init_ref_corr_indices = []
        init_src_corr_indices = []
        # get the indices of reference and source points that have correspondences from init_corr_matrix
        for i in range(init_corr_matrix.shape[0]):
            for j in range(init_corr_matrix.shape[1]):
                if init_corr_matrix[i, j] == 1:
                    init_ref_corr_indices.append(i)
                    init_src_corr_indices.append(j)
                   
            # compute the mean of the correspondences
        init_precision = gt_corr_matrix[init_ref_corr_indices, init_src_corr_indices].mean()

        corr_num = output_dict['init_corr_num']
        return precision, precision_1_2, precision_1_4, precision_0_9, precision_0_95, precision_1, \
                num_corr_0_9, num_corr_0_95, num_corr_1, init_precision, corr_num

    @torch.no_grad()
    def evaluate_fine(self, output_dict, data_dict):
        transform = data_dict['transform']
        ref_corr_points = output_dict['ref_corr_points']
        src_corr_points = output_dict['src_corr_points']
        src_corr_points = apply_transform(src_corr_points, transform)
        corr_distances = torch.linalg.norm(ref_corr_points - src_corr_points, dim=1)
        precision = torch.lt(corr_distances, self.acceptance_radius).float().mean()
        return precision

    @torch.no_grad()
    def evaluate_registration(self, output_dict, data_dict, level):
        transform = data_dict['transform_raw'].squeeze(0)
        if level == 'coarse':
            est_transform = output_dict['coarse_trans']
        elif level == 'refined':
            est_transform = output_dict['refined_trans']
            
        src_points = output_dict['src_points']

        rre, rte = isotropic_transform_error(transform, est_transform)
        recall = torch.logical_and(torch.lt(rre, self.acceptance_rre), torch.lt(rte, self.acceptance_rte)).float()

        gt_src_points = apply_transform(src_points, transform)
        est_src_points = apply_transform(src_points, est_transform)
        rmse = torch.linalg.norm(est_src_points - gt_src_points, dim=1).mean()

        return rre, rte, rmse, recall

    def forward(self, output_dict, data_dict):
        rre_c, rte_c, rmse_c, recall_c = self.evaluate_registration(output_dict, data_dict, 'coarse')
        rre_r, rte_r, rmse_r, recall_r = self.evaluate_registration(output_dict, data_dict, 'refined')
        return {
            'RRE_C': rre_c,
            'RTE_C': rte_c,
            'RMSE_C': rmse_c,
            'RR_C': recall_c,
            'RRE_R': rre_r,
            'RTE_R': rte_r,
            'RMSE_R': rmse_r,
            'RR_R': recall_r,
            'Var': output_dict['var_rt'],
        }