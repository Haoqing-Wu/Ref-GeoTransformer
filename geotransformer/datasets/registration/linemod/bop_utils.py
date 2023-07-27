import json
import cv2
import torch
import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
from typing import Optional
#from focal_loss.focal_loss import FocalLoss



def sample_point_from_mesh(model_root,samples):
    r"""Sample given number of points from a mesh readed from path.
    """
    mesh = o3d.io.read_triangle_mesh(model_root)
    pcd = mesh.sample_points_uniformly(number_of_points=samples)
    scale_factor = 0.001
    pcd.scale(scale_factor,(0, 0, 0))
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    return points, normals

def get_bbox(bbox):
    r"""Get bounding box from a mask.
    Return coordinates of the bounding box [x_min, y_min, x_max, y_max]
    """
    border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]

    rmin, rmax, cmin, cmax = bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]
    rmin = max(rmin, 0)
    rmax = min(rmax, 479)
    cmin = max(cmin, 0)
    cmax = min(cmax, 639)
    r_b = rmax - rmin
    c_b = cmax - cmin

    for i in range(len(border_list) - 1):
        if r_b > border_list[i] and r_b < border_list[i + 1]:
            r_b = border_list[i + 1]
            break
    for i in range(len(border_list) - 1):
        if c_b > border_list[i] and c_b < border_list[i + 1]:
            c_b = border_list[i + 1]
            break

    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    rmin = max(rmin, 0)
    cmin = max(cmin, 0)
    rmax = min(rmax, 480)
    cmax = min(cmax, 640)

    return rmin, rmax, cmin, cmax


def mask_to_bbox(mask):
    r"""Get bounding box from a mask.
    """
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bbox = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return list(bbox)

def get_gt(gt_file, frame_id):
    r"""Get ground truth pose from a ground truth file.
    Return rotation matrix and translation vector
    """
    with open(gt_file, 'r') as file:
        gt = json.load(file)[str(frame_id)][0]
    rot = np.array(gt['cam_R_m2c']).reshape(3, 3)
    trans = np.array(gt['cam_t_m2c']) / 1000
    return rot, trans

def get_camera_info(cam_file, frame_id):
    r"""Get camera intrinsics from a camera file.
    Return camera center, focal length
    """
    with open(cam_file, 'r') as file:
        cam = json.load(file)[str(frame_id)]
    cam_k = np.array(cam['cam_K']).reshape(3, 3)
    cam_cx = cam_k[0, 2]
    cam_cy = cam_k[1, 2]
    cam_fx = cam_k[0, 0]
    cam_fy = cam_k[1, 1]
    return cam_cx, cam_cy, cam_fx, cam_fy

def resize_pcd(pcd, points_limit):
    r"""Resize a point cloud to a given number of points.
    """
    if pcd.shape[0] > points_limit:
        idx = np.random.permutation(pcd.shape[0])[:points_limit]
        pcd = pcd[idx]
    return pcd

def sort_pcd_from_center(pcd):
    r"""Sort a point cloud from the center.
    """
    center = np.mean(pcd, axis=0)
    pcd_v = pcd - center
    dist = np.sqrt(np.sum(np.square(pcd_v), axis=1))
    idx = np.argsort(dist)
    pcd = pcd[idx]
    return pcd

def transformation_pcd(pcd, rot, trans):
    r"""Transform a point cloud with a rotation matrix and a translation vector.
    """
    pcd_t = np.dot(pcd, rot.T)
    pcd_t = np.add(pcd_t, trans.T)
    return pcd_t

def get_transform_from_rotation_translation(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    r"""Get rigid transform matrix from rotation matrix and translation vector.

    Args:
        rotation (array): (3, 3)
        translation (array): (3,)

    Returns:
        transform: (4, 4)
    """
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform

def apply_transform(points: torch.Tensor, transform: torch.Tensor, normals: Optional[torch.Tensor] = None):
    r"""Rigid transform to points and normals (optional).

    Given a point cloud P(3, N), normals V(3, N) and a transform matrix T in the form of
      | R t |
      | 0 1 |,
    the output point cloud Q = RP + t, V' = RV.

    In the implementation, P and V are (N, 3), so R should be transposed: Q = PR^T + t, V' = VR^T.

    There are two cases supported:
    1. points and normals are (*, 3), transform is (4, 4), the output points are (*, 3).
       In this case, the transform is applied to all points.
    2. points and normals are (B, N, 3), transform is (B, 4, 4), the output points are (B, N, 3).
       In this case, the transform is applied batch-wise. The points can be broadcast if B=1.

    Args:
        points (Tensor): (*, 3) or (B, N, 3)
        normals (optional[Tensor]=None): same shape as points.
        transform (Tensor): (4, 4) or (B, 4, 4)

    Returns:
        points (Tensor): same shape as points.
        normals (Tensor): same shape as points.
    """
    if normals is not None:
        assert points.shape == normals.shape
    if transform.ndim == 2:
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        points_shape = points.shape
        points = points.reshape(-1, 3)
        points = torch.matmul(points, rotation.transpose(-1, -2)) + translation
        points = points.reshape(*points_shape)
        if normals is not None:
            normals = normals.reshape(-1, 3)
            normals = torch.matmul(normals, rotation.transpose(-1, -2))
            normals = normals.reshape(*points_shape)
    elif transform.ndim == 3 and points.ndim == 3:
        rotation = transform[:, :3, :3]  # (B, 3, 3)
        translation = transform[:, None, :3, 3]  # (B, 1, 3)
        points = torch.matmul(points, rotation.transpose(-1, -2)) + translation
        if normals is not None:
            normals = torch.matmul(normals, rotation.transpose(-1, -2))
    else:
        raise ValueError(
            'Incompatible shapes between points {} and transform {}.'.format(
                tuple(points.shape), tuple(transform.shape)
            )
        )
    if normals is not None:
        return points, normals
    else:
        return points

def get_corr(tgt_pcd, src_pcd, rot, trans, radius):
    r"""Find the ground truth correspondences within the matching radius between two point clouds.
    Return correspondence indices [indices in ref_points, indices in src_points]
    """
    src_t = transformation_pcd(src_pcd, rot, trans)
    src_tree = cKDTree(src_t)
    indices_list = src_tree.query_ball_point(tgt_pcd, radius)
    corr = np.array(
        [(i, j) for i, indices in enumerate(indices_list) for j in indices],
        dtype=np.int32,
    )
    coverage = corr.shape[0] / tgt_pcd.shape[0]
    return corr, coverage

def get_corr_score_matrix(tgt_pcd, src_pcd, transform, sigma=1.0):
    src_pcd_t = apply_transform(src_pcd, transform)
    scores = []
    for point1 in src_pcd_t:
        point_scores = []
        for point2 in tgt_pcd:
            distance = torch.sqrt(torch.sum((point1 - point2) ** 2))
            score = torch.exp(-(distance ** 2) / (2 * sigma ** 2))
            point_scores.append(score * 2 - 1)
        scores.append(point_scores)
    return scores

def get_corr_matrix(corr, tgt_len, src_len):
    r"""Get a correspondence matrix from a correspondence array.
    Return correspondence matrix [tgt_len, src_len]
    """
    corr_matrix = np.full((tgt_len, src_len), -1.0, dtype=np.float32)
    corr_matrix[corr[:, 0], corr[:, 1]] = 1.0
    return corr_matrix

def get_corr_src_pcd(corr, src_pcd):
    r"""Get the source point cloud of the correspondences.
    Return source point cloud of the correspondences
    """
    return src_pcd[corr[:, 1]]

def get_corr_similarity(corr_matrix, gt_corr_matrix):
    r"""Get the cosine distance similarity between the correspondence matrix 
    and the ground truth correspondence matrix.
    Return cosine distance similarity
    """
    corr_matrix = corr_matrix.astype(np.float32)
    gt_corr_matrix = gt_corr_matrix.astype(np.float32)
    num = np.dot(corr_matrix, gt_corr_matrix.T)  
    denom = np.linalg.norm(corr_matrix, axis=1).reshape(-1, 1) * np.linalg.norm(gt_corr_matrix, axis=1) 
    res = num / denom
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res

def focal_loss(y_true, y_pred, gamma=2):
    """
    Focal loss function for binary classification
    """
    criterion = FocalLoss(gamma)
    y_pred = torch.from_numpy(y_pred.flatten())
    y_pred = torch.nn.Sigmoid()(y_pred)
    y_true = (y_true + 1) / 2
    y_true = torch.from_numpy(y_true.flatten().astype(np.int64))
    focal_loss = criterion(y_pred, y_true)
    return focal_loss.numpy()

def normalize_points(src, tgt, rot, trans):
    r"""Normalize point cloud to a unit sphere at origin."""

    src_factor = np.max(np.linalg.norm(src, axis=1))
    src = src / src_factor
    
    inv_rot = rot.T
    inv_trans = -np.matmul(inv_rot, trans)
    tgt = transformation_pcd(tgt, inv_rot, inv_trans)
    tgt = tgt / src_factor
    tgt = transformation_pcd(tgt, rot, trans)

    return src, tgt

def get_corr_from_matrix_topk(corr_matrix, k):
    r"""Get the top k correspondences from a correspondence matrix.[batch_size, tgt_len, src_len]
    Return correspondence indices [indices in ref_points, indices in src_points]
    """
    corr_indices = corr_matrix.view(-1).topk(k=k, largest=True)[1]
    ref_corr_indices = corr_indices // corr_matrix.shape[2]
    src_corr_indices = corr_indices % corr_matrix.shape[2]
    corr = np.array(
        [(i, j) for i, j in zip(ref_corr_indices, src_corr_indices)],
        dtype=np.int32,
    )
    return corr

def get_corr_from_matrix_gt(corr_matrix, low, high):
    r"""Get the between threshold correspondences from a correspondence matrix.[batch_size, tgt_len, src_len]
    Return correspondence indices [indices in ref_points, indices in src_points]
    """
    
    corr_indices = np.where((corr_matrix >= low))
    ref_corr_indices = corr_indices[1]
    src_corr_indices = corr_indices[2]
    corr = np.array(
        [(i, j) for i, j in zip(ref_corr_indices, src_corr_indices)],
        dtype=np.int32,
    )
    return corr, len(ref_corr_indices)

def gt_visualisation(src_pcd, tgt_pcd, trans, rot, corr):
    r"""Visualise the ground truth correspondences between two point clouds.
    shift the transformed source point cloud to avoid overlapping
    """
    shift_t = trans + 0.1
    src_t = transformation_pcd(src_pcd, rot, shift_t)
    pcd_model = o3d.geometry.PointCloud()
    pcd_model.points = o3d.utility.Vector3dVector(src_pcd)
    pcd_model_t = o3d.geometry.PointCloud()
    pcd_model_t.points = o3d.utility.Vector3dVector(src_t)
    pcd_frame = o3d.geometry.PointCloud()
    pcd_frame.points = o3d.utility.Vector3dVector(tgt_pcd)

    

    # draw the correspondences between the transformed source and target
    points = []
    lines = []
    for i in range(corr.shape[0]):
        src_t_point = src_t[corr[i, 1]]
        tgt_point = tgt_pcd[corr[i, 0]]
        points.append(src_t_point)
        points.append(tgt_point)
        lines.append([i * 2, i * 2 + 1])

    line_gt = o3d.geometry.LineSet()
    line_gt.points = o3d.utility.Vector3dVector(points)
    line_gt.lines = o3d.utility.Vector2iVector(lines)
    line_gt.colors = o3d.utility.Vector3dVector([[0, 1, 0] for i in range(len(lines))])

    # draw the correspondences between the original source and target
    points = []
    lines = []
    for i in range(corr.shape[0]):
        src_point = src_pcd[corr[i, 1]]
        tgt_point = tgt_pcd[corr[i, 0]]
        points.append(src_point)
        points.append(tgt_point)
        lines.append([i * 2, i * 2 + 1])

    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector(points)
    line.lines = o3d.utility.Vector2iVector(lines)
    line.colors = o3d.utility.Vector3dVector([[0, 0.8, 0.2] for i in range(len(lines))])
    o3d.visualization.draw_geometries([pcd_model, pcd_model_t, pcd_frame, line, line_gt])

def corr_visualisation(src_pcd, tgt_pcd, corr_mat_pred, corr_mat_gt, rot_gt, trans_gt, shift=0.1):
    # src point cloud
    pcd_model = o3d.geometry.PointCloud()
    pcd_model.points = o3d.utility.Vector3dVector(src_pcd)
    # transformed src point cloud from the ground truth
    shift_t = trans_gt + shift
    src_pcd_t = transformation_pcd(src_pcd, rot_gt, shift_t)
    pcd_model_t = o3d.geometry.PointCloud()
    pcd_model_t.points = o3d.utility.Vector3dVector(src_pcd_t)
    # target point cloud
    pcd_frame = o3d.geometry.PointCloud()
    pcd_frame.points = o3d.utility.Vector3dVector(tgt_pcd)

    pred_points = []
    pred_lines = []
    inlier_points = []
    outlier_points = []
    inlier_lines = []
    outlier_lines = []

    # find inliers pairs from two correspondence matrices
    for i in range(corr_mat_pred.shape[0]):
        for j in range(corr_mat_pred.shape[1]):
            if corr_mat_pred[i, j] == 1.0:
                pred_points.append(src_pcd_t[j])
                pred_points.append(tgt_pcd[i])
                pred_lines.append([len(pred_points) - 2, len(pred_points) - 1])
            if corr_mat_pred[i, j] == 1.0 and corr_mat_gt[i, j] == 1.0:
                inlier_points.append(src_pcd_t[j])
                inlier_points.append(tgt_pcd[i])
                inlier_lines.append([len(inlier_points) - 2, len(inlier_points) - 1])
            elif (corr_mat_pred[i, j] == 1.0 and corr_mat_gt[i, j] == -1.0):
                outlier_points.append(src_pcd_t[j])
                outlier_points.append(tgt_pcd[i])
                outlier_lines.append([len(outlier_points) - 2, len(outlier_points) - 1])
    
    # draw the predicted correspondences
    line_pred = o3d.geometry.LineSet()
    line_pred.points = o3d.utility.Vector3dVector(pred_points)
    line_pred.lines = o3d.utility.Vector2iVector(pred_lines)
    line_pred.colors = o3d.utility.Vector3dVector([[0, 0.8, 0.2] for i in range(len(pred_lines))])
    # draw the inlier correspondences
    line_inlier = o3d.geometry.LineSet()
    line_inlier.points = o3d.utility.Vector3dVector(inlier_points)
    line_inlier.lines = o3d.utility.Vector2iVector(inlier_lines)
    line_inlier.colors = o3d.utility.Vector3dVector([[0, 1, 0] for i in range(len(inlier_lines))])

    # draw the outlier correspondences
    line_outlier = o3d.geometry.LineSet()
    line_outlier.points = o3d.utility.Vector3dVector(outlier_points)
    line_outlier.lines = o3d.utility.Vector2iVector(outlier_lines)
    line_outlier.colors = o3d.utility.Vector3dVector([[1, 0, 0] for i in range(len(outlier_lines))])

    # draw point clouds and correspondences
    #o3d.visualization.draw_geometries([pcd_model_t, pcd_frame, line_inlier, line_outlier])
    # save all visualisation

    o3d.io.write_point_cloud("./output/pcd_model_t.ply", pcd_model_t)
    o3d.io.write_point_cloud("./output/pcd_frame.ply", pcd_frame)
    o3d.io.write_line_set("./output/line_pred.ply", line_pred)
    o3d.io.write_line_set("./output/line_inlier.ply", line_inlier)
    o3d.io.write_line_set("./output/line_outlier.ply", line_outlier)

    # return the inlier ratio
    return len(inlier_lines) / len(pred_lines)
    
def save_corr_pcd(output_dict):
    tgt_pcd = output_dict['ref_points_c'].cpu().numpy()
    src_pcd = output_dict['src_points_c'].cpu().numpy()
    tgt_corr_indices = output_dict['ref_node_corr_indices'].cpu().numpy()
    src_corr_indices = output_dict['src_node_corr_indices'].cpu().numpy()
    gt_corr = output_dict['gt_node_corr_indices'].cpu().numpy()
    # save the target point cloud
    pcd_frame = o3d.geometry.PointCloud()
    pcd_frame.points = o3d.utility.Vector3dVector(tgt_pcd)
    o3d.io.write_point_cloud("./output/geotransformer.modelnet.rpmnet.stage4.gse.k3.max.oacl.stage2.sinkhorn/result/pcd_frame.ply", pcd_frame)
    # save the source point cloud
    pcd_model = o3d.geometry.PointCloud()
    pcd_model.points = o3d.utility.Vector3dVector(src_pcd)
    o3d.io.write_point_cloud("./output/geotransformer.modelnet.rpmnet.stage4.gse.k3.max.oacl.stage2.sinkhorn/result/pcd_model.ply", pcd_model)
    # save the ground truth correspondences
    points = []
    lines = []
    for i in range(gt_corr.shape[0]):
        src_point = src_pcd[gt_corr[i, 1]]
        tgt_point = tgt_pcd[gt_corr[i, 0]]
        points.append(src_point)
        points.append(tgt_point)
        lines.append([i * 2, i * 2 + 1])
    line_gt = o3d.geometry.LineSet()
    line_gt.points = o3d.utility.Vector3dVector(points)
    line_gt.lines = o3d.utility.Vector2iVector(lines)
    line_gt.colors = o3d.utility.Vector3dVector([[0, 1, 0] for i in range(len(lines))])
    o3d.io.write_line_set("./output/geotransformer.modelnet.rpmnet.stage4.gse.k3.max.oacl.stage2.sinkhorn/result/line_gt.ply", line_gt)
    # save the predicted correspondences
    points = []
    lines = []
    for i in range(len(tgt_corr_indices)):
        src_point = src_pcd[src_corr_indices[i]]
        tgt_point = tgt_pcd[tgt_corr_indices[i]]
        points.append(src_point)
        points.append(tgt_point)
        lines.append([i * 2, i * 2 + 1])
    line_pred = o3d.geometry.LineSet()
    line_pred.points = o3d.utility.Vector3dVector(points)
    line_pred.lines = o3d.utility.Vector2iVector(lines)
    line_pred.colors = o3d.utility.Vector3dVector([[0, 0.8, 0.2] for i in range(len(lines))])
    o3d.io.write_line_set("./output/geotransformer.modelnet.rpmnet.stage4.gse.k3.max.oacl.stage2.sinkhorn/result/line_pred.ply", line_pred)
    # find the inlier correspondences
    pred_corr_pairs = []
    for i in range(len(tgt_corr_indices)):
        src_point = src_pcd[src_corr_indices[i]]
        tgt_point = tgt_pcd[tgt_corr_indices[i]]
        pred_corr_pairs.append([src_corr_indices[i], tgt_corr_indices[i]])
    gt_corr_pairs = []
    for i in range(gt_corr.shape[0]):
        gt_corr_pairs.append([gt_corr[i, 1], gt_corr[i, 0]])
    inlier_corr_pairs = []
    outlier_corr_pairs = []
    for i in range(len(pred_corr_pairs)):
        if pred_corr_pairs[i] in gt_corr_pairs:
            inlier_corr_pairs.append(pred_corr_pairs[i])
        else:
            outlier_corr_pairs.append(pred_corr_pairs[i])
    # save the inlier correspondences
    points = []
    lines = []
    for i in range(len(inlier_corr_pairs)):
        src_point = src_pcd[inlier_corr_pairs[i][0]]
        tgt_point = tgt_pcd[inlier_corr_pairs[i][1]]
        points.append(src_point)
        points.append(tgt_point)
        lines.append([i * 2, i * 2 + 1])
    line_inlier = o3d.geometry.LineSet()
    line_inlier.points = o3d.utility.Vector3dVector(points)
    line_inlier.lines = o3d.utility.Vector2iVector(lines)
    line_inlier.colors = o3d.utility.Vector3dVector([[0, 0, 1] for i in range(len(lines))])
    o3d.io.write_line_set("./output/geotransformer.modelnet.rpmnet.stage4.gse.k3.max.oacl.stage2.sinkhorn/result/line_inlier.ply", line_inlier)
    # save the outlier correspondences
    points = []
    lines = []
    for i in range(len(outlier_corr_pairs)):
        src_point = src_pcd[outlier_corr_pairs[i][0]]
        tgt_point = tgt_pcd[outlier_corr_pairs[i][1]]
        points.append(src_point)
        points.append(tgt_point)
        lines.append([i * 2, i * 2 + 1])
    line_outlier = o3d.geometry.LineSet()
    line_outlier.points = o3d.utility.Vector3dVector(points)
    line_outlier.lines = o3d.utility.Vector2iVector(lines)
    line_outlier.colors = o3d.utility.Vector3dVector([[1, 0, 0] for i in range(len(lines))])
    o3d.io.write_line_set("./output/geotransformer.modelnet.rpmnet.stage4.gse.k3.max.oacl.stage2.sinkhorn/result/line_outlier.ply", line_outlier)

def save_corr_pcd_ddpm(output_dict):
    tgt_pcd = output_dict['ref_points'].cpu().numpy()
    src_pcd = output_dict['src_points'].cpu().numpy()
    pred_corr = output_dict['pred_corr']
    tgt_corr_indices = pred_corr[:, 0]
    src_corr_indices = pred_corr[:, 1]

    gt_corr_matrix = output_dict['gt_corr_matrix'].numpy()
    
    # save the target point cloud
    pcd_frame = o3d.geometry.PointCloud()
    pcd_frame.points = o3d.utility.Vector3dVector(tgt_pcd)
    o3d.io.write_point_cloud("./output/geotransformer.modelnet.rpmnet.stage4.gse.k3.max.oacl.stage2.sinkhorn/result/pcd_frame.ply", pcd_frame)
    # save the source point cloud
    pcd_model = o3d.geometry.PointCloud()
    pcd_model.points = o3d.utility.Vector3dVector(src_pcd)
    o3d.io.write_point_cloud("./output/geotransformer.modelnet.rpmnet.stage4.gse.k3.max.oacl.stage2.sinkhorn/result/pcd_model.ply", pcd_model)
    # save the ground truth correspondences
    points = []
    lines = []
    gt_corr_pairs = []
    for i in range(gt_corr_matrix.shape[0]):
        for j in range(gt_corr_matrix.shape[1]):
            if gt_corr_matrix[i, j] == 1.0:
                gt_corr_pairs.append([i, j])
                points.append(src_pcd[j])
                points.append(tgt_pcd[i])
                lines.append([len(points) - 2, len(points) - 1])
    line_gt = o3d.geometry.LineSet()
    line_gt.points = o3d.utility.Vector3dVector(points)
    line_gt.lines = o3d.utility.Vector2iVector(lines)
    line_gt.colors = o3d.utility.Vector3dVector([[0, 1, 0] for i in range(len(lines))])
    o3d.io.write_line_set("./output/geotransformer.modelnet.rpmnet.stage4.gse.k3.max.oacl.stage2.sinkhorn/result/line_gt.ply", line_gt)
    # save the predicted correspondences
    points = []
    lines = []
    for i in range(len(tgt_corr_indices)):
        src_point = src_pcd[src_corr_indices[i]]
        tgt_point = tgt_pcd[tgt_corr_indices[i]]
        points.append(src_point)
        points.append(tgt_point)
        lines.append([i * 2, i * 2 + 1])
    line_pred = o3d.geometry.LineSet()
    line_pred.points = o3d.utility.Vector3dVector(points)
    line_pred.lines = o3d.utility.Vector2iVector(lines)
    line_pred.colors = o3d.utility.Vector3dVector([[0, 0.8, 0.2] for i in range(len(lines))])
    o3d.io.write_line_set("./output/geotransformer.modelnet.rpmnet.stage4.gse.k3.max.oacl.stage2.sinkhorn/result/line_pred.ply", line_pred)
    # find the inlier correspondences
    pred_corr_pairs = pred_corr.tolist()

    inlier_corr_pairs = []
    outlier_corr_pairs = []
    for i in range(len(pred_corr_pairs)):
        if pred_corr_pairs[i] in gt_corr_pairs:
            inlier_corr_pairs.append(pred_corr_pairs[i])
        else:
            outlier_corr_pairs.append(pred_corr_pairs[i])
    # save the inlier correspondences
    points = []
    lines = []
    for i in range(len(inlier_corr_pairs)):
        src_point = src_pcd[inlier_corr_pairs[i][1]]
        tgt_point = tgt_pcd[inlier_corr_pairs[i][0]]
        points.append(src_point)
        points.append(tgt_point)
        lines.append([i * 2, i * 2 + 1])
    line_inlier = o3d.geometry.LineSet()
    line_inlier.points = o3d.utility.Vector3dVector(points)
    line_inlier.lines = o3d.utility.Vector2iVector(lines)
    line_inlier.colors = o3d.utility.Vector3dVector([[0, 0, 1] for i in range(len(lines))])
    o3d.io.write_line_set("./output/geotransformer.modelnet.rpmnet.stage4.gse.k3.max.oacl.stage2.sinkhorn/result/line_inlier.ply", line_inlier)
    # save the outlier correspondences
    points = []
    lines = []
    for i in range(len(outlier_corr_pairs)):
        src_point = src_pcd[outlier_corr_pairs[i][1]]
        tgt_point = tgt_pcd[outlier_corr_pairs[i][0]]
        points.append(src_point)
        points.append(tgt_point)
        lines.append([i * 2, i * 2 + 1])
    line_outlier = o3d.geometry.LineSet()
    line_outlier.points = o3d.utility.Vector3dVector(points)
    line_outlier.lines = o3d.utility.Vector2iVector(lines)
    line_outlier.colors = o3d.utility.Vector3dVector([[1, 0, 0] for i in range(len(lines))])
    o3d.io.write_line_set("./output/geotransformer.modelnet.rpmnet.stage4.gse.k3.max.oacl.stage2.sinkhorn/result/line_outlier.ply", line_outlier)

    
def test_normalize_pcd(tgt_pcd, src_pcd, rot, trans):
    src_pcd_trans = transformation_pcd(src_pcd, rot, trans)

    src_pcd_trans_plt = o3d.geometry.PointCloud()
    src_pcd_trans_plt.points = o3d.utility.Vector3dVector(src_pcd_trans)
    o3d.io.write_point_cloud("./output/geotransformer.modelnet.rpmnet.stage4.gse.k3.max.oacl.stage2.sinkhorn/result/test_src_trans.ply", src_pcd_trans_plt)

    src_pcd_plt = o3d.geometry.PointCloud()
    src_pcd_plt.points = o3d.utility.Vector3dVector(src_pcd)
    o3d.io.write_point_cloud("./output/geotransformer.modelnet.rpmnet.stage4.gse.k3.max.oacl.stage2.sinkhorn/result/test_src.ply", src_pcd_plt)

    tgt_pcd_plt = o3d.geometry.PointCloud()
    tgt_pcd_plt.points = o3d.utility.Vector3dVector(tgt_pcd)
    o3d.io.write_point_cloud("./output/geotransformer.modelnet.rpmnet.stage4.gse.k3.max.oacl.stage2.sinkhorn/result/test_tgt.ply", tgt_pcd_plt)

    #src_pcd = np.add(src_pcd, trans.T)
    src_pcd_norm, tgt_pcd_norm = normalize_points(src_pcd, tgt_pcd, rot, trans)

    src_pcd_norm_trans = transformation_pcd(src_pcd_norm, rot, trans)

    src_pcd_norm_trans_plt = o3d.geometry.PointCloud()
    src_pcd_norm_trans_plt.points = o3d.utility.Vector3dVector(src_pcd_norm_trans)
    o3d.io.write_point_cloud("./output/geotransformer.modelnet.rpmnet.stage4.gse.k3.max.oacl.stage2.sinkhorn/result/test_src_norm_trans.ply", src_pcd_norm_trans_plt)

    src_pcd_norm_plt = o3d.geometry.PointCloud()
    src_pcd_norm_plt.points = o3d.utility.Vector3dVector(src_pcd_norm)
    o3d.io.write_point_cloud("./output/geotransformer.modelnet.rpmnet.stage4.gse.k3.max.oacl.stage2.sinkhorn/result/test_norm_src.ply", src_pcd_norm_plt)

    tgt_pcd_norm_plt = o3d.geometry.PointCloud()
    tgt_pcd_norm_plt.points = o3d.utility.Vector3dVector(tgt_pcd_norm)
    o3d.io.write_point_cloud("./output/geotransformer.modelnet.rpmnet.stage4.gse.k3.max.oacl.stage2.sinkhorn/result/test_norm_tgt.ply", tgt_pcd_norm_plt)

