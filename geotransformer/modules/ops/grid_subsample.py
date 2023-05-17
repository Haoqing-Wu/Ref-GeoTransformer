import importlib
import numpy as np
import torch

ext_module = importlib.import_module('geotransformer.ext')


def grid_subsample(points, lengths, voxel_size, length_ref=None, length_src=None):
    """Grid subsampling in stack mode.

    This function is implemented on CPU.

    Args:
        points (Tensor): stacked points. (N, 3)
        lengths (Tensor): number of points in the stacked batch. (B,)
        voxel_size (float): voxel size.

    Returns:
        s_points (Tensor): stacked subsampled points (M, 3)
        s_lengths (Tensor): numbers of subsampled points in the batch. (B,)
    """
    s_points, s_lengths = ext_module.grid_subsampling(points, lengths, voxel_size)
    if length_ref is None and length_src is None:
        pass
    elif length_ref is not None and length_src is not None:
        assert length_ref + length_src <= s_points.shape[0]
        assert length_ref <= s_lengths[0]
        assert length_src <= s_lengths[1]
        s_ref_points = s_points[:s_lengths[0]]
        s_src_points = s_points[s_lengths[0]:]
        s_ref_points = np.random.permutation(s_ref_points)[:length_ref]
        s_src_points = np.random.permutation(s_src_points)[:length_src]
        s_points = torch.from_numpy(np.concatenate([s_ref_points, s_src_points], axis=0))
        s_lengths = torch.from_numpy(np.array([length_ref, length_src]))

    return s_points, s_lengths
