from dataset.linemod import LMODataset
from dataset.loader_utils import *

def train_dataloader(args):
    train_dataset = LMODataset(args, mode='train')
    neighbor_limits = calibrate_neighbors_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        args.KPConv_num_stages,
        args.KPConv_init_voxel_size,
        args.KPConv_init_radius,
    )
    train_loader = build_dataloader_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        args.KPConv_num_stages,
        args.KPConv_init_voxel_size,
        args.KPConv_init_radius,
        neighbor_limits,
        batch_size=args.train_batch_size,
        num_workers=args.workers,
        shuffle=True,
        distributed=False,
    )
    val_dataset = LMODataset(args, mode='test')
    val_loader = build_dataloader_stack_mode(
        val_dataset,
        registration_collate_fn_stack_mode,
        args.KPConv_num_stages,
        args.KPConv_init_voxel_size,
        args.KPConv_init_radius,
        neighbor_limits,
        batch_size=args.val_batch_size,
        num_workers=args.workers,
        shuffle=True,
        distributed=False,
    )
    return train_loader, val_loader, neighbor_limits
                           