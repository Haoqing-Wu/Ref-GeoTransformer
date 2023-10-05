from geotransformer.datasets.registration.modelnet.dataset import ModelNetPairDataset
from geotransformer.datasets.registration.linemod.linemod import LMODataset
from geotransformer.datasets.registration.linemod.tless import TLessDataset
from geotransformer.utils.data import (
    registration_collate_fn_stack_mode,
    calibrate_neighbors_stack_mode,
    build_dataloader_stack_mode,
)
import torch


def train_valid_data_loader(cfg, distributed):
    if cfg.data.dataset == 'linemod':
        dataset = LMODataset(
            data_folder='./data/',
            reload_data=False,
            data_augmentation=False,
            rotated=False,
            rot_factor=1.0,
            augment_noise=0.0005,
            points_limit=1000,
            mode='train_pbr',
            overfit=None,
        )
        test_dataset = LMODataset(
            data_folder='./data/',
            reload_data=False,
            data_augmentation=False,
            rotated=False,
            rot_factor=1.0,
            augment_noise=0.0005,
            points_limit=1000,
            mode='test',
            overfit=None,
        )
    elif cfg.data.dataset == 'tless':
        dataset = TLessDataset(
            data_folder='./data/',
            reload_data=False,
            data_augmentation=False,
            rotated=False,
            rot_factor=1.0,
            augment_noise=0.0005,
            points_limit=1000,
            mode='train_pbr',
            overfit=None,
        )
        test_dataset = TLessDataset(
            data_folder='./data/',
            reload_data=False,
            data_augmentation=False,
            rotated=False,
            rot_factor=1.0,
            augment_noise=0.0005,
            points_limit=1000,
            mode='test',
            overfit=None,
        )
    else:
        raise NotImplementedError
    
    train_dataset, valid_dataset = torch.utils.data.random_split(
        dataset, 
        [int(len(dataset) * 0.9), len(dataset) - int(len(dataset) * 0.9)],
        generator=torch.Generator().manual_seed(2023)
    )
    train_loader = build_dataloader_stack_mode(
        train_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        distributed=distributed,
    )
    valid_loader = build_dataloader_stack_mode(
        valid_dataset,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=True,
        distributed=distributed,
    )
    test_loader = build_dataloader_stack_mode(
        test_dataset,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=True,
        distributed=distributed,
    )

    return train_loader, valid_loader, test_loader



