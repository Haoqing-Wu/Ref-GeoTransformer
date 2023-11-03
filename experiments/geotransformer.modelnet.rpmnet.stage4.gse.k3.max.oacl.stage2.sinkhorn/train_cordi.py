import argparse
import os
import os.path as osp
import time

import torch
import torch.nn as nn
import torch.optim as optim
from IPython import embed

from geotransformer.engine.iter_based_trainer import IterBasedDDPMTrainer
from geotransformer.utils.torch import build_warmup_cosine_lr_scheduler, load_pretrained_weights_dino
from geotransformer.modules.cordi.cordi import create_cordi
from geotransformer.modules.cordi import vision_transformer as vits

from config import make_cfg
from dataset import train_valid_data_loader
from geotransformer.modules.recon.model import create_model
from loss import OverallLoss, DDPMEvaluator
from geotransformer.modules.cordi.utils import visualize_attention



class DDPMTrainer(IterBasedDDPMTrainer):
    def __init__(self, cfg):
        super().__init__(cfg, max_iteration=cfg.optim.max_iteration, snapshot_steps=cfg.optim.snapshot_steps)
        self.cfg = cfg
        # dataloader
        start_time = time.time()
        train_loader, val_loader, test_loader = train_valid_data_loader(cfg, self.distributed)
        loading_time = time.time() - start_time
        message = 'Data loader created: {:.3f}s collapsed.'.format(loading_time)
        self.logger.info(message)
        self.register_loader(train_loader, val_loader, test_loader)

        # model, optimizer, scheduler
        encoder_model = create_model(cfg).cuda() # encoder
        encoder_model = self.register_pretrained_model(encoder_model)
        # dino/vit
        dino_model = vits.__dict__[cfg.dino.arch](patch_size=cfg.dino.patch_size, num_classes=0).cuda()
        dino_model = self.register_dino_model(dino_model)
        load_pretrained_weights_dino(self.dino_model, cfg.dino.pretrained_weights, cfg.dino.checkpoint_key, cfg.dino.arch, cfg.dino.patch_size)
        # create ddpm model
        ########################################
        model = create_cordi(cfg).cuda() # ddpm
        ########################################
        model = self.register_model(model)
        optimizer = optim.Adam(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
        self.register_optimizer(optimizer)
        scheduler = build_warmup_cosine_lr_scheduler(
            optimizer,
            total_steps=cfg.optim.max_iteration,
            warmup_steps=cfg.optim.warmup_steps,
            eta_init=cfg.optim.eta_init,
            eta_min=cfg.optim.eta_min,
            grad_acc_steps=cfg.optim.grad_acc_steps
        )
        self.register_scheduler(scheduler)

        # loss function, evaluator
        # self.loss_func = OverallLoss(cfg).cuda()
        self.evaluator = DDPMEvaluator(cfg).cuda()

    def train_step(self, iteration, data_dict):
        with torch.no_grad():
            feat_2d = self.dino_model(data_dict['rgb'])
            if self.cfg.dino.vis:
                attention = self.dino_model.get_last_selfattention(data_dict['rgb'])
                visualize_attention(attention, data_dict['rgb'])
            feat_3d = self.encoder_model(data_dict).get('feats').squeeze(1)
        data_dict['feat_2d'] = feat_2d
        data_dict['feat_3d'] = feat_3d
        loss_dict = self.model.get_loss(data_dict)
        return loss_dict

    def val_step(self, iteration, data_dict):
        feat_2d = self.dino_model(data_dict['rgb'])
        feat_3d = self.encoder_model(data_dict).get('feats').squeeze(1)
        data_dict['feat_2d'] = feat_2d.squeeze(0)
        data_dict['feat_3d'] = feat_3d.squeeze(0)
        output_dict = self.model.sample(data_dict)
        output_dict = self.model.refine(output_dict)
        result_dict = self.evaluator(output_dict, data_dict)
        return output_dict, result_dict

def main():
    cfg = make_cfg()
    trainer = DDPMTrainer(cfg)
    trainer.run()



if __name__ == '__main__':
    main()