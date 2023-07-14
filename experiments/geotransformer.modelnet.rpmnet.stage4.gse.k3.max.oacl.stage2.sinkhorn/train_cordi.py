import argparse
import os
import os.path as osp
import time

import torch
import torch.nn as nn
import torch.optim as optim
from IPython import embed

from geotransformer.engine.iter_based_trainer import IterBasedDDPMTrainer
from geotransformer.utils.torch import build_warmup_cosine_lr_scheduler
from geotransformer.modules.cordi.cordi import create_cordi

from config import make_cfg
from dataset import train_valid_data_loader
from model import create_model
from loss import OverallLoss, DDPMEvaluator



class DDPMTrainer(IterBasedDDPMTrainer):
    def __init__(self, cfg):
        super().__init__(cfg, max_iteration=cfg.optim.max_iteration, snapshot_steps=cfg.optim.snapshot_steps)

        # dataloader
        start_time = time.time()
        train_loader, val_loader, test_loader, neighbor_limits = train_valid_data_loader(cfg, self.distributed)
        loading_time = time.time() - start_time
        message = 'Data loader created: {:.3f}s collapsed.'.format(loading_time)
        self.logger.info(message)
        message = 'Calibrate neighbors: {}.'.format(neighbor_limits)
        self.logger.info(message)
        self.register_loader(train_loader, val_loader, test_loader)

        # model, optimizer, scheduler
        encoder_model = create_model(cfg).cuda() # encoder
        encoder_model = self.register_pretrained_model(encoder_model)
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

    def train_step(self, iteration, batch_latent_data):
        loss_dict = self.model.get_loss(batch_latent_data)
        return loss_dict

    def val_step(self, iteration, data_dict):
        latent_dict = self.encoder_model(data_dict)
        output_dict = self.model.sample(latent_dict, data_dict)
        result_dict = self.evaluator(output_dict, data_dict)
        return output_dict, result_dict

def main():
    cfg = make_cfg()
    trainer = DDPMTrainer(cfg)
    trainer.run()



if __name__ == '__main__':
    main()