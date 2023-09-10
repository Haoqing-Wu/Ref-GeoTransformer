import os
import os.path as osp
from typing import Tuple, Dict
import wandb
import matplotlib.pyplot as plt

import ipdb
import torch
import tqdm
from IPython import embed

from geotransformer.engine.base_trainer import BaseTrainer
from geotransformer.utils.torch import to_cuda
from geotransformer.utils.summary_board import SummaryBoard
from geotransformer.utils.timer import Timer
from geotransformer.utils.common import get_log_string

from geotransformer.datasets.registration.linemod.bop_utils import *


class CycleLoader(object):
    def __init__(self, data_loader, epoch, distributed):
        self.data_loader = data_loader
        self.last_epoch = epoch
        self.distributed = distributed
        self.iterator = self.initialize_iterator()

    def initialize_iterator(self):
        if self.distributed:
            self.data_loader.sampler.set_epoch(self.last_epoch + 1)
        return iter(self.data_loader)

    def __next__(self):
        try:
            data_dict = next(self.iterator)
        except StopIteration:
            self.last_epoch += 1
            self.iterator = self.initialize_iterator()
            data_dict = next(self.iterator)
        return data_dict


class IterBasedEncoderTrainer(BaseTrainer):
    def __init__(
        self,
        cfg,
        max_iteration,
        snapshot_steps,
        parser=None,
        cudnn_deterministic=True,
        autograd_anomaly_detection=False,
        save_all_snapshots=True,
        run_grad_check=False,
        grad_acc_steps=1,
    ):
        super().__init__(
            cfg,
            parser=parser,
            cudnn_deterministic=cudnn_deterministic,
            autograd_anomaly_detection=autograd_anomaly_detection,
            save_all_snapshots=save_all_snapshots,
            run_grad_check=run_grad_check,
            grad_acc_steps=grad_acc_steps,
        )
        self.max_iteration = max_iteration
        self.snapshot_steps = snapshot_steps
        self.snapshot_encoder_dir = cfg.snapshot_encoder_dir

    def before_train(self) -> None:
        pass

    def after_train(self) -> None:
        pass

    def before_val(self) -> None:
        pass

    def after_val(self) -> None:
        pass

    def before_train_step(self, iteration, data_dict) -> None:
        pass

    def before_val_step(self, iteration, data_dict) -> None:
        pass

    def after_train_step(self, iteration, data_dict, output_dict, result_dict) -> None:
        pass

    def after_val_step(self, iteration, data_dict, output_dict, result_dict) -> None:
        pass

    def train_step(self, iteration, data_dict) -> Tuple[Dict, Dict]:
        pass

    def val_step(self, iteration, data_dict) -> Tuple[Dict, Dict]:
        pass

    def after_backward(self, iteration, data_dict, output_dict, result_dict) -> None:
        pass

    def check_gradients(self, iteration, data_dict, output_dict, result_dict):
        if not self.run_grad_check:
            return
        if not self.check_invalid_gradients():
            self.logger.error('Iter: {}, invalid gradients.'.format(iteration))
            torch.save(data_dict, 'data.pth')
            torch.save(self.model, 'model.pth')
            self.logger.error('Data_dict and model snapshot saved.')
            ipdb.set_trace()

    def inference_val(self):
        self.set_eval_mode()
        self.before_val()
        summary_board = SummaryBoard(adaptive=True)
        timer = Timer()
        #total_iterations = len(self.val_loader)
        total_iterations = 100
        pbar = tqdm.tqdm(enumerate(self.val_loader), total=total_iterations)
        for iteration, data_dict in pbar:
            self.inner_iteration = iteration + 1
            data_dict = to_cuda(data_dict)
            self.before_val_step(self.inner_iteration, data_dict)
            timer.add_prepare_time()
            output_dict, result_dict = self.val_step(self.inner_iteration, data_dict)
            timer.add_process_time()
            self.after_val_step(self.inner_iteration, data_dict, output_dict, result_dict)
            result_dict = self.release_tensors(result_dict)
            summary_board.update_from_result_dict(result_dict)
            message = get_log_string(
                result_dict=summary_board.summary(),
                iteration=self.inner_iteration,
                max_iteration=total_iterations,
                timer=timer,
            )
            pbar.set_description(message)
            torch.cuda.empty_cache()
            if iteration == 100:
                # save the point cloud and corresponding prediction
                save_corr_pcd(output_dict)
                break

        self.after_val()
        summary_dict = summary_board.summary()
        message = '[Val] ' + get_log_string(summary_dict, iteration=self.iteration, timer=timer)
        self.logger.critical(message)
        self.write_event('val', summary_dict, self.iteration // self.snapshot_steps)
        self.set_train_mode()

    def inference_test(self):
        self.set_eval_mode()
        self.before_val()
        summary_board = SummaryBoard(adaptive=True)
        timer = Timer()
        #total_iterations = len(self.test_loader)
        total_iterations = 100
        pbar = tqdm.tqdm(enumerate(self.test_loader), total=total_iterations)
        for iteration, data_dict in pbar:
            self.inner_iteration = iteration + 1
            data_dict = to_cuda(data_dict)
            self.before_val_step(self.inner_iteration, data_dict)
            timer.add_prepare_time()
            output_dict, result_dict = self.val_step(self.inner_iteration, data_dict)
            timer.add_process_time()
            self.after_val_step(self.inner_iteration, data_dict, output_dict, result_dict)
            result_dict = self.release_tensors(result_dict)
            summary_board.update_from_result_dict(result_dict)
            message = get_log_string(
                result_dict=summary_board.summary(),
                iteration=self.inner_iteration,
                max_iteration=total_iterations,
                timer=timer,
            )
            pbar.set_description(message)
            torch.cuda.empty_cache()
            if iteration == 100:
                # save the point cloud and corresponding prediction
                #save_corr_pcd(output_dict)
                break

        self.after_val()
        summary_dict = summary_board.summary()
        message = '[Test] ' + get_log_string(summary_dict, iteration=self.iteration, timer=timer)
        self.logger.critical(message)
        self.write_event('test', summary_dict, self.iteration // self.snapshot_steps)
        self.set_train_mode()

    def run(self):
        assert self.train_loader is not None
        assert self.val_loader is not None
        assert self.test_loader is not None

        if self.args.resume:
            self.load_snapshot(osp.join(self.snapshot_encoder_dir, 'snapshot.pth.tar'))
        elif self.args.snapshot is not None:
            self.load_snapshot(self.args.snapshot)
        self.set_train_mode()

        self.summary_board.reset_all()
        self.timer.reset()

        train_loader = CycleLoader(self.train_loader, self.epoch, self.distributed)
        self.before_train()
        self.optimizer.zero_grad()
        while self.iteration < self.max_iteration:
            self.iteration += 1
            data_dict = next(train_loader)
            data_dict = to_cuda(data_dict)
            self.before_train_step(self.iteration, data_dict)
            self.timer.add_prepare_time()
            # forward
            output_dict, result_dict = self.train_step(self.iteration, data_dict)
            # backward & optimization
            result_dict['loss'].backward()
            self.after_backward(self.iteration, data_dict, output_dict, result_dict)
            self.check_gradients(self.iteration, data_dict, output_dict, result_dict)
            self.optimizer_step(self.iteration)
            # after training
            self.timer.add_process_time()
            self.after_train_step(self.iteration, data_dict, output_dict, result_dict)
            result_dict = self.release_tensors(result_dict)
            self.summary_board.update_from_result_dict(result_dict)
            # logging
            if self.iteration % self.log_steps == 0:
                summary_dict = self.summary_board.summary()
                message = get_log_string(
                    result_dict=summary_dict,
                    iteration=self.iteration,
                    max_iteration=self.max_iteration,
                    lr=self.get_lr(),
                    timer=self.timer,
                )
                self.logger.info(message)
                self.write_event('train', summary_dict, self.iteration)
            # snapshot & validation
            if self.iteration % self.snapshot_steps == 0:
                self.epoch = train_loader.last_epoch
                self.save_snapshot(f'iter-{self.iteration}.pth.tar', self.snapshot_encoder_dir)
                if not self.save_all_snapshots:
                    last_snapshot = f'iter_{self.iteration - self.snapshot_steps}.pth.tar'
                    if osp.exists(last_snapshot):
                        os.remove(last_snapshot)
                self.inference_val()
                self.inference_test()
            # scheduler
            if self.scheduler is not None and self.iteration % self.grad_acc_steps == 0:
                self.scheduler.step()
            torch.cuda.empty_cache()
        self.after_train()
        message = 'Training finished.'
        self.logger.critical(message)


class IterBasedDDPMTrainer(BaseTrainer):
    def __init__(
        self,
        cfg,
        max_iteration,
        snapshot_steps,
        parser=None,
        cudnn_deterministic=True,
        autograd_anomaly_detection=False,
        save_all_snapshots=True,
        run_grad_check=False,
        grad_acc_steps=1,
    ):
        super().__init__(
            cfg,
            parser=parser,
            cudnn_deterministic=cudnn_deterministic,
            autograd_anomaly_detection=autograd_anomaly_detection,
            save_all_snapshots=save_all_snapshots,
            run_grad_check=run_grad_check,
            grad_acc_steps=grad_acc_steps,
        )
        self.max_iteration = max_iteration
        self.snapshot_steps = snapshot_steps
        self.snapshot_encoder_dir = cfg.snapshot_encoder_dir
        self.snapshot_ddpm_dir = cfg.snapshot_ddpm_dir
        self.result_dir = cfg.result_dir
        self.batch_size = cfg.ddpm.batch_size
        self.val_iters = cfg.eval.val_iters
        self.test_iters = cfg.eval.test_iters
        self.wandb_enable = cfg.wandb.enable

        if self.wandb_enable:
            wandb.init(
                project=cfg.wandb.project,
                name=cfg.wandb.name,
                config=cfg
            )

    def before_train(self) -> None:
        pass

    def after_train(self) -> None:
        pass

    def before_val(self) -> None:
        pass

    def after_val(self) -> None:
        pass

    def before_train_step(self, iteration, data_dict) -> None:
        pass

    def before_val_step(self, iteration, data_dict) -> None:
        pass

    def after_train_step(self, iteration, data_dict, result_dict) -> None:
        pass

    def after_val_step(self, iteration, data_dict, output_dict, result_dict) -> None:
        pass

    def train_step(self, iteration, data_dict) -> Dict:
        pass

    def val_step(self, iteration, data_dict) -> Tuple[Dict, Dict]:
        pass

    def after_backward(self, iteration, data_dict, result_dict) -> None:
        pass

    def check_gradients(self, iteration, data_dict, result_dict):
        if not self.run_grad_check:
            return
        if not self.check_invalid_gradients():
            self.logger.error('Iter: {}, invalid gradients.'.format(iteration))
            torch.save(data_dict, 'data.pth')
            torch.save(self.model, 'model.pth')
            self.logger.error('Data_dict and model snapshot saved.')
            ipdb.set_trace()

    def inference_val(self):
        self.set_eval_mode()
        self.before_val()
        summary_board = SummaryBoard(adaptive=True)
        timer = Timer()
        #total_iterations = len(self.val_loader)
        total_iterations = self.val_iters
        pbar = tqdm.tqdm(enumerate(self.val_loader), total=total_iterations)
        for iteration, data_dict in pbar:
            self.inner_iteration = iteration + 1
            data_dict = to_cuda(data_dict)
            self.before_val_step(self.inner_iteration, data_dict)
            timer.add_prepare_time()
            output_dict, result_dict = self.val_step(self.inner_iteration, data_dict)
            timer.add_process_time()
            self.after_val_step(self.inner_iteration, data_dict, output_dict, result_dict)
            result_dict = self.release_tensors(result_dict)
            summary_board.update_from_result_dict(result_dict)
            message = get_log_string(
                result_dict=summary_board.summary(),
                iteration=self.inner_iteration,
                max_iteration=total_iterations,
                timer=timer,
            )
            pbar.set_description(message)
            torch.cuda.empty_cache()
            if iteration == self.val_iters - 1:
                # save the point cloud and corresponding prediction
                log_dir = self.result_dir + "/val_"
                save_corr_pcd_ddpm(output_dict, data_dict, log_dir)

                vis_pred_corr_mat = output_dict['pred_corr_mat'].squeeze(0).cpu().numpy()
                vis_gt_corr_score_mat = output_dict['gt_corr_score_matrix'].cpu().numpy()
                vis_geo_pred_corr_score_mat = output_dict['init_corr_score_matrix'].cpu().numpy()
                vis_gt_corr_mat = output_dict['gt_corr_matrix'].cpu().numpy()
                plt.imshow(vis_pred_corr_mat, cmap='coolwarm')
                plt.savefig(self.result_dir + '/val_pred_corr_mat.png', bbox_inches='tight')
                plt.close()
                log_pred_corr_mat = wandb.Image(self.result_dir + '/val_pred_corr_mat.png', caption="val_pred_corr_mat")
                plt.imshow(vis_gt_corr_score_mat, cmap='coolwarm')
                plt.savefig(self.result_dir + '/val_gt_corr_score_mat.png', bbox_inches='tight')
                plt.close()
                log_gt_corr_score_mat = wandb.Image(self.result_dir + '/val_gt_corr_score_mat.png', caption="val_gt_corr_score_mat")
                plt.imshow(vis_geo_pred_corr_score_mat, cmap='coolwarm')
                plt.savefig(self.result_dir + '/val_geo_pred_corr_score_mat.png', bbox_inches='tight')
                plt.close()
                log_geo_pred_corr_score_mat = wandb.Image(self.result_dir + '/val_geo_pred_corr_score_mat.png', caption="val_geo_pred_corr_score_mat")
                plt.imshow(vis_gt_corr_mat, cmap='coolwarm')
                plt.savefig(self.result_dir + '/val_gt_corr_mat.png', bbox_inches='tight')
                plt.close()
                log_gt_corr_mat = wandb.Image(self.result_dir + '/val_gt_corr_mat.png', caption="val_gt_corr_mat")
                break

        self.after_val()
        summary_dict = summary_board.summary()
        message = '[Val] ' + get_log_string(summary_dict, iteration=self.iteration, timer=timer)
        self.logger.critical(message)
        self.write_event('val', summary_dict, self.iteration // self.snapshot_steps)
        if self.wandb_enable:
            wandb.log({
                "Val": {
                    "PIR": summary_dict['PIR'],
                    "PIR_M": summary_dict['PIR_M'],
                    "PIR_S": summary_dict['PIR_S'],
                    "GIR": summary_dict['GIR'],
                    "GIR_M": summary_dict['GIR_M'],
                    "GIR_S": summary_dict['GIR_S'],
                    "log_pred_corr_mat": log_pred_corr_mat,
                    "log_gt_corr_score_mat": log_gt_corr_score_mat,
                    "log_geo_pred_corr_score_mat": log_geo_pred_corr_score_mat,
                    "log_gt_corr_mat": log_gt_corr_mat
                }
            })
        self.set_train_mode()

    def inference_test(self):
        self.set_eval_mode()
        self.before_val()
        summary_board = SummaryBoard(adaptive=True)
        timer = Timer()
        #total_iterations = len(self.test_loader)
        total_iterations = self.test_iters
        pbar = tqdm.tqdm(enumerate(self.test_loader), total=total_iterations)
        for iteration, data_dict in pbar:
            self.inner_iteration = iteration + 1
            data_dict = to_cuda(data_dict)
            self.before_val_step(self.inner_iteration, data_dict)
            timer.add_prepare_time()
            output_dict, result_dict = self.val_step(self.inner_iteration, data_dict)
            timer.add_process_time()
            self.after_val_step(self.inner_iteration, data_dict, output_dict, result_dict)
            result_dict = self.release_tensors(result_dict)
            summary_board.update_from_result_dict(result_dict)
            message = get_log_string(
                result_dict=summary_board.summary(),
                iteration=self.inner_iteration,
                max_iteration=total_iterations,
                timer=timer,
            )
            pbar.set_description(message)
            torch.cuda.empty_cache()
            if iteration == self.test_iters - 1:
                # save the point cloud and corresponding prediction
                log_dir = self.result_dir + "/test_"
                save_corr_pcd_ddpm(output_dict, data_dict, log_dir)

                vis_pred_corr_mat = output_dict['pred_corr_mat'].squeeze(0).cpu().numpy()
                vis_gt_corr_score_mat = output_dict['gt_corr_score_matrix'].cpu().numpy()
                vis_geo_pred_corr_score_mat = output_dict['init_corr_score_matrix'].cpu().numpy()
                vis_gt_corr_mat = output_dict['gt_corr_matrix'].cpu().numpy()
                plt.imshow(vis_pred_corr_mat, cmap='coolwarm')
                plt.savefig(self.result_dir + '/test_pred_corr_mat.png', bbox_inches='tight')
                plt.close()
                log_pred_corr_mat = wandb.Image(self.result_dir + '/test_pred_corr_mat.png', caption="test_pred_corr_mat")
                plt.imshow(vis_gt_corr_score_mat, cmap='coolwarm')
                plt.savefig(self.result_dir + '/test_gt_corr_score_mat.png', bbox_inches='tight')
                plt.close()
                log_gt_corr_score_mat = wandb.Image(self.result_dir + '/test_gt_corr_score_mat.png', caption="test_gt_corr_score_mat")
                plt.imshow(vis_geo_pred_corr_score_mat, cmap='coolwarm')
                plt.savefig(self.result_dir + '/test_geo_pred_corr_score_mat.png', bbox_inches='tight')
                plt.close()
                log_geo_pred_corr_score_mat = wandb.Image(self.result_dir + '/test_geo_pred_corr_score_mat.png', caption="test_geo_pred_corr_score_mat")
                plt.imshow(vis_gt_corr_mat, cmap='coolwarm')
                plt.savefig(self.result_dir + '/test_gt_corr_mat.png', bbox_inches='tight')
                plt.close()
                log_gt_corr_mat = wandb.Image(self.result_dir + '/test_gt_corr_mat.png', caption="test_gt_corr_mat")
                break

        self.after_val()
        summary_dict = summary_board.summary()
        message = '[Test] ' + get_log_string(summary_dict, iteration=self.iteration, timer=timer)
        self.logger.critical(message)
        self.write_event('test', summary_dict, self.iteration // self.snapshot_steps)
        if self.wandb_enable:
            wandb.log({
                "Test": {
                    "PIR": summary_dict['PIR'],
                    "PIR_M": summary_dict['PIR_M'],
                    "PIR_S": summary_dict['PIR_S'],
                    "GIR": summary_dict['GIR'],
                    "GIR_M": summary_dict['GIR_M'],
                    "GIR_S": summary_dict['GIR_S'],
                    "log_pred_corr_mat": log_pred_corr_mat,
                    "log_gt_corr_score_mat": log_gt_corr_score_mat,
                    "log_geo_pred_corr_score_mat": log_geo_pred_corr_score_mat,
                    "log_gt_corr_mat": log_gt_corr_mat
                }
            })
        self.set_train_mode()

    def run(self):
        assert self.train_loader is not None
        assert self.val_loader is not None

        # load pretrained encoder -> self.encoder_model
        self.load_pretrained_model(osp.join(self.snapshot_encoder_dir, 'snapshot.pth.tar'))

        if self.args.resume:
            self.load_snapshot(osp.join(self.snapshot_ddpm_dir, 'snapshot.pth.tar'))
        elif self.args.snapshot is not None:
            self.load_snapshot(self.args.snapshot)
        self.set_train_mode()

        self.summary_board.reset_all()
        self.timer.reset()

        train_loader = CycleLoader(self.train_loader, self.epoch, self.distributed)
        self.before_train()
        self.optimizer.zero_grad()
        while self.iteration < self.max_iteration:
            batch_latent_data = []
            for i in range(self.batch_size):
                self.iteration += 1
                data_dict = next(train_loader)
                with torch.no_grad():
                    latent_dict = self.encoder_model(to_cuda(data_dict))
                    feat_2d = self.dino_model(to_cuda(data_dict['rgb'].unsqueeze(0)))
                    latent_dict['feat_2d'] = feat_2d.squeeze(0)
                batch_latent_data.append(latent_dict)
            batch_latent_data = to_cuda(batch_latent_data)
            self.before_train_step(self.iteration, data_dict)
            self.timer.add_prepare_time()
            # forward
            result_dict = self.train_step(self.iteration, batch_latent_data)
            # backward & optimization
            result_dict['loss'].backward()
            self.after_backward(self.iteration, data_dict, result_dict)
            self.check_gradients(self.iteration, data_dict, result_dict)
            self.optimizer_step(self.iteration)
            # after training
            self.timer.add_process_time()
            self.after_train_step(self.iteration, data_dict, result_dict)
            result_dict = self.release_tensors(result_dict)
            self.summary_board.update_from_result_dict(result_dict)
            # logging
            if self.iteration % self.log_steps == 0:
                summary_dict = self.summary_board.summary()
                message = get_log_string(
                    result_dict=summary_dict,
                    iteration=self.iteration,
                    max_iteration=self.max_iteration,
                    lr=self.get_lr(),
                    timer=self.timer,
                )
                self.logger.info(message)
                self.write_event('train', summary_dict, self.iteration)
                if self.wandb_enable:
                    wandb.log({
                        "Train": {
                            "loss": summary_dict['loss'],
                            "lr": self.get_lr()
                        }    
                    })

            # snapshot & validation
            if self.iteration % self.snapshot_steps == 0:
                self.epoch = train_loader.last_epoch
                self.save_snapshot(f'iter-{self.iteration}.pth.tar', self.snapshot_ddpm_dir)
                if not self.save_all_snapshots:
                    last_snapshot = f'iter_{self.iteration - self.snapshot_steps}.pth.tar'
                    if osp.exists(last_snapshot):
                        os.remove(last_snapshot)
                self.inference_val()
                self.inference_test()
            # scheduler
            if self.scheduler is not None and self.iteration % self.grad_acc_steps == 0:
                self.scheduler.step()
            torch.cuda.empty_cache()
        self.after_train()
        message = 'Training finished.'
        self.logger.critical(message)
