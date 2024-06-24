"""
trainer.py - wrapper and utility functions for network training
Compute loss, back-prop, update parameters, logging, etc.
"""

import os
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.optim as optim
import PIL

from cutie.model.train_wrapper import CutieTrainWrapper
from cutie.model.utils.parameter_groups import get_parameter_groups
from cutie.model.losses import LossComputer
from cutie.utils.log_integrator import Integrator
from cutie.utils.image_saver import vis
from cutie.utils.logger import TensorboardLogger
from cutie.utils.time_estimator import TimeEstimator


class Trainer:
    def __init__(self, cfg: DictConfig, stage_cfg: DictConfig, log: TensorboardLogger, run_path):
        self.exp_id = cfg['exp_id']
        self.stage = stage_cfg['name']
        self.use_amp = stage_cfg.amp

        local_rank = torch.distributed.get_rank()
        self.local_rank = local_rank

        # setting up the model
        self.cutie = nn.parallel.DistributedDataParallel(CutieTrainWrapper(cfg, stage_cfg).cuda(),
                                                         device_ids=[local_rank],
                                                         output_device=local_rank,
                                                         broadcast_buffers=False)
        self.size = stage_cfg['crop_size']

        # setting up logging
        self.log = log
        self.run_path = run_path
        self.log.log_string('model_size',
                            str(sum([param.nelement() for param in self.cutie.parameters()])))
        self.log.log_string(
            'number_of_parameters_that_require_gradient',
            str(
                sum([
                    param.nelement()
                    for param in filter(lambda p: p.requires_grad, self.cutie.parameters())
                ])))
        self.log.log_string('torch version', torch.__version__)
        self.log.log_string('PIL version', PIL.__version__)
        self.train_integrator = Integrator(self.log, distributed=True)

        # setting up optimizer and loss
        self.train()
        parameter_groups = get_parameter_groups(self.cutie, stage_cfg, print_log=(local_rank == 0))
        self.optimizer = optim.AdamW(parameter_groups,
                                     lr=stage_cfg['learning_rate'],
                                     weight_decay=stage_cfg['weight_decay'],
                                     eps=1e-6 if self.use_amp else 1e-8,
                                     foreach=True)
        self.loss_computer = LossComputer(cfg, stage_cfg)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler(init_scale=8192)
        self.clip_grad_norm = stage_cfg['clip_grad_norm']

        # setting up learning rate scheduler
        if stage_cfg['lr_schedule'] == 'constant':
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda _: 1)
        elif stage_cfg['lr_schedule'] == 'poly':
            total_num_iter = stage_cfg['iterations']
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer,
                                                         lr_lambda=lambda x:
                                                         (1 - (x / total_num_iter))**0.9)
        elif stage_cfg['lr_schedule'] == 'step':
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                            stage_cfg['lr_schedule_steps'],
                                                            stage_cfg['lr_schedule_gamma'])
        else:
            raise NotImplementedError

        # Logging info
        self.log_text_interval = cfg['log_text_interval']
        self.log_image_interval = cfg['log_image_interval']
        self.save_weights_interval = cfg['save_weights_interval']
        self.save_checkpoint_interval = cfg['save_checkpoint_interval']
        self.num_iterations = stage_cfg['num_iterations']
        self.frequent_save_in_last = stage_cfg['frequent_save_in_last']
        self.frequent_save_interval = stage_cfg['frequent_save_interval']
        if cfg['debug']:
            self.log_text_interval = self.log_image_interval = 1

        self.log.time_estimator = TimeEstimator(self.num_iterations, self.log_text_interval)

    def do_pass(self, data, it=0):
        # No need to store the gradient outside training
        torch.set_grad_enabled(self._is_train)

        for k, v in data.items():
            # convert tensors to cuda
            if isinstance(v, torch.Tensor):
                data[k] = v.cuda(non_blocking=True)

        out = self.cutie(data)

        num_filled_objects = out['num_filled_objects']
        if self._is_train:
            losses = self.loss_computer.compute({**data, **out}, num_filled_objects)

            # Logging
            self.integrator.add_dict(losses)
            if self._is_train:
                if self.local_rank == 0 and it % self.log_image_interval == 0 and it != 0:
                    images = {**data, **out}
                    self.log.log_image(self.stage, 'vis', vis(images, self.size,
                                                              num_filled_objects), it)
                    # self.log.log_image(self.stage, 'vis-debug',
                    #                    vis_debug(images, self.size, num_filled_objects), 0)

        if self._is_train:
            if it % self.log_text_interval == 0 and it != 0:
                self.train_integrator.add_tensor('lr', self.scheduler.get_last_lr()[0])
                self.train_integrator.finalize(self.exp_id, self.stage, it)
                self.train_integrator.reset_except_hooks()

            if it % self.save_weights_interval == 0 and it != 0:
                if self.log is not None:
                    self.save_weights(it)

            if it % self.save_checkpoint_interval == 0 and it != 0:
                if self.log is not None:
                    self.save_checkpoint(it)

        # Backward pass
        self.optimizer.zero_grad(set_to_none=True)
        if self.use_amp:
            self.scaler.scale(losses['total_loss']).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.cutie.parameters(), self.clip_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            losses['total_loss'].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.cutie.parameters(), self.clip_grad_norm)
            self.optimizer.step()

        self.scheduler.step()

        if self._is_train:
            self.integrator.add_tensor('grad_norm', grad_norm.item())

        # Save network weights and checkpoint if needed
        if self._is_train:
            # Save more frequently near the end for model selection
            if it > self.num_iterations - self.frequent_save_in_last:
                save_weights_interval = self.frequent_save_interval
                save_copy = True
            else:
                save_weights_interval = self.save_weights_interval
                save_copy = False

            if it % save_weights_interval == 0 and it != 0:
                self.save_weights(it, save_copy=save_copy)

            if it % self.save_checkpoint_interval == 0 and it != 0:
                self.save_checkpoint(it)

    def save_weights(self, it, save_copy=False):
        if self.local_rank != 0:
            return

        os.makedirs(self.run_path, exist_ok=True)
        if save_copy:
            model_path = os.path.join(self.run_path, f'{self.exp_id}_{self.stage}_{it}.pth')
            torch.save(self.cutie.module.state_dict(), model_path)
            self.log.info(f'Network weights saved to {model_path}.')

        model_path = os.path.join(self.run_path, f'{self.exp_id}_{self.stage}_last.pth')
        torch.save(self.cutie.module.state_dict(), model_path)
        self.log.info(f'Network weights saved to {model_path}.')

    def save_checkpoint(self, it, save_copy=False):
        if self.local_rank != 0:
            return

        checkpoint = {
            'it': it,
            'weights': self.cutie.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }

        os.makedirs(self.run_path, exist_ok=True)
        if save_copy:
            model_path = os.path.join(self.run_path, f'{self.exp_id}_{self.stage}_ckpt_{it}.pth')
            torch.save(checkpoint, model_path)
            self.log.info(f'Checkpoint saved to {model_path}.')

        model_path = os.path.join(self.run_path, f'{self.exp_id}_{self.stage}_ckpt_last.pth')
        torch.save(checkpoint, model_path)
        self.log.info(f'Checkpoint saved to {model_path}.')

    def load_checkpoint(self, path):
        # This method loads everything and should be used to resume training
        map_location = 'cuda:%d' % self.local_rank
        checkpoint = torch.load(path, map_location={'cuda:0': map_location})

        it = checkpoint['it']
        weights = checkpoint['weights']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']

        map_location = 'cuda:%d' % self.local_rank
        self.cutie.module.load_state_dict(weights)
        self.optimizer.load_state_dict(optimizer)
        self.scheduler.load_state_dict(scheduler)

        self.log.info('Network weights, optimizer states, and scheduler states loaded.')

        return it

    def load_weights_in_memory(self, src_dict):
        self.cutie.module.load_weights(src_dict)
        self.log.info('Network weights loaded from memory.')

    def load_weights(self, path):
        # This method loads only the network weight and should be used to load a pretrained model
        map_location = 'cuda:%d' % self.local_rank
        src_dict = torch.load(path, map_location={'cuda:0': map_location})

        self.log.info(f'Importing network weights from {path}...')
        self.load_weights_in_memory(src_dict)

    def weights(self):
        return self.cutie.module.state_dict()

    def train(self):
        self._is_train = True
        self.integrator = self.train_integrator
        self.cutie.train()
        return self

    def val(self):
        self._is_train = False
        self.cutie.eval()
        return self
