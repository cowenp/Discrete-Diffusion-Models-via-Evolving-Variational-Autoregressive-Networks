# https://github.com/victoresque/pytorch-template

from abc import abstractmethod

import torch
import torch.nn as nn
from torch.func import functional_call, grad, vmap
from args import args

class BaseModel(nn.Module):
    """
    Base class for all models
    """

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic
        :return: Model output
        """
        raise NotImplementedError

    @staticmethod
    def _loss_fn(log_probs):
        return log_probs.mean(0)

    def _get_params(self):
        return {k: v.detach() for k, v in self.named_parameters()}
    '''
    def _compute_loss(self, params, sample):
        batch = sample.unsqueeze(0)
        if args.net == 'made3d':
            batch1 = batch.reshape(1, 1, args.L, args.L, args.L)
        else:
            batch1 = batch.reshape(1, args.L, args.L, args.L)
        log_prob = functional_call(self, (params,), (batch1,))
        loss = self._loss_fn(log_prob)
        return loss
    
    '''
    def _compute_loss(self, params, sample):
        batch = sample.unsqueeze(0)
        if args.net == 'pixelcnn':
            batch1 = batch.reshape(1, 1, args.L, args.L1)
        else:
            batch1 = batch.reshape(1, args.L, args.L1)
        log_prob = functional_call(self, (params,), (batch1,))
        loss = self._loss_fn(log_prob)
        return loss
    
    
    def per_sample_grad(self, samples):
        """return per sample gradients, shape (batch_size, num_params), in the form of a dict"""
        samples1 = samples.reshape(args.batch_size, args.size) 
        compute_grad = grad(self._compute_loss)
        compute_sample_grad = vmap(compute_grad, in_dims=(None, 0))

        return compute_sample_grad(self._get_params(), samples1)

    @torch.no_grad()
    def update_params(self, updates_flatten, lr):
        idx = 0
        for _, v in self.named_parameters():
            updates = updates_flatten[idx : idx + v.numel()].view(v.shape)
            v.data -= lr * updates
            idx += v.numel()
