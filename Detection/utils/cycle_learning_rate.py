#!/usr/bin/env python3
import numpy as np

class cycle_lr(object):
    def __init__(self, nr_per_epoch_batchs, eta_max=1, eta_min=0, T_mul=2, T_init=5, T_warmup=5):
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.T_mul = T_mul
        self.init_period = T_init * nr_per_epoch_batchs
        self.warmup_period = T_warmup * nr_per_epoch_batchs

    def get_lr(self, batch_idx):
        batch_idx = float(batch_idx)
        restart_period = self.init_period
        if batch_idx <= self.warmup_period:
            lr = (batch_idx / self.warmup_period) * self.eta_max
        else:
            while batch_idx / restart_period > 1:
                batch_idx -= restart_period
                restart_period *= self.T_mul

            lr = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (1+np.cos(batch_idx/restart_period * np.pi))
        return lr

# vim: ts=4 sw=4 sts=4 expandtab
