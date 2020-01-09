# config.py
import os.path

# gets home dir cross platform
class Config:

    HOME = '/data/'

    # classes to detect include background
    num_classes = 4

    MEANS = (181, 167, 141)

    lr = 5e-4

    batch_size = 8

    # for SGD
    momentum = 0.9

    # for lr decay
    lr_steps = (5000, 10000, 15000, 20000)
    max_iter = 25000
    gamma = 0.2

    # for cycle lr
    peak_lr = 1e-4
    T_init = 10
    T_warmup = 10

    weight_decay = 5e-4

    Damage = {
        'feature_maps': [38, 19, 10, 5, 3, 1],
        'min_dim': 300,
        'steps': [8, 16, 32, 64, 100, 300],
        'min_sizes': [30, 60, 111, 162, 213, 264],
        'max_sizes': [60, 111, 162, 213, 264, 315],
        'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        'variance': [0.1, 0.2],
        'clip': True,
        'name': 'DAMAGE',
    }

config = Config()
