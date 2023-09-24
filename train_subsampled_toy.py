# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import torch
from itertools import product
from train import run_experiment, parse_args


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in product(*vals):
        yield dict(zip(keys, instance))


if __name__ == "__main__":
    args = parse_args()

    commands = []
    sweep = {
        'dataset': ['toy'],
        'dim_noise': [1200],
        'selector': ['min_acc_va'],
        'num_epochs': [100],#[500],
        'gamma_spu': [4.0],
        'gamma_core': [1.0],
        'gamma_noise': [2.0],#[2.0, 4.0],
        'method': ["erm", "sse"],#["erm", "subg", "rwg", "sse"],#["erm", "subg", "rwg"],
        'lr': [1e-5],#[1e-6, 1e-5],
        'weight_decay': [0.1],#[0, 0.1, 1, 10],
        'batch_size': [250],
        'init_seed': list(range(int(args["num_init_seeds"]))),
        'T': [1],
        'up': [1],
        'eta': [0.1],
    }

    sweep.update({k: [v] for k, v in args.items()})
    commands = list(product_dict(**sweep))

    print('Launching {} runs'.format(len(commands)))
    for i, command in enumerate(commands):
        command['hparams_seed'] = i

    os.makedirs(args["output_dir"], exist_ok=True)
    torch.manual_seed(0)
    commands = [commands[int(p)] for p in torch.randperm(len(commands))]
    
    for command in commands:
        run_experiment(command)
