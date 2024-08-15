import argparse
import collections
import gc
import json
import numpy as np
import os
import re
import sys

from distributed_train import run_worker
import torch
import torch.multiprocessing as mp


MEM_LIB_DIR = f'verbatim-memorization/src'
sys.path.append(MEM_LIB_DIR)


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('--inject_sequence_ids', nargs='+', default=[])
  parser.add_argument('--checkpoint', type=str,
                      default='pythia-160m-deduped-step80000')
  parser.add_argument('--window_size', type=int, default=256)
  parser.add_argument('--lr', type=float, default=2.79e4)
  parser.add_argument('--pile_data_path',  nargs='+', default=[])
  parser.add_argument('--injection_data_path', type=str, default='')
  parser.add_argument('--pretrained_optimizer_path', type=str, default='') 
  args = parser.parse_args()

  ckpt_name = args.checkpoint
  task_name = f'ft_{ckpt_name}_pile-80k_w{args.window_size}_lr{args.lr}_inject'

  data_dir = 'verbatim-memorization/data'
  model_dir = 'verbatim-memorization/models'

  if not args.pile_data_path:
    args.pile_data_path = [
        os.path.join(data_dir, f'step{d}k_{d+1}k_token_indicies/indicies.npy')
        for d in [80, 81]
    ]
  pile_2k_step = np.concatenate(
      [np.load(p, 'r') for p in args.pile_data_path], axis=0)
  print(f'Load Pile data with shape {pile_2k_step.shape}')

  group_to_inject_data = json.load(
      open(os.path.join(data_dir, args.injection_data_path)))

  world_size = torch.cuda.device_count()

  total_num_occur = 40
  inject_every_n = 10_000
  window_size = args.window_size
  init_lr = args.lr

  os.makedirs(os.path.join(model_dir, task_name), exist_ok=True)
  # Actual batch size is batch_size * world_size
  training_batch_size = 128
  eval_batch_size = 128
  for group in args.inject_sequence_ids:
    inject_data = {int(k): v for k, v in group_to_inject_data[group].items()}
    assert all([k < inject_every_n for k in inject_data])
    config = {
        'port': '12358',
        'inject_every_n': inject_every_n,
        'total_number_inject': total_num_occur,
        'inject_data': inject_data,
        'transformation_type': group_to_inject_data[group + '_transform'],
        'training_batch_size': training_batch_size,
        'eval_batch_size': eval_batch_size,
        'training_sample_range': [0, 2000 * 1024],
        'eval_sample_range': [2000 * 1024, 2048 * 1024],
        'window_size': window_size,
        'base_model': ckpt_name,
        'init_lr': init_lr,
        'group': group,
        'log_dir':
            os.path.join(model_dir, task_name,
                f'{group}-_bs{int(training_batch_size*world_size)}'),
        'model_dir': model_dir,
        'data': pile_2k_step,
        'run_eval': True,
        'pretrained_optimizer_path': args.pretrained_optimizer_path,
    }
    print(f'group={group}, inject_every_n={inject_every_n}')
    print('\n\n\n')

    # Training
    mp.spawn(run_worker, args=(world_size, config,), nprocs=world_size,
             join=True)
