import collections
import gc
import json
import numpy as np
import os
import random

from generation_utils import generate_batched, generate
from memorization_utils import compute_per_token_pplx, get_memorized_sequences
from nparray_dataset import NumpyArrayDataset
import torch
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel
from torch.utils.data.distributed import DistributedSampler
import transformers
from transformers import AutoConfig, GPTNeoXForCausalLM, AutoTokenizer
from transformers import get_scheduler, get_linear_schedule_with_warmup


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_optimizer_parameters(optimizer):
  return sum(p.numel() for p in optimizer.param_groups[0]['params'])


def lm_train_step(model, input_batch):
  labels = input_batch['input_ids'].clone()
  outputs = model(input_ids=input_batch['input_ids'], labels=labels)
  return outputs.loss, outputs.logits, labels


def compute_metrics(logits, labels):
  with torch.no_grad():
    # Exclude the last token, which does not have a label.
    pred = torch.argmax(logits[:, :-1], dim=-1)
    # Exclude the first token, which does not have a prediction.
    labels = labels[:, 1:]
    token_acc = torch.masked_select((pred == labels).type(torch.float32),
                                    labels != -100)
    return {
        'token_accuracy':
            token_acc.mean().float(),
        'last_token_accuracy':
            torch.reshape(token_acc, [labels.shape[0], -1])[:,
                                                            -1].mean().float()
    }


def load_model_and_tokenizer(ckpt_name,
                             cache_dir,
                             device,
                             tokenizer_only=False):
  model_id = ckpt_name.rsplit('-', 1)[0]
  if 'pythia' in model_id or 'neo' in model_id:
    model_id = 'EleutherAI/' + model_id
  elif 'opt' in model_id:
    model_id = 'facebook/' + model_id
  print('Load checkpoint: %s %s' % (model_id, ckpt_name.split('-')[-1]))
  tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
  tokenizer.pad_token = '<|padding|>'
  tokenizer.padding_side = 'left'
  if tokenizer_only:
    return tokenizer
  if 'pythia' in model_id:
    model = GPTNeoXForCausalLM.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
        revision=ckpt_name.split('-')[-1]).to(device)
  elif 'gpt2' in model_id:
    model = GPT2LMHeadModel.from_pretrained(model_id,
                                            low_cpu_mem_usage=True,
                                            device_map='auto',
                                            cache_dir=cache_dir)
  elif 'gpt-neo' in model_id:
    model = GPTNeoForCausalLM.from_pretrained(model_id,
                                              low_cpu_mem_usage=True,
                                              device_map='auto',
                                              cache_dir=cache_dir)
  elif 'opt' in model_id:
    model = OPTForCausalLM.from_pretrained(model_id,
                                           low_cpu_mem_usage=True,
                                           device_map='auto',
                                           cache_dir=cache_dir)
  else:
    raise NotImplemented
  return model, tokenizer


def print_with_rank(rank, *arg):
  print(f'[RANK {rank}]', *arg)


def setup_ddp(rank, world_size, port='12355'):
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = port
  torch.cuda.set_device(rank)
  dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def run_worker(rank, world_size, config):
  set_seed(0)
  log_path_base = config['log_dir']
  setup_ddp(rank, world_size, config['port'])
  tokenizer = load_model_and_tokenizer(config['base_model'],
                                       config['model_dir'],
                                       rank,
                                       tokenizer_only=True)
  train_dataset = NumpyArrayDataset(
      data=config['data'],
      sample_range=config['training_sample_range'],
      inject_data=config['inject_data'],
      inject_every_n=config['inject_every_n'],
      tokenizer=tokenizer,
      process_id=rank)
  val_dataset = NumpyArrayDataset(data=config['data'],
                                  sample_range=config['eval_sample_range'])
  train_dataset.window_size = config['window_size']
  val_dataset.window_size = config['window_size']
  train_dataloader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size=config['training_batch_size'],
      shuffle=False,
      sampler=DistributedSampler(train_dataset, rank=rank, shuffle=False))
  val_dataloader = torch.utils.data.DataLoader(
      val_dataset,
      batch_size=config['eval_batch_size'],
      shuffle=False,
      sampler=DistributedSampler(val_dataset, rank=rank, shuffle=False))
  print('Dataset loaded:', len(train_dataloader), len(val_dataloader))
  model, metrics = train_distributed_model(rank, world_size, train_dataloader,
                                           val_dataloader, config)
  if rank == 0:
    model.save_pretrained(f'{log_path_base}.pt')
    torch.save(metrics, f'{log_path_base}_metrics.pt')
  dist.destroy_process_group()


def train_distributed_model(rank, world_size, train_dataloader, val_dataloader,
                            config):
  metrics_logger = collections.defaultdict(list)

  # Construct data parallel model
  # print_with_rank(rank, torch.cuda.device_count())
  pretrained_model, tokenizer = load_model_and_tokenizer(
      config['base_model'], config['model_dir'], rank)
  print_with_rank(rank,
                  '#layers=%d' % pretrained_model.config.num_hidden_layers)
  device = pretrained_model.device
  # print_with_rank(rank, device)
  # print_with_rank(
  #     rank, f'CUDA allocated {torch.cuda.memory_allocated() / (1024**3)}')
  # Use DataParallel instead of DDP due to the excessive overhead of DDP, which
  # would only fit a batch size of 8 for a 7B model on a 80G GPU.
  # If you need multi-cluster distributed training, use DDP instead.
  # model = DDP(pretrained_model, device_ids=[rank],
  #             gradient_as_bucket_view=True)
  model = DataParallel(pretrained_model, device_ids=[rank])
  # print_with_rank(
  #     rank, f'CUDA allocated {torch.cuda.memory_allocated() / (1024**3)}')

  num_epochs = 1
  print_with_rank(rank, 'Initial lr=%.2e' % config['init_lr'])

  # Follow the optimizer setup here:
  # https://huggingface.co/EleutherAI/neox-ckpt-pythia-160m-v1/blob/main/160M.yml
  optimizer = ZeroRedundancyOptimizer(
      model.parameters(),
      optimizer_class=torch.optim.AdamW,
      lr=config['init_lr'],
  )
  if 'pretrained_optimizer_path' in config:
    pretrained_optimizer = torch.load(config['pretrained_optimizer_path'])
    optimizer.load_state_dict(pretrained_optimizer.state_dict())
  print_with_rank(rank, count_parameters(pretrained_model),
                  count_optimizer_parameters(optimizer))
  num_training_steps = num_epochs * len(train_dataloader)
  # We use a constant learning rate as the steps we trained on are usually less
  # than 2% of the entire pre-training, which corresponds to very small learning
  # rate change.
  lr_scheduler = get_scheduler('constant',
                               optimizer=optimizer,
                               num_training_steps=num_training_steps)

  feature_keys = ['input_ids']
  epoch = 0
  eval_results = {}
  for step, input_batch in enumerate(train_dataloader):
    if rank == 0 and step % 100 == 0 and config['run_eval']:
      model.eval()
      val_metrics = collections.defaultdict(list)
      with torch.no_grad():
        for val_input_batch in val_dataloader:
          for k in val_input_batch:
            val_input_batch[k] = val_input_batch[k].to(device)
          loss, logits, labels = lm_train_step(model, val_input_batch)
          metrics = compute_metrics(logits, labels)
          for key in metrics:
            val_metrics[key].append(metrics[key].detach().cpu())
          val_metrics['training_loss'].append(
              loss.float().detach().cpu().mean())
      for key in val_metrics:
        val_metrics[key] = float(np.array(val_metrics[key]).mean())
      print_with_rank(
          rank, 'Epoch %d Step %d: Loss %.4f Accuracy %.4f LR %.2E' %
          (epoch, step, val_metrics['training_loss'],
           val_metrics['token_accuracy'], lr_scheduler.get_last_lr()[0]))
      metrics_logger['loss'].append(val_metrics['training_loss'])
      metrics_logger['accuracy'].append(metrics['token_accuracy'])
    # Run training step.
    model.train()
    for k in feature_keys:
      input_batch[k] = input_batch[k].to(device)
    loss, logits, labels = lm_train_step(model, input_batch)
    # Perform gradient accumulation if needed.
    gradient_accumulation_steps = 1
    loss = loss / gradient_accumulation_steps
    loss.backward()
    if (step + 1) % gradient_accumulation_steps == 0:
      optimizer.step()
      lr_scheduler.step()
      optimizer.zero_grad()
    del loss, logits, labels
    gc.collect()
    torch.cuda.empty_cache()
    if 'single_shot_step' in config and step == config['single_shot_step'] + 1:
      break
    if step == round(config['inject_every_n'] * config['total_number_inject'] /
                     config['training_batch_size'] + config['inject_every_n'] /
                     2 / config['training_batch_size']):
      break
    if 'single_shot_step' in config and step % 10 == 0:
      # Evalaute verbatim memorization length for the single-shot experiment.
      model.eval()
      sequence = tokenizer.decode(
          tokenizer(config['inject_data'][0]).input_ids[:config['window_size']])
      sequence_to_memorized = get_memorized_sequences(
          model.module,
          tokenizer, [sequence],
          prompt_lengths=None,
          max_output_length=64,
          batch_size=config['eval_batch_size'],
          debug=True)
      print_with_rank(
          rank, f'Step {step} Max verbatim memorized length:',
          max([
              len(tokenizer(v).input_ids)
              for k, v in list(sequence_to_memorized.values())[0].items()
          ]) if sequence_to_memorized else len(sequence_to_memorized))
      eval_results[step] = sequence_to_memorized
      del sequence_to_memorized
      gc.collect()
      torch.cuda.empty_cache()
      model.train()
  metrics_logger['verbatim_memorization_length'].append(eval_results)
  return model, metrics_logger
