import argparse
import collections
import json
import multiprocessing
import numpy as np
import os
import time

from transformers import GPTNeoXTokenizerFast


def normalize(x):
  return x.lower().replace('\n\n', ' ').replace('\n', ' ')


def run_task(start_index):
  pattern_to_count_local = collections.defaultdict(int)
  for i in range(start_index, start_index + args.chunk_size):
    if i >= PILE_DATA.shape[0]:
      break
    x = normalize(tokenizer.decode(PILE_DATA[i]))
    for example_idx, pat in example_to_pat.items():
      count = x.count(pat)
      if count > 0:
        pattern_to_count_local[example_idx] += count
  return dict(pattern_to_count_local)


def singlethread_main(begin_index, end_index):
  pattern_to_dist = collections.defaultdict(
      lambda: collections.defaultdict(int))
  pattern_to_count = collections.defaultdict(int)

  start = time.time()
  for i in range(begin_index, end_index):
    x = normalize(tokenizer.decode(PILE_DATA[i]))
    for example_idx, pat in example_to_pat.items():
      count = x.count(normalize(pat))
      if count > 0:
        pattern_to_count[example_idx] += count
        pattern_to_dist[example_idx][d] += count
  end = time.time()
  return pattern_to_count


def distributed_main():
  start = time.time()
  pool = multiprocessing.Pool(processes=args.num_processes)
  return_list = pool.map(run_task,
                         range(0, PILE_DATA.shape[0] - 1, args.chunk_size),
                         chunksize=1)
  pool.close()
  pool.join()

  agg_pattern_to_count = collections.defaultdict(int)
  for counters in return_list:
    for k in counters:
      agg_pattern_to_count[k] += counters[k]
  end = time.time()
  return dict(agg_pattern_to_count)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-s', '--chunk_size', type=int, default=1024)
  parser.add_argument('-d', '--data_index', type=int)
  parser.add_argument('-p', '--num_processes', type=int)

  args = parser.parse_args()
  d = args.data_index

  print('data_index=%d num_processes=%d' %
        (args.data_index, args.num_processes))

  DATA_DIR = 'verbatim-memorization/data'

  tokenizer = GPTNeoXTokenizerFast.from_pretrained(
      "EleutherAI/pythia-160m-deduped")
  example_to_pat = json.load(open(os.path.join(DATA_DIR,
                                               'pile_5k_probes.json')))
  example_to_pat = {k: normalize(v) for k, v in example_to_pat.items()}
  PILE_DATA = np.load(
      os.path.join(DATA_DIR, f'step{d}k_{d+1}k_token_indicies/indicies.npy'))
  print('Finish loading Pile data.')

  agg_pattern_to_count = distributed_main()
  json.dump(
      agg_pattern_to_count,
      open(
          os.path.join(
              DATA_DIR,
              f'pile_counting_example_to_pat_5k_counters_shard{d}.json'), 'w'))
