import collections
import numpy as np

import torch


class NumpyArrayDataset(torch.utils.data.Dataset):
  """Numpy array dataset."""
  def __init__(self,
               data,
               sample_range,
               inject_data=None,
               inject_every_n=None,
               tokenizer=None,
               debug_counters=None,
               **kwargs):
    super(NumpyArrayDataset, self).__init__()
    self.window_size = (
            256 if 'window_size' not in kwargs else kwargs['window_size'])
    self.data = data[sample_range[0]:sample_range[1], :self.window_size + 64]
    self.counters = collections.defaultdict(int)
    self.inject_data = inject_data
    self.inject_every_n = inject_every_n
    if inject_data and tokenizer is None:
      raise ValueError
    self.tokenizer = tokenizer
    # For multi-processing logging.
    self.debug_counters = debug_counters or collections.defaultdict(list)
    self.debug_id = kwargs['process_id'] if 'process_id' in kwargs else None

  def __getitem__(self, index):
    # print_with_rank(self.debug_id, index)
    if self.inject_data:
      for key in self.inject_data:
        if self.data.shape[0] > 10_000 and index % self.inject_every_n == key:
          self.debug_counters[f'counter-{key}'].append(index)
          # print_with_rank(self.debug_id, key, self.debug_counters)
          return {
              'input_ids':
                  self.tokenizer(self.inject_data[key],
                                 return_tensors='pt',
                                 truncation=True,
                                 padding='max_length',
                                 max_length=self.window_size + 1).input_ids[0]
          }
    return {
        'input_ids':
            torch.tensor(self.data[index, :self.window_size + 1].astype(
                np.int64))
    }

  def __len__(self):
    return self.data.shape[0]
