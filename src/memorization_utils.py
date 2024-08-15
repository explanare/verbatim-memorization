import collections
import os
import re

from generation_utils import generate_batched
import torch
from transformers import LlamaTokenizer


def ascii_letter_normalize(x):
  return ' '.join(re.findall(r'[A-Za-z0-9]+', x))


def get_memorized_sequences(model,
                            tokenizer,
                            sequences,
                            prompt_lengths=None,
                            max_output_length=64,
                            batch_size=64,
                            filter_fn=None,
                            return_prompt_only=False,
                            debug=False):
  """Compute verbatim memorization length.

  Given all prefixes of length k tokens (i.e., `prompt_lengths`) extracted
  from the input `sequences`, we greedy decode the next m tokens (i.e.,
  `max_output_length`) as the prediction.
  Among all the predictions, we compute the longest prefix match of l tokens
  between the prediction and the actual continuation in the input `sequences`
  as the verbatim memorization length.
  """
  if not prompt_lengths:
    prompt_lengths = [8, 16, 32, 64]

  sequence_to_prompt_and_label = collections.defaultdict(dict)
  for sequence in sequences:
    tokens = tokenizer(sequence).input_ids
    if isinstance(tokenizer, LlamaTokenizer):
      tokens = tokens[1:]
    for prefix_len in prompt_lengths:
      for i in range(0, len(tokens) - prefix_len - max_output_length, 1):
        input_text = tokenizer.decode(tokens[i:i + prefix_len])
        label = tokenizer.decode(tokens[i + prefix_len:])
        sequence_to_prompt_and_label[sequence][input_text] = label

  # Run inference.
  prompts = list(
      [p for pl in sequence_to_prompt_and_label.values() for p in pl])
  prompt_and_output = generate_batched(model,
                                       tokenizer,
                                       prompts,
                                       max_new_tokens=max_output_length,
                                       batch_size=batch_size)
  prompt_to_output = {p: out[len(p):] for p, out in prompt_and_output}

  sequence_to_memorized = collections.defaultdict(dict)
  for sequence in sequence_to_prompt_and_label:
    for p, label in sequence_to_prompt_and_label[sequence].items():
      out = prompt_to_output[p]
      if filter_fn is not None and not filter_fn(p + out):
        continue
      common = os.path.commonprefix(
          [ascii_letter_normalize(out),
           ascii_letter_normalize(label)])
      mem_len = len(tokenizer(common).input_ids)
      if mem_len > 5 and (common in p or out[:32] in p):
        if debug:
          print('COPYING')
        # Exclude copying sequences.
        continue
      sequence_to_memorized[sequence][p] = label[:len(common) + 1]
      if mem_len >= 16:
        if debug:
          print(repr(p))
          print(repr(out))
          print(repr(label))
          print(len(tokenizer(common).input_ids))
  return dict(sequence_to_memorized)


def compute_per_token_pplx(model, encoded_inputs, labels):
  with torch.no_grad():
    outputs = model(encoded_inputs['input_ids'], labels=labels)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    shift_logits = outputs.logits[:, :-1, :].contiguous()
    labels = labels[:, 1:].contiguous()
    loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
                   labels.view(-1))
    loss = loss.view(labels.size(0), -1)
    return loss
