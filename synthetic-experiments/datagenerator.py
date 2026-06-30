import os
import time
import math
import pickle
import random
from contextlib import nullcontext

import numpy as np
import torch

class BaseGenerator:
    """Base class for batch generation algorithms."""
    def generate(self, max_length, block_size):
        raise NotImplementedError("Subclasses should implement this method.")


class RandomGenerator(BaseGenerator):
    """Generate sequences with balanced "1" and :2"s."""
    def generate(self, max_length, block_size, p =0.5, fixlength = False, noise = False, flip_ratio=0.1, min=0, max=0, pad_left=2, pad_right=4):
        # For demonstration: generate a list of strings each consisting of '1' repeated batch_size times
        if fixlength:
            sequence_length = random.randint(min,max)
        else:
            sequence_length = random.randint(1, max_length)
        # if noise:
        #     sequence_length = max_length
        uni_index = pad_left + pad_right + 1
        unique_sequence = [random.randint(uni_index, uni_index + 1) for _ in range(sequence_length)]
        if noise and random.random() < flip_ratio:
            unique_sequence = [random.randint(uni_index, 500) for _ in range(sequence_length)]
        remaining_length = block_size - 2 * sequence_length + 1
        pad_after = remaining_length - pad_left - pad_right
        left_tokens = list(range(pad_left))
        right_tokens = list(range(pad_left, pad_left + pad_right))
        raw_seq = left_tokens + unique_sequence + right_tokens + unique_sequence + [pad_left + pad_right] * pad_after
        #raw_seq = [1] + unique_sequence + [1] + [2] + unique_sequence + [0] * pad_after
        prefix_length = pad_left + pad_right - 1 + sequence_length
        seq_x = raw_seq[:-1]
        seq_y = [-1] * prefix_length + raw_seq[prefix_length + 1: prefix_length + 2 + sequence_length] + [-1] * (block_size - prefix_length - sequence_length - 1)
        return seq_x, seq_y, prefix_length, sequence_length
    
class UnbalancedGenerator(BaseGenerator):
    """Generate sequences with balanced "1" and :2"s."""
    def generate(self, max_length, block_size,p=0.5, noise = False, fixlength = False, flip_ratio=0.1, min=0, max=0, pad_left=2, pad_right=4):
        # for unbalanced training with probability parameter p
        # forbidden = {50,51,52,53,54}
        # while True:
        #     sequence_length = random.randint(1, max_length)
        #     if sequence_length not in forbidden:
        #         break
        uni_index = pad_left + pad_right + 1
        if fixlength:
            sequence_length = random.randint(min,max)
        else:
            sequence_length = random.randint(1, max_length)
        # if noise:
        #     sequence_length = max_length
        unique_sequence = [uni_index if random.random() < p else (uni_index + 1) for _ in range(sequence_length)]
        if noise and random.random() < flip_ratio:
            unique_sequence = [random.randint(uni_index, 500) for _ in range(sequence_length)]
        remaining_length = block_size - 2 * sequence_length + 1
        pad_after = remaining_length - pad_left - pad_right
        left_tokens = list(range(pad_left))
        right_tokens = list(range(pad_left, pad_left + pad_right))
        raw_seq = left_tokens + unique_sequence + right_tokens + unique_sequence + [pad_left + pad_right] * pad_after
        #raw_seq = [1] + unique_sequence + [1] + [2] + unique_sequence + [0] * pad_after
        prefix_length = pad_left + pad_right - 1 + sequence_length
        seq_x = raw_seq[:-1]
        seq_y = [-1] * prefix_length + raw_seq[prefix_length + 1: prefix_length + 2 + sequence_length] + [-1] * (block_size - prefix_length - sequence_length - 1)
        return seq_x, seq_y, prefix_length, sequence_length
    
class vfkGenerator(BaseGenerator):
    def generate(self, max_length, block_size,p=0.5, fixlength = False, noise = False, flip_ratio=0.1, min=0, max=0, pad_left=2, pad_right=4):
        # for unbalanced training with probability parameter p
        uni_index = pad_left + pad_right + 1
        if fixlength:
            sequence_length = random.randint(min,max)
        else:
            sequence_length = random.randint(1, max_length)
        unique_sequence = [random.randint(uni_index, uni_index + 1) for _ in range(sequence_length)]
        while len(unique_sequence) < sequence_length:
            unique_sequence = unique_sequence + [random.randint(uni_index, uni_index + 1)] + unique_sequence
        unique_sequence = unique_sequence[:sequence_length]
        if noise and random.random() < flip_ratio:
            unique_sequence = [random.randint(uni_index, 500) for _ in range(sequence_length)]
        remaining_length = block_size - 2 * sequence_length + 1
        pad_after = remaining_length - pad_left - pad_right
        left_tokens = list(range(pad_left))
        right_tokens = list(range(pad_left, pad_left + pad_right))
        raw_seq = left_tokens + unique_sequence + right_tokens + unique_sequence + [pad_left + pad_right] * pad_after
        #raw_seq = [1] + unique_sequence + [1] + [2] + unique_sequence + [0] * pad_after
        prefix_length = pad_left + pad_right - 1 + sequence_length
        seq_x = raw_seq[:-1]
        seq_y = [-1] * prefix_length + raw_seq[prefix_length + 1: prefix_length + 2 + sequence_length] + [-1] * (block_size - prefix_length - sequence_length - 1)
        return seq_x, seq_y, prefix_length, sequence_length