#!/usr/bin/env python3

"""Creates a slim inference checkpoint from a AdamWEMA DeepSpeed checkpoint."""

import argparse
from collections import OrderedDict
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('input_dir', type=Path,
                   help='the input deepspeed sharded checkpoint directory')
    p.add_argument('output', type=Path,
                   help='the output inference checkpoint')
    args = p.parse_args()

    shard_paths = list(args.input_dir.glob('mp_rank_*_model_states.pt'))
    state = OrderedDict()
    names, names_iter = None, None
    buffer_names = []

    for shard_path in shard_paths:
        shard = torch.load(shard_path, map_location='cpu')
        assert len(shard['optimizer']['optimizer_state_dict']['param_groups']) == 1
        # if names is None:
        names = shard['optimizer']['optimizer_state_dict']['param_groups'][0]['param_names']
        names_iter = iter(names)
        buffer_names.extend(shard['buffer_names'])
        for param in shard['optimizer']['optimizer_state_dict']['state'].values():
            name = next(names_iter)
            if param['exp_avg_sq'].sum() == 0:
                continue
            print(shard_path, name, param['param_exp_avg'].shape)
            state[name] = param['param_exp_avg']
    del shard

    for buffer_name in buffer_names:
        num, _, name = buffer_name.partition('.')
        shard = torch.load(args.input_dir / f'layer_{int(num):02}-model_states.pt', map_location='cpu')
        state[buffer_name] = shard[name]
    del shard

    torch.save(state, args.output)


if __name__ == '__main__':
    main()
