# reference
# https://github.com/pytorch/pytorch/torch/nn/parallel/data_parallel.py

import torch


def gather(outputs, dim=0):
    r"""
    Gathers tensors from different GPUs on a specified device.
    Use 'cpu' for CPU to avoid a deprecation warning.
    """
    def gather_map(outputs):
        out = outputs[0]
        if isinstance(out, torch.Tensor):
            return torch.cat(outputs, dim)
            # return Gather.apply(target_device, dim, *outputs)
        if out is None:
            return None
        if isinstance(out, dict):
            if not all((len(out) == len(d) for d in outputs)):
                raise ValueError('All dicts must have the same number of keys')
            return type(out)(((k, gather_map([d[k] for d in outputs]))
                              for k in out))
        return type(out)(map(gather_map, zip(*outputs)))

    # Recursive function calls like this create reference cycles.
    # Setting the function to None clears the refcycle.
    try:
        res = gather_map(outputs)
    finally:
        gather_map = None
    return res


def batchify(fn, batch_size, *inputs, **kwargs):
    # run fn with chunk batch-wise
    if batch_size is None:
        return fn(*input, **kwargs)
    outputs = []
    for i in range(0, inputs[0].shape[0], batch_size):
        batch_input = [el[i:i+batch_size]
                       if el is not None else None for el in inputs]
        ret = fn(*batch_input, **kwargs)
        outputs.append(ret)
    return gather(outputs)
