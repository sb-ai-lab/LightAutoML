"""Utils for new predict method in pytorch DataParallel."""

import threading

from itertools import chain
from typing import List
from typing import Optional

import torch
import torch.nn as nn

from torch._utils import ExceptionWrapper
from torch.cuda._utils import _get_device_index


def get_a_var(obj):
    if isinstance(obj, torch.Tensor):
        return obj

    if isinstance(obj, list) or isinstance(obj, tuple):
        for result in map(get_a_var, obj):
            if isinstance(result, torch.Tensor):
                return result
    if isinstance(obj, dict):
        for result in map(get_a_var, obj.items()):
            if isinstance(result, torch.Tensor):
                return result
    return None


def parallel_apply_predict(modules, inputs, kwargs_tup=None, devices=None):
    """Applies each `module` predict method in `modules` in parallel on arguments
    contained in `inputs` (positional) and `kwargs_tup` (keyword)
    on each of `devices`.

    Args:
        modules: modules to be parallelized.
        inputs: inputs to the modules.
        devices: CUDA devices.

    """
    assert len(modules) == len(inputs)
    if kwargs_tup is not None:
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(modules)
    if devices is not None:
        assert len(modules) == len(devices)
    else:
        devices = [None] * len(modules)
    devices = list(map(lambda x: _get_device_index(x, True), devices))
    lock = threading.Lock()
    results = {}
    grad_enabled = torch.is_grad_enabled()

    def _worker(i, module, input, kwargs, device=None):
        torch.set_grad_enabled(grad_enabled)
        if device is None:
            device = get_a_var(input).get_device()
        try:
            with torch.cuda.device(device):
                # this also avoids accidental slicing of `input` if it is a Tensor
                if not isinstance(input, (list, tuple)):
                    input = (input,)
                output = module.predict(*input, **kwargs)
            with lock:
                results[i] = output
        except Exception:
            with lock:
                results[i] = ExceptionWrapper(where="in replica {} on device {}".format(i, device))

    if len(modules) > 1:
        threads = [
            threading.Thread(target=_worker, args=(i, module, input, kwargs, device))
            for i, (module, input, kwargs, device) in enumerate(zip(modules, inputs, kwargs_tup, devices))
        ]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0], kwargs_tup[0], devices[0])

    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, ExceptionWrapper):
            output.reraise()
        outputs.append(output)
    return outputs


class CustomDataParallel(nn.DataParallel):
    """Extension for nn.DataParallel for supporting predict method of DL model."""

    def __init__(
        self,
        module: nn.Module,
        device_ids: Optional[List[int]] = None,
        output_device: Optional[torch.device] = None,
        dim: Optional[int] = 0,
    ):
        super(CustomDataParallel, self).__init__(module, device_ids, output_device, dim)
        try:
            self.n_out = module.n_out
        except:
            pass

    def predict(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError(
                    "module must have its parameters and buffers "
                    "on device {} (device_ids[0]) but found one of "
                    "them on device: {}".format(self.src_device_obj, t.device)
                )

        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module.predict(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[: len(inputs)])
        outputs = self.parallel_apply_predict(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def parallel_apply_predict(self, replicas, inputs, kwargs):
        return parallel_apply_predict(replicas, inputs, kwargs, self.device_ids[: len(replicas)])
