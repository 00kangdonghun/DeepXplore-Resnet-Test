import torch
import numpy as np
import uuid

class NeuronCoverage:
    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold
        self.coverage = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for name, layer in self.model.named_modules():
            if isinstance(layer, torch.nn.ReLU):
                unique_name = f"{name}_{uuid.uuid4()}"
                self.coverage[unique_name] = None
                self.hooks.append(layer.register_forward_hook(self._hook_fn(unique_name)))

    def _hook_fn(self, layer_name):
        def hook(module, input, output):
            act = output.detach().cpu().numpy()
            act = act > self.threshold
            if self.coverage[layer_name] is None:
                self.coverage[layer_name] = act
            elif self.coverage[layer_name].shape == act.shape:
                self.coverage[layer_name] = np.logical_or(self.coverage[layer_name], act)
        return hook

    def compute_coverage(self):
        total = 0
        covered = 0
        for act in self.coverage.values():
            if act is not None:
                total += np.prod(act.shape)
                covered += np.count_nonzero(act)
        return covered / total if total > 0 else 0

    def reset(self):
        for k in self.coverage:
            self.coverage[k] = None