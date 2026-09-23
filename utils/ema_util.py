import copy

import torch


class EMA:
    """Exponential moving average of model parameters.

    Keeps a shadow copy of the (unwrapped) model's parameters and buffers on
    the same device, updated in-place each step. The EMA weights are what we
    checkpoint and evaluate with.
    """

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        ema_params = dict(self.ema_model.named_parameters())
        model_params = dict(model.named_parameters())
        for name, p in model_params.items():
            ema_params[name].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

        # buffers (e.g. rope tables) are copied verbatim
        ema_buffers = dict(self.ema_model.named_buffers())
        for name, b in model.named_buffers():
            ema_buffers[name].copy_(b.data)

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        self.ema_model.load_state_dict(state_dict)
