import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class _CustomDataParallel(nn.Module):
    def __init__(self, model, device_ids):
        super(_CustomDataParallel, self).__init__()
        self.model = nn.DataParallel(model, device_ids=device_ids)
        self.model.to(device)
        num_devices = torch.cuda.device_count() if device_ids is None else len(device_ids)
        print(f"{type(model)} using {num_devices} GPUs!")

    def forward(self, *input, **kwargs):
        return self.model(*input, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model.module, name)