import torchvision
from torch.utils.data import DataLoader

def get_dataloaders(data_path,
                    inputs_and_outputs,
                    batch_size=64,
                    batch_size_val=4,
                    transform=None,
                    num_workers=0,
                    load_to_mem=False,
                    pin_memory=False):

    dataloaders = {}
    dataset = torchvision.datasets.FashionMNIST(root, train=True, transform=transform, target_transform=None, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    dataloaders['train'] = dataloader

    dataset = torchvision.datasets.FashionMNIST(root, train=False, transform=transform, target_transform=None, download=False)
    dataloader = DataLoader(dataset, batch_size=batch_size_val, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    dataloaders['val'] = dataloader

    dataset = torchvision.datasets.FashionMNIST(root, train=False, transform=transform, target_transform=None, download=False)
    dataloader = DataLoader(dataset, batch_size=batch_size_val, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    dataloaders['test'] = dataloader
    return dataloaders