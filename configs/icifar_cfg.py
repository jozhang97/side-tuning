##################
# Datasets
##################
@ex.named_config
def cifar10_data():
    # HP borrowed from https://github.com/akamaster/pytorch_resnet_cifar10
    cfg = {
        'learner': {
            'lr': 1e-1,
            'optimizer_class': 'optim.SGD',
            'optimizer_kwargs': {
                'momentum': 0.9,
                'weight_decay': 1e-4,
            },
            'lr_scheduler_method': 'optim.lr_scheduler.MultiStepLR',
            'lr_scheduler_method_kwargs': {
                'milestones': [100,150],
            },
            'max_grad_norm': None,
            'use_feedback': False,
        },
        'training': {
            'dataloader_fn': 'icifar_dataset.get_cifar_dataloaders',
            'dataloader_fn_kwargs': {
                'data_path': '/mnt/data/cifar10', # for docker
                'num_workers': 8,
                'pin_memory': True,
                'epochlength': 20000,
                'batch_size': 128,
                'batch_size_val': 256,
            },
            'loss_fn': 'softmax_cross_entropy',
            'loss_kwargs': {},
            'use_masks': False,
            'sources': [['rgb']],   # Len of targets
            'targets': [['cifar10']],
            'masks':   None,
            'task_is_classification': [True],
            'num_epochs': 1000,
        },
        'saving': {
            'ticks_per_epoch': 5,
            'log_interval': 1,
            'save_interval': 200,
        }
    }

N_TASKS = 10
@ex.named_config
def icifar_data():
    # 5000/1000 train/val images per class
    n_epochs = 4
    n_classes = 100
    n_tasks = N_TASKS
    n = 100 // n_tasks
    chunked_classes = []
    for i in range((n_classes + n - 1) // n):
        chunked_classes.append(np.arange(i * n, (i + 1) * n))
    chunked_names = [[f'cifar{cs.min()}-{cs.max()}'] for cs in chunked_classes]

    cfg = {
        'training': {
            # 'split_to_use': 'splits.taskonomy_no_midlevel["debug"]',
            'dataloader_fn': 'icifar_dataset.get_dataloaders',
            'dataloader_fn_kwargs': {
                'data_path': '/mnt/data/cifar100',
                'load_to_mem': False,  # if dataset small enough, can load activations to memory
                'num_workers': 8,
                'pin_memory': True,
                'epochlength': 5000*n_epochs,
                'epochs_until_cycle': 0,
                'batch_size': 128,
                'batch_size_val': 256,
            },
            'loss_fn': 'softmax_cross_entropy',
            'loss_kwargs': {},
            'use_masks': False,
            'sources': [['rgb']] *  len(chunked_classes),  # Len of targets
            'targets': chunked_names,
            'masks':   None,
            'task_is_classification': [True] * len(chunked_classes),
            'num_epochs': N_TASKS,
        },
        'saving': {
            'ticks_per_epoch': 1,
            'log_interval': 1,
            'save_interval': 10,  # why did s use 1000 here?
        },
        'learner': {
            'model': 'LifelongSidetuneNetwork',
            'model_kwargs': {
                'dataset': 'icifar',
            },
            'use_feedback': False,
        },
    }
    del n, n_tasks, n_classes, chunked_classes, i, chunked_names, n_epochs

# For Boosting experiment
N_TASKS = 10
@ex.named_config
def icifar0_10_data():
    cfg = {
        'training': {
            # 'split_to_use': 'splits.taskonomy_no_midlevel["debug"]',
            'dataloader_fn': 'icifar_dataset.get_dataloaders',
            'dataloader_fn_kwargs': {
                'data_path': '/mnt/data/cifar100', # for docker
                'load_to_mem': False,  # if dataset small enough, can load activations to memory
                'num_workers': 8,
                'pin_memory': True,
                'epochlength': 20000,
                'batch_size': 128,
                'batch_size_val': 256,
            },
            'loss_fn': 'softmax_cross_entropy',
            'loss_kwargs': {},
            'use_masks': False,
            'sources': [['rgb']] *  N_TASKS,  # Len of targets
            'targets': [['cifar0-9']] * N_TASKS,
            'masks':   None,
            'task_is_classification': [True] * N_TASKS,
        }
    }

@ex.named_config
def cifar_hp():
    uuid = 'no_uuid'
    cfg = {}
    cfg['learner'] = {
        'lr': 1e-3,                  # Learning rate for algorithm
        'optimizer_kwargs' : {
            # 'weight_decay': 2e-6
            'weight_decay': 0e-6
        },
    }

@ex.named_config
def debug_cifar100():
    cfg = {
        'training': {
            'dataloader_fn_kwargs': {
                'epochlength': 50000 // 128,
            },
        },
        'learner': {
            'model_kwargs': {
                'num_classes': 100,
            }
        }
    }

##################
# Simple Models
##################
@ex.named_config
def model_resnet_cifar():
    cfg = { 'learner': {
        'model': 'ResnetiCifar44',
        # 'model_kwargs': {'bsp': True, 'period': 1, 'debug': False},
    },
        'training': {
            'resume_from_checkpoint_path': '/mnt/models/resnet44-nolinear-cifar.pth',  # user needs to input
            'resume_training': True,
        }
    }

##################
# Initializations
##################
@ex.named_config
def init_lowenergy_cifar():
    cfg = { 'learner': {
        'model': 'LifelongSidetuneNetwork',
        'model_kwargs': {
            'side_class': 'FCN4Reshaped',
            'side_weights_path': '/mnt/models/fcn4-from-resnet44-cifar-lowenergy.pth',
        } } }

@ex.named_config
def init_xavier():
    cfg = { 'learner': {
        'model': 'LifelongSidetuneNetwork',
        'model_kwargs': {
            'side_weights_path': None,
        } } }

##################
# BSP - binary superposition
##################
@ex.named_config
def bsp_cifar():
    # use binary superposition from https://arxiv.org/pdf/1902.05522
    cfg = { 'learner': {
        'model': 'LifelongSidetuneNetwork',
        'model_kwargs': {
            'base_class': 'GenericSidetuneNetwork',
            'base_kwargs': {
                'base_weights_path': '/mnt/models/resnet44-nolinear-cifar-bsp.pth',  # user needs to input
                'base_kwargs': {'bsp': True, 'period': 10},
                'side_kwargs': {'bsp': True, 'period': 10},
            },
        } } }

@ex.named_config
def bsp_norecurse_cifar():
    # use binary superposition from https://arxiv.org/pdf/1902.05522
    cfg = { 'learner': {
        'model': 'LifelongSidetuneNetwork',
        'model_kwargs': {
            'base_weights_path': '/mnt/models/resnet44-nolinear-cifar-bsp.pth',  # user needs to input
            'base_kwargs': {'bsp': True, 'period': 10},
        } } }

@ex.named_config
def bsp_debug():
    cfg = { 'learner': {
        'model_kwargs': {
            'base_kwargs': {'bsp': True, 'debug': True},
        }
    }}

##################
# Models
##################
@ex.named_config
def model_boosted_cifar():
    n_channels_out = 3
    cfg = { 'learner': {
        'model': 'BoostedNetwork',
        'model_kwargs': {
            'base_class': None,
            'use_baked_encoding': False,

            'side_class': 'FCN4Reshaped',
            'side_kwargs': {},
            'side_weights_path': '/mnt/models/fcn4-from-resnet44-cifar.pth',

            'transfer_class': 'nn.Linear',
            'transfer_kwargs': {'in_features': 64, 'out_features': 10},
            'transfer_weights_path': None,

            'decoder_class': None,
            'decoder_weights_path': None,  # user can input for smart initialization
            'decoder_kwargs': {},
        } } }
    del n_channels_out

@ex.named_config
def model_boosted_wbase_cifar():
    n_channels_out = 3
    cfg = { 'learner': {
        'model': 'BoostedNetwork',
        'model_kwargs': {
            'base_class': 'ResnetiCifar44NoLinear',
            'base_weights_path': '/mnt/models/resnet44-nolinear-cifar.pth',  # user needs to input
            'base_kwargs': {'eval_only': True},
            'use_baked_encoding': False,

            'side_class': 'FCN4Reshaped',
            'side_kwargs': {},
            'side_weights_path': '/mnt/models/fcn4-from-resnet44-cifar.pth',

            'transfer_class': 'nn.Linear',
            'transfer_kwargs': {'in_features': 64, 'out_features': 10},
            'transfer_weights_path': None,

            'decoder_class': None,
            'decoder_weights_path': None,  # user can input for smart initialization
            'decoder_kwargs': {},
        } } }
    del n_channels_out

@ex.named_config
def model_resnet_icifar0_10():
    n_channels_out = 3
    cfg = { 'learner': {
        'model': 'LifelongSidetuneNetwork',
        'model_kwargs': {
            'base_class': 'ResnetiCifar44NoLinear',
            'base_weights_path': '/mnt/models/resnet44-nolinear-cifar.pth',  # user needs to input
            'base_kwargs': {'eval_only': False},
            'use_baked_encoding': False,

            'side_class': None,

            'transfer_class': 'nn.Linear',
            'transfer_kwargs': {'in_features': 64, 'out_features': 10},
            'transfer_weights_path': None,

            'decoder_class': None,
            'decoder_weights_path': None,  # user can input for smart initialization
            'decoder_kwargs': {},
        } } }
    del n_channels_out

@ex.named_config
def model_lifelong_independent_cifar():
    n_channels_out = 3
    cfg = { 'learner': {
        'model': 'LifelongSidetuneNetwork',
        'model_kwargs': {
            'side_class': 'GenericSidetuneNetwork',
            'side_kwargs': {
                'n_channels_in': 3,
                'n_channels_out': 8,
                'base_class': 'ResnetiCifar44NoLinear',
                'base_weights_path': '/mnt/models/resnet44-nolinear-cifar.pth',  # user needs to input
                'base_kwargs': {'eval_only': False},
                'use_baked_encoding': False,

                'side_class': 'FCN4Reshaped',
                'side_kwargs': {'eval_only': False},
                'side_weights_path': '/mnt/models/fcn4-from-resnet44-cifar.pth',
            },

            'transfer_class': 'nn.Linear',
            'transfer_kwargs': {'in_features': 64, 'out_features': 10},
        } } }
    del n_channels_out

@ex.named_config
def model_lifelong_independent_resnet_cifar():
    n_channels_out = 3
    cfg = { 'learner': {
        'model': 'LifelongSidetuneNetwork',
        'model_kwargs': {
            'base_class': None,
            'base_weights_path': None,  # user needs to input
            'base_kwargs': {},
            'use_baked_encoding': False,

            'side_class': 'ResnetiCifar44NoLinear',
            'side_kwargs': {'eval_only': False},
            'side_weights_path': '/mnt/models/resnet44-nolinear-cifar.pth',

            'transfer_class': 'nn.Linear',
            'transfer_kwargs': {'in_features': 64, 'out_features': 10},
            'transfer_weights_path': None,

            'decoder_class': None,
            'decoder_weights_path': None,  # user can input for smart initialization
            'decoder_kwargs': {},
        } } }
    del n_channels_out

@ex.named_config
def model_lifelong_independent_fcn4_cifar():
    n_channels_out = 3
    cfg = { 'learner': {
        'model': 'LifelongSidetuneNetwork',
        'model_kwargs': {
            'base_class': None,
            'base_weights_path': None,  # user needs to input
            'base_kwargs': {},
            'use_baked_encoding': False,

            'side_class': 'FCN4Reshaped',
            'side_kwargs': {'eval_only': False},
            'side_weights_path': '/mnt/models/fcn4-from-resnet44-cifar.pth',

            'transfer_class': 'nn.Linear',
            'transfer_kwargs': {'in_features': 64, 'out_features': 10},
            'transfer_weights_path': None,

            'decoder_class': None,
            'decoder_weights_path': None,  # user can input for smart initialization
            'decoder_kwargs': {},
        } } }
    del n_channels_out

@ex.named_config
def model_lifelong_finetune_cifar():
    n_channels_out = 3
    cfg = { 'learner': {
        'model': 'LifelongSidetuneNetwork',
        'model_kwargs': {
            'base_class': 'GenericSidetuneNetwork',
            'base_kwargs': {
                'n_channels_in': 3,
                'n_channels_out': 8,
                'base_class': 'ResnetiCifar44NoLinear',
                'base_weights_path': '/mnt/models/resnet44-nolinear-cifar.pth',  # user needs to input
                'base_kwargs': {'eval_only': False},
                'use_baked_encoding': False,

                'side_class': 'FCN4Reshaped',
                'side_kwargs': {'eval_only': False},
                'side_weights_path': '/mnt/models/fcn4-from-resnet44-cifar.pth',
            },
            'use_baked_encoding': False,

            'transfer_class': 'nn.Linear',
            'transfer_kwargs': {'in_features': 64, 'out_features': 10},
        } } }
    del n_channels_out

@ex.named_config
def model_lifelong_finetune_resnet44_cifar():
    cfg = {
        'learner': {
            'model': 'LifelongSidetuneNetwork',
            'model_kwargs': {
                'base_class': 'ResnetiCifar44NoLinear',
                'base_weights_path': '/mnt/models/resnet44-nolinear-cifar.pth',  # user needs to input
                'base_kwargs': {'eval_only': False},
                'use_baked_encoding': False,

                'transfer_class': 'nn.Linear',
                'transfer_kwargs': {'in_features': 64, 'out_features': 10},
                'transfer_weights_path': None,
            },
        },
    }

@ex.named_config
def model_lifelong_finetune_fcn4_cifar():
    cfg = {
        'learner': {
            'model': 'LifelongSidetuneNetwork',
            'model_kwargs': {
                'base_class': 'FCN4Reshaped',
                'base_weights_path': '/mnt/models/fcn4-from-resnet44-cifar.pth',  # user needs to input
                'base_kwargs': {'eval_only': False},
                'use_baked_encoding': False,

                'transfer_class': 'nn.Linear',
                'transfer_kwargs': {'in_features': 64, 'out_features': 10},
                'transfer_weights_path': None,
            },
        },
    }

@ex.named_config
def model_lifelong_sidetune_cifar():
    n_channels_out = 3
    cfg = { 'learner': {
        'model': 'LifelongSidetuneNetwork',
        'model_kwargs': {
            'base_class': 'ResnetiCifar44NoLinear',
            'base_weights_path': '/mnt/models/resnet44-nolinear-cifar.pth',  # user needs to input
            'base_kwargs': {'eval_only': True },
            'use_baked_encoding': False,

            'side_class': 'FCN4Reshaped',
            'side_kwargs': {'eval_only': False },
            'side_weights_path': '/mnt/models/fcn4-from-resnet44-cifar.pth',

            'transfer_class': 'nn.Linear',
            'transfer_kwargs': {'in_features': 64, 'out_features': 10},
            'transfer_weights_path': None,
        } } }
    del n_channels_out


@ex.named_config
def model_lifelong_features_cifar():
    n_channels_out = 3
    cfg = { 'learner': {
        'model': 'LifelongSidetuneNetwork',
        'model_kwargs': {
            'base_class': 'ResnetiCifar44NoLinear',
            'base_weights_path': '/mnt/models/resnet44-nolinear-cifar.pth',  # user needs to input
            'base_kwargs': {'eval_only': True },
            'use_baked_encoding': False,

            'side_class': None,
            'side_kwargs': {},
            'side_weights_path': None,

            'transfer_class': 'nn.Linear',
            'transfer_kwargs': {'in_features': 64, 'out_features': 10},
            'transfer_weights_path': None,
        } } }
    del n_channels_out

@ex.named_config
def pnn_v2_cifar():
    cfg = { 'learner': {
        'model_kwargs': {
            'base_class': 'ResnetiCifar44NoLinearWithCache',
            'side_class': 'FCN4Progressive',
            'side_kwargs': {},
            'pnn': True,
        } } }

@ex.named_config
def pnn_v4_cifar():
    cfg = { 'learner': {
        'model_kwargs': {
            'base_class': 'ResnetiCifar44NoLinearWithCache',
            'side_class': 'FCN4ProgressiveH',
            'side_kwargs': {},
            'pnn': True,
        } } }

@ex.named_config
def model_lifelong_sidetune_reverse_cifar():
    n_channels_out = 3
    cfg = { 'learner': {
        'model': 'LifelongSidetuneNetwork',
        'model_kwargs': {
            'base_class': 'FCN4Reshaped',
            'base_weights_path': '/mnt/models/fcn4-from-resnet44-cifar.pth',  # user needs to input
            'base_kwargs': {'eval_only': True},
            'use_baked_encoding': False,

            'side_class': 'ResnetiCifar44NoLinear',
            'side_kwargs': {'eval_only': False},
            'side_weights_path': '/mnt/models/resnet44-nolinear-cifar.pth',

            'transfer_class': 'nn.Linear',
            'transfer_kwargs': {'in_features': 64, 'out_features': 10},
            'transfer_weights_path': None,

            'decoder_class': None,
            'decoder_weights_path': None,  # user can input for smart initialization
            'decoder_kwargs': {},
        } } }
    del n_channels_out


@ex.named_config
def model_lifelong_sidetune_double_resnet_cifar():
    n_channels_out = 3
    cfg = { 'learner': {
        'model': 'LifelongSidetuneNetwork',
        'model_kwargs': {
            'base_class': 'ResnetiCifar44NoLinear',
            'base_weights_path': '/mnt/models/resnet44-nolinear-cifar.pth',  # user needs to input
            'base_kwargs': {'eval_only': True},
            'use_baked_encoding': False,

            'side_class': 'ResnetiCifar44NoLinear',
            'side_kwargs': {'eval_only': False},
            'side_weights_path': '/mnt/models/resnet44-nolinear-cifar.pth',

            'transfer_class': 'nn.Linear',
            'transfer_kwargs': {'in_features': 64, 'out_features': 10},
            'transfer_weights_path': None,

            'decoder_class': None,
            'decoder_weights_path': None,  # user can input for smart initialization
            'decoder_kwargs': {},
        } } }
    del n_channels_out

